from __future__ import absolute_import

import os.path
import numpy as np
from skimage import draw
from shapely.geometry import Polygon, asPolygon
from shapely.prepared import prep
from shapely.ops import unary_union

from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml, PageType
)
from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    coordinates_for_segment,
    points_from_polygon,
    MIMETYPE_PAGE
)

from .. import get_ocrd_tool
from .ocrolib import midrange
from .common import (
    pil2array,
    # DSAVE,
    # binarize,
    check_page,
    check_region,
    compute_segmentation
    #borderclean_bin
)
from .segment import (
    masks2polygons,
    polygon_for_parent,
    make_valid,
    make_intersection
)

TOOL = 'ocrd-cis-ocropy-resegment'

class OcropyResegment(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyResegment, self).__init__(*args, **kwargs)

    def process(self):
        """Resegment lines of the workspace.

        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the line level.

        Next, get the image according to the layout annotation (from
        the alternative image of the page, or by cropping from annotated
        Border and rotating from annotated orientation), and compute a new
        line segmentation for that (as a label mask, suppressing all non-text
        regions' foreground), and polygonalize its contours.

        Then calculate overlaps between the new and existing lines, i.e.
        which existing line polygons (or rectangles) contain most of each
        new line polygon: Among the existing lines covering most of each
        new line's foreground and background area, assign the one with the
        largest share of the existing line. Next, for each existing line,
        calculate the hull polygon of all assigned new lines, and if the
        foreground and background overlap is sufficient, and no overlapping
        but yet unassigned lines would be lost, then annotate that polygon
        as new coordinates.
        Thus, at the end, all new and existing lines will have been used
        at most once, but not all existing lines might have been resegmented
        (either because there were no matches at all, or the loss would have
        been too large, either by fg/bg share or by unassigned line labels).

        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.OcropyResegment')
        # This makes best sense for bad/coarse line segmentation, like current GT
        # or as postprocessing for bbox-only steps like Tesseract.
        # Most notably, it can convert rectangles to polygons (polygonalization),
        # and enforce conflicting lines shared between neighbouring regions.
        # It depends on a decent line segmentation from ocropy though. So it
        # _should_ ideally be run after deskewing (on the page level),
        # _must_ be run after binarization (on page level). Also, the method's
        # accuracy crucially depends on a good estimate of the images'
        # pixel density (at least if source input is not 300 DPI).
        level = self.parameter['level-of-operation']
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            file_id = make_file_id(input_file, self.output_file_grp)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page_id = pcgts.pcGtsId or input_file.pageId or input_file.ID # (PageType has no id)
            page = pcgts.get_Page()
            
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id, feature_selector='binarized')
            if self.parameter['dpi'] > 0:
                zoom = 300.0/self.parameter['dpi']
            elif page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi *= 2.54
                LOG.info('Page "%s" uses %f DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1

            ignore = (page.get_ImageRegion() +
                      page.get_LineDrawingRegion() +
                      page.get_GraphicRegion() +
                      page.get_ChartRegion() +
                      page.get_MapRegion() +
                      page.get_MathsRegion() +
                      page.get_ChemRegion() +
                      page.get_MusicRegion() +
                      page.get_AdvertRegion() +
                      page.get_NoiseRegion() +
                      page.get_SeparatorRegion() +
                      page.get_UnknownRegion() +
                      page.get_CustomRegion())
            regions = page.get_AllRegions(classes=['Text'], order='reading-order')
            if not regions:
                LOG.warning('Page "%s" contains no text regions', page_id)
            elif level == 'page':
                lines = [line for region in regions
                         for line in region.get_TextLine()]
                if lines:
                    self._process_segment(page, page_image, page_coords, page_id, zoom, lines, ignore)
                else:
                    LOG.warning('Page "%s" contains no text regions with lines', page_id)
            else:
                for region in regions:
                    lines = region.get_TextLine()
                    if lines:
                        region_image, region_coords = self.workspace.image_from_segment(
                            region, page_image, page_coords, feature_selector='binarized')
                        self._process_segment(region, region_image, region_coords, page_id, zoom, lines, ignore)
                    else:
                        LOG.warning('Page "%s" region "%s" contains no text lines', page_id, region.id)

            # update METS (add the PAGE file):
            file_path = os.path.join(self.output_file_grp, file_id + '.xml')
            pcgts.set_pcGtsId(file_id)
            out = self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
            LOG.info('created file ID: %s, file_grp: %s, path: %s',
                     file_id, self.output_file_grp, out.local_filename)

    def _process_segment(self, parent, parent_image, parent_coords, page_id, zoom, lines, ignore):
        LOG = getLogger('processor.OcropyResegment')
        threshold = self.parameter['min_fraction']
        margin = self.parameter['extend_margins']
        # prepare line segmentation
        parent_array = pil2array(parent_image)
        #parent_array, _ = common.binarize(parent_array, maxskew=0) # just in case still raw
        parent_bin = np.array(parent_array <= midrange(parent_array), np.bool)
        ignore_bin = np.ones_like(parent_bin, np.bool)
        if isinstance(parent, PageType):
            tag = 'page'
            fullpage = True
            report = check_page(parent_bin, zoom)
        else:
            tag = 'region'
            fullpage = False
            report = check_region(parent_bin, zoom)
        try:
            if report:
                raise Exception(report)
            # draw.polygon: If any segment_polygon lies outside of parent
            # (causing negative/above-max indices), either fully or partially,
            # then this will silently ignore them. The caller does not need
            # to concern herself with this.
            # get existing line labels:
            line_labels = np.zeros_like(parent_bin, np.bool)
            line_labels = np.tile(np.expand_dims(line_labels, -1), (1,1,len(lines)))
            for i, segment in enumerate(lines):
                segment_polygon = coordinates_of_segment(segment, parent_image, parent_coords)
                segment_polygon = np.array(make_valid(Polygon(segment_polygon)).buffer(margin).exterior, np.int)[:-1]
                segment_y, segment_x = draw.polygon(segment_polygon[:, 1],
                                                    segment_polygon[:, 0],
                                                    parent_bin.shape)
                line_labels[segment_y, segment_x, i] = True
            # only text region(s) may contain new text lines
            for i, segment in enumerate(set(line.parent_object_ for line in lines)):
                LOG.debug('unmasking area of text region "%s" for "%s"',
                          segment.id, page_id)
                segment_polygon = coordinates_of_segment(segment, parent_image, parent_coords)
                ignore_bin[draw.polygon(segment_polygon[:, 1],
                                        segment_polygon[:, 0],
                                        parent_bin.shape)] = False
            # mask/ignore overlapping neighbours
            for i, segment in enumerate(ignore):
                LOG.debug('masking area of %s "%s" for "%s"',
                          type(segment).__name__[:-4], segment.id, page_id)
                segment_polygon = coordinates_of_segment(segment, parent_image, parent_coords)
                ignore_bin[draw.polygon(segment_polygon[:, 1],
                                        segment_polygon[:, 0],
                                        parent_bin.shape)] = True
            new_line_labels, _, _, _, _, scale = compute_segmentation(
                parent_bin, seps=ignore_bin, zoom=zoom, fullpage=fullpage,
                maxseps=0, maxcolseps=0, maximages=0)
        except Exception as err:
            LOG.warning('Cannot line-segment %s "%s": %s',
                        tag, page_id if fullpage else parent.id, err)
            return
        LOG.info("Found %d new line labels for %d existing lines on %s '%s'",
                 new_line_labels.max(), len(lines), tag, page_id if fullpage else parent.id)
        # polygonalize and prepare comparison
        new_line_polygons, new_line_labels = masks2polygons(
            new_line_labels, parent_bin,
            '%s "%s"' % (tag, page_id if fullpage else parent.id),
            min_area=640/zoom/zoom)
        # DSAVE('line_labels', [np.mean(line_labels, axis=2), parent_bin])
        # DSAVE('new_line_labels', [new_line_labels, parent_bin], disabled=False)
        new_line_polygons = [make_valid(Polygon(line_poly))
                             for line_label, line_poly in new_line_polygons]
        line_polygons = [prep(make_valid(Polygon(coordinates_of_segment(
            line, parent_image, parent_coords))).buffer(margin))
                         for line in lines]
        # polygons for intersecting pairs
        intersections = dict()
        # ratio of overlap between intersection and new line
        fits_bg = np.zeros((len(new_line_polygons), len(line_polygons)), np.float)
        fits_fg = np.zeros((len(new_line_polygons), len(line_polygons)), np.float)
        # ratio of overlap between intersection and existing line
        covers_bg = np.zeros((len(new_line_polygons), len(line_polygons)), np.float)
        covers_fg = np.zeros((len(new_line_polygons), len(line_polygons)), np.float)
        # compare segmentations, calculating ratios of overlapping fore/background area
        for i, new_line_poly in enumerate(new_line_polygons):
            for j, line_poly in enumerate(line_polygons):
                # too strict: .contains
                if line_poly.intersects(new_line_poly):
                    inter = make_intersection(line_poly.context, new_line_poly)
                    if not inter:
                        continue
                    new_line_mask = (new_line_labels == i+1) & parent_bin
                    line_mask = line_labels[:,:,j] & parent_bin
                    inter_mask = new_line_mask & line_mask
                    if (not np.count_nonzero(inter_mask) or
                        not np.count_nonzero(new_line_mask) or
                        not np.count_nonzero(line_mask)):
                        continue
                    intersections[(i, j)] = inter
                    fits_bg[i, j] = inter.area / new_line_poly.area
                    covers_bg[i, j] = inter.area / line_poly.context.area
                    fits_fg[i, j] = np.count_nonzero(inter_mask) / np.count_nonzero(new_line_mask)
                    covers_fg[i, j] = np.count_nonzero(inter_mask) / np.count_nonzero(line_mask)
                    # LOG.debug("new %d old %d (%s): %.1f%% / %.1f%% bg, %.1f%% / %.1f%% fg",
                    #           i, j, lines[j].id,
                    #           fits_bg[i,j]*100, covers_bg[i,j]*100,
                    #           fits_fg[i,j]*100, covers_fg[i,j]*100)
        # assign new lines to existing lines, if possible
        assignments = np.ones(len(new_line_polygons), np.int) * -1
        for i, new_line_poly in enumerate(new_line_polygons):
            if not fits_bg[i].any():
                LOG.debug("new line %d fits no existing line's background", i)
                continue
            if not fits_fg[i].any():
                LOG.debug("new line %d fits no existing line's foreground", i)
                continue
            fits = (fits_bg[i] > 0.6) & (fits_fg[i] > 0.9)
            if not fits.any():
                j = np.argmax(fits_bg[i] * fits_fg[i])
                LOG.debug("best fit '%s' for new line %d covers only %.1f%% bg / %.1f%% fg",
                          lines[j].id,
                          i, fits_bg[i,j] * 100, fits_fg[i,j] * 100)
                continue
            covers = covers_bg[i] * covers_fg[i] * fits
            j = np.argmax(covers)
            line = lines[j]
            inter_polygon = intersections[(i,j)]
            new_line_polygon = new_line_polygons[i]
            new_center = inter_polygon.centroid
            center = new_line_polygon.centroid
            # FIXME: apply reasonable threshold for centroid distance
            LOG.debug("new line for '%s' has centroid distance %.2f",
                      line.id, center.distance(new_center))
            assignments[i] = j
        # validate assignments retain enough area and do not loose unassigned matches
        for j, line in enumerate(lines):
            new_lines = np.nonzero(assignments == j)[0]
            if not np.prod(new_lines.shape):
                LOG.debug("no lines for '%s' match or fit", line.id)
                continue
            covers = np.sum(covers_bg[new_lines,j])
            if covers < threshold / 3:
                LOG.debug("new lines for '%s' only cover %.1f%% bg",
                          line.id, covers * 100)
                continue
            covers = np.sum(covers_fg[new_lines,j])
            if covers < threshold:
                LOG.debug("new lines for '%s' only cover %.1f%% fg",
                          line.id, covers * 100)
                continue
            looses = (assignments < 0) & (covers_bg[:,j] > 0.1)
            if looses.any():
                covers = np.sum(covers_bg[np.nonzero(looses)[0],j])
                LOG.debug("new lines for '%s' would loose %d non-matching segments totalling %.1f%% bg",
                          line.id, np.count_nonzero(looses), covers * 100)
                continue
            line_count = np.count_nonzero(line_labels[:,:,j] & parent_bin)
            new_count = covers * line_count
            LOG.debug('Black pixels before/after resegment of line "%s": %d/%d',
                      line.id, line_count, new_count)
            # combine all assigned new lines to single outline polygon
            if len(new_lines) > 1:
                LOG.debug("joining %d new line polygons for '%s'", len(new_lines), line.id)
            new_polygon = join_polygons([intersections[(i, j)] for i in new_lines],
                                        contract=scale//2)
            # convert back to absolute (page) coordinates:
            line_polygon = coordinates_for_segment(new_polygon.exterior.coords[:-1],
                                                   parent_image, parent_coords)
            line_polygon = polygon_for_parent(line_polygon, line.parent_object_)
            if line_polygon is None:
                LOG.warning("Ignoring extant new polygon for line '%s'", line.id)
                return
            # annotate result:
            line.get_Coords().set_points(points_from_polygon(line_polygon))

def join_polygons(polygons, contract=2):
    # construct convex hull
    compoundp = unary_union(polygons)
    jointp = compoundp.convex_hull
    # FIXME: calculate true alpha shape
    # make hull slightly concave by dilation and reconstruction
    for step in range(int(contract)+1):
        nextp = jointp.buffer(-1)
        if (nextp.type == 'MultiPolygon' or
            nextp.union(compoundp).type == 'MultiPolygon'):
            break
        jointp = nextp
    jointp = jointp.union(compoundp)
    if jointp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        jointp = asPolygon(np.round(jointp.exterior.coords))
        jointp = make_valid(jointp)
    return jointp
