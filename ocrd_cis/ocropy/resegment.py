from __future__ import absolute_import

import os.path
import numpy as np
from skimage import draw
from shapely.geometry import Polygon
from shapely.prepared import prep

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
from .ocrolib import midrange, morph
from .common import (
    pil2array,
    # DSAVE,
    # binarize,
    check_page,
    check_region,
    hmerge_line_seeds,
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
        new line polygon, sorting by the shared area's ratio. Next, for each
        new line, if the largest relative overlap is sufficient, and its
        center is close to the center of the existing line, then assign that
        new line to that existing line, annotating the new coordinates, and
        stripping it from the overlap candidates for the other new lines.
        Thus, at the end, all new and existing lines will have been used
        at most once, but not all existing lines might have been resegmented
        (either because there were no matches at all, or the percentage of
        overlap was too small).

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
            regions = page.get_AllRegions(classes=['Text'])
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
            # only text region(s) may contain new text lines
            regions = set(line.parent_object_ for line in lines)
            region_labels = np.zeros_like(parent_bin, np.bool)
            region_labels = np.tile(np.expand_dims(region_labels, -1), (1,1,len(regions)))
            for i, segment in enumerate(regions):
                LOG.debug('unmasking area of text region "%s" for "%s"',
                          segment.id, page_id)
                segment_polygon = coordinates_of_segment(segment, parent_image, parent_coords)
                segment_y, segment_x = draw.polygon(segment_polygon[:, 1],
                                                    segment_polygon[:, 0],
                                                    ignore_bin.shape)
                ignore_bin[segment_y, segment_x] = False
                region_labels[segment_y, segment_x, i] = True
            # mask/ignore overlapping neighbours
            for i, segment in enumerate(ignore):
                LOG.debug('masking area of %s "%s" for "%s"',
                          type(segment).__name__[:-4], segment.id, page_id)
                segment_polygon = coordinates_of_segment(segment, parent_image, parent_coords)
                # If segment_polygon lies outside of element (causing
                # negative/above-max indices), either fully or partially,
                # then this will silently ignore them. The caller does
                # not need to concern herself with this.
                segment_y, segment_x = draw.polygon(segment_polygon[:, 1],
                                                    segment_polygon[:, 0],
                                                    ignore_bin.shape)
                ignore_bin[segment_y, segment_x] = True
            new_line_labels, _, _, _, _, scale = compute_segmentation(
                parent_bin, seps=ignore_bin, zoom=zoom, fullpage=fullpage,
                maxseps=0, maxcolseps=0, maximages=0)
            if fullpage:
                # on page level, long horizontal gaps will split lines
                # we still should try to merge them, but not across region boundaries
                # (lines can already belong to multiple regions before, but
                #  they should not get any more conflicts through hmerge):
                merged_labels = hmerge_line_seeds(parent_bin, new_line_labels, scale)
                merged = morph.correspondences(new_line_labels, merged_labels)
                across = np.zeros(merged_labels.max() + 1, np.bool)
                for ilabel, olabel, _ in merged.T:
                    iregions = region_labels[new_line_labels == ilabel].nonzero()[1]
                    oregions = region_labels[merged_labels == olabel].nonzero()[1]
                    if np.setdiff1d(oregions,iregions).any():
                        across[olabel] = True
                for ilabel, olabel, _ in merged.T:
                    if across[olabel]:
                        merged_labels[merged_labels == olabel] = new_line_labels[merged_labels == olabel]
                new_line_labels = merged_labels
        except Exception as err:
            LOG.warning('Cannot line-segment %s "%s": %s',
                        tag, page_id if fullpage else parent.id, err)
            return
        # line_labels = np.zeros_like(parent_array, np.uint8)
        # for j, line in enumerate(lines, 1):
        #     line_polygon = coordinates_of_segment(line, parent_image, parent_coords)
        #     line_labels[draw.polygon(line_polygon[:,1], line_polygon[:,0],
        #                              line_labels.shape)] = j
        # DSAVE('line_labels', line_labels + 0.5 * parent_bin)
        # DSAVE('new_line_labels', new_line_labels + 0.5 * parent_bin)
        LOG.info("Found %d new line labels for %d existing lines on %s '%s'",
                 new_line_labels.max(), len(lines), tag, page_id if fullpage else parent.id)
        # polygonalize and prepare comparison
        new_line_polygons = masks2polygons(new_line_labels, parent_bin,
                                           '%s "%s"' % (tag, page_id if fullpage else parent.id),
                                           min_area=640/zoom/zoom)
        new_line_polygons = [make_valid(Polygon(line_poly))
                             for line_label, line_poly in new_line_polygons]
        line_polygons = [prep(make_valid(Polygon(coordinates_of_segment(
            line, parent_image, parent_coords))).buffer(margin))
                         for line in lines]
        # compare segmentations, calculating ratio of overlapping background area
        intersections = dict()
        overlaps = np.zeros((len(new_line_polygons), len(line_polygons)), np.float)
        for i, new_line_poly in enumerate(new_line_polygons):
            for j, line_poly in enumerate(line_polygons):
                # too strict: .contains
                if line_poly.intersects(new_line_poly):
                    inter = make_intersection(line_poly.context, new_line_poly)
                    if not inter:
                        continue
                    ratio = inter.area / line_poly.context.area
                    overlaps[i, j] = ratio
                    intersections[(i, j)] = inter
        # assign new lines to existing lines if possible
        order = np.argsort(np.max(overlaps, axis=1))
        for i in order[::-1]:
            j = np.argmax(overlaps[i])
            line = lines[j]
            ratio = overlaps[i, j]
            if ratio < threshold:
                LOG.debug("new line for '%s' only has %.1f%% bg",
                          line.id, ratio * 100)
                continue
            # now determine ratio of overlapping foreground area
            inter_polygon = intersections[(i,j)]
            line_polygon = line_polygons[j].context
            line_mask = np.zeros_like(parent_bin, np.bool)
            inter_coords = np.array(inter_polygon.exterior.coords[:-1], np.int)
            line_coords = np.array(line_polygon.exterior.coords[:-1], np.int)
            line_mask[draw.polygon(inter_coords[:,1], inter_coords[:,0],
                                   line_mask.shape)] = True
            # DSAVE('inter_mask', line_mask + 0.5 * parent_bin)
            inter_count = np.count_nonzero(parent_bin * line_mask)
            line_mask[:,:] = False
            line_mask[draw.polygon(line_coords[:,1], line_coords[:,0],
                                   line_mask.shape)] = True
            # DSAVE('line_mask', line_mask + 0.5 * parent_bin)
            line_count = np.count_nonzero(parent_bin * line_mask)
            if not line_count:
                LOG.debug("existing line '%s' had no fg", line.id)
                continue
            ratio = inter_count / line_count
            if ratio < threshold:
                LOG.debug("new line for '%s' only has %.1f%% fg",
                          line.id, ratio * 100)
                continue
            LOG.debug('Black pixels before/after resegment of line "%s": %d/%d',
                      line.id, line_count, inter_count)
            new_center = inter_polygon.centroid
            center = line_polygon.centroid
            # FIXME: apply reasonable threshold for centroid distance
            LOG.debug("new line for '%s' has centroid distance %.2f",
                      line.id, center.distance(new_center))
            # avoid getting another new line assigned
            overlaps[:, j] = 0
            # convert back to absolute (page) coordinates:
            line_polygon = coordinates_for_segment(inter_polygon.exterior.coords[:-1],
                                                   parent_image, parent_coords)
            line_polygon = polygon_for_parent(line_polygon, line.parent_object_)
            if line_polygon is None:
                LOG.warning("Ignoring extant new polygon for line '%s'", line.id)
                return
            # annotate result:
            line.get_Coords().set_points(points_from_polygon(line_polygon))
        
