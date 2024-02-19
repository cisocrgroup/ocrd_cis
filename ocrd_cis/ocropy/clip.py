from __future__ import absolute_import

import os.path
import numpy as np
from PIL import Image, ImageStat, ImageOps
from shapely.geometry import Polygon
from shapely.prepared import prep

from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml, AlternativeImageType
)
from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    polygon_from_points,
    bbox_from_polygon,
    image_from_polygon,
    polygon_mask,
    crop_image,
    MIMETYPE_PAGE
)

from .. import get_ocrd_tool
from .ocrolib import midrange, morph
from .common import (
    # binarize,
    pil2array, array2pil
)

TOOL = 'ocrd-cis-ocropy-clip'

class OcropyClip(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools'][TOOL]
        kwargs['version'] = self.ocrd_tool['version']
        super(OcropyClip, self).__init__(*args, **kwargs)

    def process(self):
        """Clip text regions / lines of the workspace at intersections with neighbours.

        Open and deserialise PAGE input files and their respective images,
        then iterate over the element hierarchy down to the requested
        ``level-of-operation``.

        Next, get each segment image according to the layout annotation (by cropping
        via coordinates into the higher-level image), as well as all its neighbours',
        binarize them (without deskewing), and make a connected component analysis.
        (Segments must not already have AlternativeImage annotated, otherwise they
        will be skipped.)

        Then, for each section of overlap with a neighbour, re-assign components
        which are only contained in the neighbour by clipping them to white (background),
        and export the (final) result as image file.

        Add the new image file to the workspace along with the output fileGrp,
        and using a file ID with suffix ``.IMG-CLIP`` along with further
        identification of the input element.

        Reference each new image in the AlternativeImage of the element.

        Produce a new output file by serialising the resulting hierarchy.
        """
        # This makes best sense for overlapping segmentation, like current GT
        # or Tesseract layout analysis. Most notably, it can suppress graphics
        # and separators within or across a region or line. It _should_ ideally
        # be run after binarization (on page level for region-level clipping,
        # and on the region level for line-level clipping), because the
        # connected component analysis after implicit binarization could be
        # suboptimal, and the explicit binarization after clipping could be,
        # too. However, region-level clipping _must_ be run before region-level
        # deskewing, because that would make segments incomensurable with their
        # neighbours.
        LOG = getLogger('processor.OcropyClip')
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

            # FIXME: what about text regions inside table regions?
            regions = list(page.get_TextRegion())
            num_texts = len(regions)
            regions += (
                page.get_AdvertRegion() +
                page.get_ChartRegion() +
                page.get_ChemRegion() +
                page.get_GraphicRegion() +
                page.get_ImageRegion() +
                page.get_LineDrawingRegion() +
                page.get_MathsRegion() +
                page.get_MusicRegion() +
                page.get_NoiseRegion() +
                page.get_SeparatorRegion() +
                page.get_TableRegion() +
                page.get_UnknownRegion())
            if not num_texts:
                LOG.warning('Page "%s" contains no text regions', page_id)
            background = ImageStat.Stat(page_image)
            # workaround for Pillow#4925
            if len(background.bands) > 1:
                background = tuple(background.median)
            else:
                background = background.median[0]
            if level == 'region':
                background_image = Image.new(page_image.mode, page_image.size, background)
                page_array = pil2array(page_image)
                page_bin = np.array(page_array <= midrange(page_array), np.uint8)
                # in absolute coordinates merely for comparison/intersection
                shapes = [Polygon(polygon_from_points(region.get_Coords().points))
                          for region in regions]
                # in relative coordinates for mask/cropping
                polygons = [coordinates_of_segment(region, page_image, page_coords)
                            for region in regions]
                for i, polygon in enumerate(polygons[num_texts:], num_texts):
                    # for non-text regions, extend mask by 3 pixels in each direction
                    # to ensure they do not leak components accidentally
                    # (accounts for bad cropping of such regions in GT):
                    polygon = Polygon(polygon).buffer(3).exterior.coords[:-1] # keep open
                    polygons[i] = polygon
                masks = [pil2array(polygon_mask(page_image, polygon)).astype(np.uint8)
                         for polygon in polygons]
            for i, region in enumerate(regions):
                if i >= num_texts:
                    break # keep non-text regions unchanged
                if level == 'region':
                    if region.get_AlternativeImage():
                        # FIXME: This should probably be an exception (bad workflow configuration).
                        LOG.warning('Page "%s" region "%s" already contains image data: skipping',
                                    page_id, region.id)
                        continue
                    shape = prep(shapes[i])
                    neighbours = [(regionj, maskj) for shapej, regionj, maskj
                                  in zip(shapes[:i] + shapes[i+1:],
                                         regions[:i] + regions[i+1:],
                                         masks[:i] + masks[i+1:])
                                  if shape.intersects(shapej)]
                    if neighbours:
                        self.process_segment(region, masks[i], polygons[i],
                                             neighbours, background_image,
                                             page_image, page_coords, page_bin,
                                             input_file.pageId, file_id + '_' + region.id)
                    continue
                # level == 'line':
                lines = region.get_TextLine()
                if not lines:
                    LOG.warning('Page "%s" region "%s" contains no text lines', page_id, region.id)
                    continue
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords, feature_selector='binarized')
                background_image = Image.new(region_image.mode, region_image.size, background)
                region_array = pil2array(region_image)
                region_bin = np.array(region_array <= midrange(region_array), np.uint8)
                # in absolute coordinates merely for comparison/intersection
                shapes = [Polygon(polygon_from_points(line.get_Coords().points))
                          for line in lines]
                # in relative coordinates for mask/cropping
                polygons = [coordinates_of_segment(line, region_image, region_coords)
                            for line in lines]
                masks = [pil2array(polygon_mask(region_image, polygon)).astype(np.uint8)
                         for polygon in polygons]
                for j, line in enumerate(lines):
                    if line.get_AlternativeImage():
                        # FIXME: This should probably be an exception (bad workflow configuration).
                        LOG.warning('Page "%s" region "%s" line "%s" already contains image data: skipping',
                                    page_id, region.id, line.id)
                        continue
                    shape = prep(shapes[j])
                    neighbours = [(linej, maskj) for shapej, linej, maskj
                                  in zip(shapes[:j] + shapes[j+1:],
                                         lines[:j] + lines[j+1:],
                                         masks[:j] + masks[j+1:])
                                  if shape.intersects(shapej)]
                    if neighbours:
                        self.process_segment(line, masks[j], polygons[j],
                                             neighbours, background_image,
                                             region_image, region_coords, region_bin,
                                             input_file.pageId, file_id + '_' + region.id + '_' + line.id)

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

    def process_segment(self, segment, segment_mask, segment_polygon, neighbours,
                        background_image, parent_image, parent_coords, parent_bin,
                        page_id, file_id):
        LOG = getLogger('processor.OcropyClip')
        # initialize AlternativeImage@comments classes from parent, except
        # for those operations that can apply on multiple hierarchy levels:
        features = ','.join(
            [feature for feature in parent_coords['features'].split(',')
             if feature in ['binarized', 'grayscale_normalized',
                            'despeckled', 'dewarped']]) + ',clipped'
        # mask segment within parent image:
        segment_image = image_from_polygon(parent_image, segment_polygon)
        segment_bbox = bbox_from_polygon(segment_polygon)
        for neighbour, neighbour_mask in neighbours:
            if not np.any(segment_mask > neighbour_mask):
                LOG.info('Ignoring enclosing neighbour "%s" of segment "%s" on page "%s"',
                         neighbour.id, segment.id, page_id)
                continue
            # find connected components that (only) belong to the neighbour:
            intruders = segment_mask * morph.keep_marked(parent_bin, neighbour_mask > 0) # overlaps neighbour
            intruders = morph.remove_marked(intruders, segment_mask > neighbour_mask) # but exclusively
            num_intruders = np.count_nonzero(intruders)
            num_foreground = np.count_nonzero(segment_mask * parent_bin)
            if not num_intruders:
                continue
            LOG.debug('segment "%s" vs neighbour "%s": suppressing %d of %d pixels on page "%s"',
                      segment.id, neighbour.id, num_intruders, num_foreground, page_id)
            # suppress in segment_mask so these intruders can stay in the neighbours
            # (are not removed from both sides)
            segment_mask -= intruders
            # suppress in derived image result to be annotated
            clip_mask = array2pil(intruders)
            segment_image.paste(background_image, mask=clip_mask) # suppress in raw image
            if segment_image.mode in ['RGB', 'L', 'RGBA', 'LA']:
                # for consumers that do not have to rely on our
                # guessed background color, but can cope with transparency:
                segment_image.putalpha(ImageOps.invert(clip_mask))
        # recrop segment into rectangle, just as image_from_segment would do
        # (and also clipping with background colour):
        segment_image = crop_image(segment_image,box=segment_bbox)
        # update METS (add the image file):
        file_path = self.workspace.save_image_file(
            segment_image, file_id + '.IMG-CLIP', self.output_file_grp,
            page_id=page_id)
        # update PAGE (reference the image file):
        segment.add_AlternativeImage(AlternativeImageType(
            filename=file_path,
            comments=features))
