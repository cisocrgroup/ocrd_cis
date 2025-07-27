from __future__ import absolute_import
from logging import Logger
from typing import Optional

import numpy as np
from PIL import Image, ImageStat, ImageOps
from shapely.geometry import Polygon
from shapely.prepared import prep

from ocrd_models.ocrd_page import AlternativeImageType, OcrdPage
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage
from ocrd_utils import (
    bbox_from_polygon,
    coordinates_of_segment,
    crop_image,
    image_from_polygon,
    polygon_from_points,
    polygon_mask,
)

from .common import array2pil, determine_zoom, pil2array
from .ocrolib import midrange, morph


class OcropyClip(Processor):
    @property
    def executable(self):
        return 'ocrd-cis-ocropy-clip'

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: str = None) -> OcrdPageResult:
        """Clip text regions / lines of a page at intersections with neighbours.

        Open and deserialize PAGE input file and its respective image,
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

        Return the resulting OcrdPage.
        """
        # This makes best sense for overlapping segmentation, like current GT
        # or Tesseract layout analysis. Most notably, it can suppress graphics
        # and separators within or across a region or line. It _should_ ideally
        # be run after binarization (on page level for region-level clipping,
        # and on the region level for line-level clipping), because the
        # connected component analysis after implicit binarization could be
        # suboptimal, and the explicit binarization after clipping could be,
        # too. However, region-level clipping _must_ be run before region-level
        # deskewing, because that would make segments incommensurable with their
        # neighbours.
        level = self.parameter['level-of-operation']
        assert self.workspace
        self.logger.debug(f'Level of operation: "{level}"')

        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        assert page

        page_image, page_xywh, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_selector='binarized')
        # The zoom is not used anywhere
        zoom = determine_zoom(self.logger, page_id, self.parameter['dpi'], page_image_info)
        ret = OcrdPageResult(pcgts)

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
            self.logger.warning(f'Page "{page_id}" contains no text regions')
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
            shapes = [Polygon(polygon_from_points(region.get_Coords().points)) for region in regions]
            # in relative coordinates for mask/cropping
            polygons = [coordinates_of_segment(region, page_image, page_xywh) for region in regions]
            for i, polygon in enumerate(polygons[num_texts:], num_texts):
                # for non-text regions, extend mask by 3 pixels in each direction
                # to ensure they do not leak components accidentally
                # (accounts for bad cropping of such regions in GT):
                polygon = Polygon(polygon).buffer(3).exterior.coords[:-1] # keep open
                polygons[i] = polygon
            masks = [pil2array(polygon_mask(page_image, polygon)).astype(np.uint8) for polygon in polygons]
        for i, region in enumerate(regions):
            if i >= num_texts:
                break  # keep non-text regions unchanged
            if level == 'region':
                if region.get_AlternativeImage():
                    # FIXME: This should probably be an exception (bad workflow configuration).
                    self.logger.warning(f'Page "{page_id}" region "{region.id}" already contains image data: skipping')
                    continue
                shape = prep(shapes[i])
                neighbours = [
                    (regionj, maskj) for shapej, regionj, maskj in
                    zip(shapes[:i] + shapes[i + 1:], regions[:i] + regions[i + 1:], masks[:i] + masks[i + 1:])
                    if shape.intersects(shapej)]
                if neighbours:
                    ret.images.append(self.process_segment(
                        region, masks[i], polygons[i], neighbours, background_image,
                        page_image, page_xywh, page_bin, page_id))
                continue
            # level == 'line':
            lines = region.get_TextLine()
            if not lines:
                self.logger.warning(f'Page "{page_id}" region "{region.id}" contains no text lines')
                continue
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_xywh, feature_selector='binarized')
            background_image = Image.new(region_image.mode, region_image.size, background)
            region_array = pil2array(region_image)
            region_bin = np.array(region_array <= midrange(region_array), np.uint8)
            # in absolute coordinates merely for comparison/intersection
            shapes = [Polygon(polygon_from_points(line.get_Coords().points)) for line in lines]
            # in relative coordinates for mask/cropping
            polygons = [coordinates_of_segment(line, region_image, region_coords) for line in lines]
            masks = [pil2array(polygon_mask(region_image, polygon)).astype(np.uint8) for polygon in polygons]
            for j, line in enumerate(lines):
                if line.get_AlternativeImage():
                    # FIXME: This should probably be an exception (bad workflow configuration).
                    self.logger.warning(
                        f'Page "{page_id}" region "{region.id}" line "{line.id}" already contains image data: skipping')
                    continue
                shape = prep(shapes[j])
                neighbours = [
                    (linej, maskj) for shapej, linej, maskj in
                    zip(shapes[:j] + shapes[j + 1:], lines[:j] + lines[j + 1:], masks[:j] + masks[j + 1:])
                    if shape.intersects(shapej)]
                if neighbours:
                    ret.images.append(self.process_segment(
                        line, masks[j], polygons[j], neighbours, background_image,
                        region_image, region_coords, region_bin, page_id))
        return ret

    def process_segment(
            self, segment, segment_mask, segment_polygon, neighbours, background_image, parent_image, parent_coords,
            parent_bin, page_id
    ) -> OcrdPageResultImage:
        # initialize AlternativeImage@comments classes from parent, except
        # for those operations that can apply on multiple hierarchy levels:
        features = ','.join(
            [feature for feature in parent_coords['features'].split(',')
             if feature in ['binarized', 'grayscale_normalized', 'despeckled', 'dewarped']]) + ',clipped'
        # mask segment within parent image:
        segment_image = image_from_polygon(parent_image, segment_polygon)
        segment_bbox = bbox_from_polygon(segment_polygon)
        for neighbour, neighbour_mask in neighbours:
            if not np.any(segment_mask > neighbour_mask):
                self.logger.info(
                    f'Ignoring enclosing neighbour "{neighbour.id}" of segment "{segment.id}" on page "{page_id}"')
                continue
            # find connected components that (only) belong to the neighbour:
            intruders = segment_mask * morph.keep_marked(parent_bin, neighbour_mask > 0)  # overlaps neighbour
            intruders = morph.remove_marked(intruders, segment_mask > neighbour_mask)  # but exclusively
            num_intruders = np.count_nonzero(intruders)
            num_foreground = np.count_nonzero(segment_mask * parent_bin)
            if not num_intruders:
                continue
            self.logger.debug(
                f'segment "{segment.id}" vs neighbour "{neighbour.id}": suppressing {num_intruders} of '
                f'{num_foreground} pixels on page "{page_id}"')
            # suppress in segment_mask so these intruders can stay in the neighbours
            # (are not removed from both sides)
            segment_mask -= intruders
            # suppress in derived image result to be annotated
            clip_mask = array2pil(intruders)
            segment_image.paste(background_image, mask=clip_mask)  # suppress in raw image
            if segment_image.mode in ['RGB', 'L', 'RGBA', 'LA']:
                # for consumers that do not have to rely on our
                # guessed background color, but can cope with transparency:
                segment_image.putalpha(ImageOps.invert(clip_mask))
        # recrop segment into rectangle, just as image_from_segment would do
        # (and also clipping with background colour):
        segment_image = crop_image(segment_image, box=segment_bbox)
        # update PAGE (reference the image file):
        suffix = f'{segment.id}.IMG_CLIP'
        alternative_image = AlternativeImageType(comments=features)
        segment.add_AlternativeImage(alternative_image)
        return OcrdPageResultImage(segment_image, suffix, alternative_image)
