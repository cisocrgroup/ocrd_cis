from __future__ import absolute_import

from ocrd_cis import get_ocrd_tool

import sys
import os.path
import cv2
import numpy as np
from PIL import Image


from ocrd_utils import getLogger
from ocrd_modelfactory import page_from_file
from ocrd import Processor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def bounding_box(coord_points):
    point_list = [[int(p) for p in pair.split(',')]
                  for pair in coord_points.split(' ')]
    x_coordinates, y_coordinates = zip(*point_list)
    return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))


def resize_keep_ratio(image, baseheight=48):
    hpercent = (baseheight / float(image.size[1]))
    wsize = int((float(image.size[0] * float(hpercent))))
    image = image.resize((wsize, baseheight), Image.LANCZOS)
    return image


def binarize(pil_image):
    # Convert RGB to OpenCV
    img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    bin_img = Image.fromarray(th3)
    return bin_img





class Cutter(Processor):

    def __init__(self, *args, **kwargs):
        self.ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = self.ocrd_tool['tools']['ocrd-cis-cutter']
        kwargs['version'] = self.ocrd_tool['version']
        super(Cutter, self).__init__(*args, **kwargs)
        self.log = getLogger('Cutter')

    def process(self):
        """
        Performs the (text) recognition.
        """
        # print(self.parameter)
        linesdir = self.parameter['linesdir']


        # self.log.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            # self.log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            pil_image = self.workspace.resolve_image_as_pil(
                pcgts.get_Page().imageFilename)

            self.log.info("Preparing page '%s'", pcgts.get_pcGtsId())
            page = pcgts.get_Page()

            # region, line, word, or glyph level:
            regions = page.get_TextRegion()
            if not regions:
                self.log.warning("Page contains no text regions")


            for region in regions:
                self.log.info("Preparing region '%s'", region.id)

                textlines = region.get_TextLine()
                if not textlines:
                    self.log.warning(
                        "Region '%s' contains no text lines", region.id)
                else:

                    for line in textlines:
                        self.log.info("Cutting line '%s'", line.id)

                        # get box from points
                        box = bounding_box(line.get_Coords().points)

                        # crop word from page
                        croped_image = pil_image.crop(box=box)

                        # binarize with Otsu's thresholding after Gaussian filtering
                        bin_image = binarize(croped_image)

                        # resize image to 48 pixel height
                        final_img = resize_keep_ratio(bin_image)

                        index = input_file.url.rfind('/')
                        fgrp = input_file.url[index:-4]
                        # save temp image
                        suffix = fgrp + '-' + str(region.id) + '-' + str(line.id) + '.png'
                        imgpath = linesdir + suffix

                        if not os.path.exists(linesdir):
                            os.makedirs(linesdir)

                        final_img.save(imgpath)
