from __future__ import absolute_import

import sys, os.path, cv2
from ocrd_modelfactory import page_from_file
from ocrd import Processor
from ocrd_utils import getLogger
from ocrd_cis import get_ocrd_tool

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from .ocropus_rtrain import *

np.seterr(divide='raise',over='raise',invalid='raise',under='ignore')




def bounding_box(coord_points):
    point_list = [[int(p) for p in pair.split(',')] for pair in coord_points.split(' ')]
    x_coordinates, y_coordinates = zip(*point_list)
    return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))


def deletefiles(filelist):
    for file in filelist:
        if os.path.exists(file):
            os.remove(file)
        if os.path.exists(file[:-3]+'gt.txt'):
            os.remove(file[:-3]+'gt.txt')

def resize_keep_ratio(image, baseheight=48):
    hpercent = (baseheight / float(image.size[1]))
    wsize = int((float(image.size[0] * float(hpercent))))
    image = image.resize((wsize, baseheight), Image.ANTIALIAS)
    return image


def binarize(pil_image):
    # Convert RGB to OpenCV
    img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)

    # global thresholding
    #ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    #ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    bin_img = Image.fromarray(th3)
    return bin_img



class OcropyTrain(Processor):

    def __init__(self, *args, **kwargs):
        self.log = getLogger('OcropyTrain')
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['ocrd-cis-ocropy-train']
        kwargs['version'] = ocrd_tool['version']
        super(OcropyTrain, self).__init__(*args, **kwargs)


    def process(self):
        """
        Performs the training
        """
        #print(self.parameter)
        if self.parameter['textequiv_level'] not in ['line', 'word', 'glyph']:
            raise Exception("currently only implemented at the line/glyph level")

        filepath = os.path.dirname(os.path.abspath(__file__))




        if 'model' in self.parameter:
            model = self.parameter['model']
            modelpath = filepath + '/models/' + model + '.gz'
            outputpath = filepath + '/output/' + model
            if 'outputpath' in self.parameter:
                outputpath = self.parameter + '/' + model
            if os.path.isfile(modelpath) == False:
                raise Exception("configured model " + model + " is not in models folder")
        else:
            modelpath = None
            outputpath = filepath + '/output/' + 'lstm'
            if 'outputpath' in self.parameter:
                outputpath = self.parameter + '/' +'lstm'

        if 'ntrain' in self.parameter:
            ntrain = self.parameter['ntrain']



        filelist = []

        #self.log.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            #self.log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            pil_image = self.workspace.resolve_image_as_pil(pcgts.get_Page().imageFilename)


            self.log.info("page %s", pcgts)
            for region in pcgts.get_Page().get_TextRegion():
                textlines = region.get_TextLine()
                self.log.info("About to extract %i lines in region '%s'", len(textlines), region.id)
                for line in textlines:

                    if self.parameter['textequiv_level'] == 'line':
                        self.log.debug("Extracting line '%s'", line.id)

                        #get box from points
                        box = bounding_box(line.get_Coords().points)

                        #crop word from page
                        croped_image = pil_image.crop(box=box)

                        #binarize with Otsu's thresholding after Gaussian filtering
                        bin_image = binarize(croped_image)

                        #resize image to 48 pixel height
                        final_img = resize_keep_ratio(bin_image)

                        #save temp image
                        path = os.path.join(filepath, 'temp', str(input_file.ID) + str(region.id) + str(line.id))
                        imgpath = path + '.png'
                        final_img.save(imgpath)

                        filelist.append(imgpath)

                        #ground truth
                        gt = line.get_TextEquiv()[0].Unicode.strip()
                        gtpath = path + '.gt.txt'
                        with open(gtpath, "w", encoding='utf-8') as f:
                            f.write(gt)



                    if self.parameter['textequiv_level'] == 'word' or 'glyph':
                        for word in line.get_Word():

                            if self.parameter['textequiv_level'] == 'word':
                                self.log.debug("Extracting word '%s'", word.id)

                                #get box from points
                                box = bounding_box(word.get_Coords().points)

                                #crop word from page
                                croped_image = pil_image.crop(box=box)

                                #binarize with Otsu's thresholding after Gaussian filtering
                                bin_image = binarize(croped_image)

                                #resize image to 48 pixel height
                                final_img = resize_keep_ratio(bin_image)

                                #save temp image
                                path = os.path.join(filepath, 'temp', str(input_file.ID) + str(region.id) + str(line.id) + str(word.id))
                                imgpath = path + '.png'
                                final_img.save(imgpath)

                                filelist.append(imgpath)

                                #ground truth
                                gt = word.get_TextEquiv()[0].Unicode.strip()
                                gtpath = path + '.gt.txt'

                                with open(gtpath, "w", encoding='utf-8') as f:
                                    f.write(gt)

                            else:
                                for glyph in word.get_Glyph():
                                    self.log.debug("Extracting glyph '%s'", glyph.id)

                                    #get box from points
                                    box = bounding_box(glyph.get_Coords().points)

                                    #crop word from page
                                    croped_image = pil_image.crop(box=box)

                                    #binarize with Otsu's thresholding after Gaussian filtering
                                    bin_image = binarize(croped_image)

                                    #resize image to 48 pixel height
                                    final_img = resize_keep_ratio(bin_image)

                                    #save temp image
                                    path = os.path.join(filepath, 'temp', str(input_file.ID) + str(region.id) + str(line.id) + str(word.id) + str(glyph.id))
                                    imgpath = path + '.png'
                                    final_img.save(imgpath)

                                    filelist.append(imgpath)

                                    #ground truth
                                    gt = glyph.get_TextEquiv()[0].Unicode.strip()
                                    with open(gtpath, "w", encoding='utf-8') as f:
                                        f.write(gt)


        rtrain(filelist, modelpath, outputpath, ntrain)
        deletefiles(filelist)
