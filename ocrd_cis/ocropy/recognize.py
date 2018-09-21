from __future__ import absolute_import

from ocrd.utils import getLogger, concat_padded, xywh_from_points, points_from_x0y0x1y1
from ocrd.model.ocrd_page import from_file, to_xml, TextEquivType, CoordsType, GlyphType
from ocrd import Processor, MIMETYPE_PAGE

from ocrd_cis import get_ocrd_tool

import sys, os, numpy, subprocess, re


from PIL import Image
from PIL import ImageOps
from scipy.ndimage import measurements



def bounding_box(coord_points):
    point_list = [[int(p) for p in pair.split(',')] for pair in coord_points.split(' ')]
    x_coordinates, y_coordinates = zip(*point_list)
    return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))

def binarize_array(numpy_array, threshold=130):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array



def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True) #, executable='/bin/bash')
    out, err = process.communicate(command.encode('utf-8'))
    return out.decode('utf-8')


class OcropyRecognize(Processor):

    def __init__(self, *args, **kwargs):
        ocrd_tool = get_ocrd_tool()
        kwargs['ocrd_tool'] = ocrd_tool['tools']['cis-ocrd-ocropy-recognize']
        kwargs['version'] = ocrd_tool['version']
        super(OcropyRecognize, self).__init__(*args, **kwargs)
        self.log = getLogger('Processor.OcropyRecognize')


    def process(self):
        """
        Performs the (text) recognition.
        """
        print(self.parameter)
        if self.parameter['textequiv_level'] not in ['line', 'glyph']:
            raise Exception("currently only implemented at the line/glyph level")
        
        filepath = os.path.dirname(os.path.abspath(__file__))
        model = 'fraktur.pyrnn.gz' # default model
        modelpath = filepath + '/models/' + model + '.gz'
        
        if 'model' in self.parameter:
            model = self.parameter['model']
            modelpath = filepath + '/models/' + model + '.gz'
            if os.path.isfile(modelpath) == False:
                raise Exception("configured model " + model + " is not in models folder")


        #self.log.info("Using model %s in %s for recognition", model)
        for (n, input_file) in enumerate(self.input_files):
            #self.log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = from_file(self.workspace.download_file(input_file))
            # TODO use binarized / gray
            pil_image = self.workspace.resolve_image_as_pil(pcgts.get_Page().imageFilename)

            #binarize
            pil_image = pil_image.convert('L')
            pil_image = numpy.array(pil_image)
            pil_image = binarize_array(pil_image,130)
            pil_image = Image.fromarray(pil_image, 'L')


            self.log.info("page %s", pcgts)
            for region in pcgts.get_Page().get_TextRegion():
                textlines = region.get_TextLine()
                self.log.info("About to recognize text in %i lines of region '%s'", len(textlines), region.id)
                for line in textlines:
                    self.log.debug("Recognizing text in line '%s'", line.id)
                    

                    #get box from points
                    box = bounding_box(line.get_Coords().points)
                        
                    #crop word from page
                    croped_image = pil_image.crop(box=box)

                    imgpath = filepath + '/temp/temp.png'
                    croped_image.save(imgpath)

                    #use ocropy to recognize word
                    ocropyfile = filepath + '/ocropus-rpred.py'

                    ocropycmd = '''
                    python2.7 {ocropyfile} -Q 4 -m {modelpath} '{imgpath}'
                    '''.format(ocropyfile=ocropyfile, modelpath=modelpath, imgpath=imgpath)

                    ocropyoutput = subprocess_cmd(ocropycmd)

                    matchObj = re.match( r'<ocropy>(.*?)</ocropy>', ocropyoutput)
                    if matchObj:
                        linepred = matchObj.group(1)

                    line.add_TextEquiv(TextEquivType(Unicode=linepred))
                    print(linepred)



                    if self.parameter['textequiv_level'] == 'glyph':
                        for word in line.get_Word():
                            self.log.debug("Recognizing text in word '%s'", word.id)

                            #get box
                            box = bounding_box(word.get_Coords().points)
                            
                            #crop word from page
                            croped_image = pil_image.crop(box=box)

                            imgpath = filepath + '/temp/temp.png'
                            croped_image.save(imgpath)

                            #use ocropy to recognize word
                            ocropyfile = filepath + '/ocropus-rpred.py'

                            ocropycmd = '''
                            python2.7 {ocropyfile} -Q 4 -m {modelpath} '{imgpath}'
                            '''.format(ocropyfile=ocropyfile, modelpath=modelpath, imgpath=imgpath)

                            ocropyoutput = subprocess_cmd(ocropycmd)

                            matchObj = re.match( r'<ocropy>(.*?)</ocropy>', ocropyoutput)
                            if matchObj:
                                wordpred = matchObj.group(1)

                            word.add_TextEquiv(TextEquivType(Unicode=wordpred))
                            print(wordpred)

                

            ID = concat_padded(self.output_file_grp, n)
            self.add_output_file(
                ID=ID,
                file_grp=self.output_file_grp,
                basename=ID + '.xml',
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )
