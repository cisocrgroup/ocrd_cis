
import subprocess
import os
import re
from PIL import Image


def cmd_to_string(cmd):
    """remove unneeded whitespace from command strings"""
    return re.sub("""\\s+""", " ", cmd).strip()


def subprocess_cmd(command, want=0):
    print("running command: {}".format(cmd_to_string(command)))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, _ = process.communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    returncode = process.wait()
    if returncode != want:
        raise Exception("invalid returncode for {cmd}: {c}"
                        .format(cmd=cmd_to_string(command), c=returncode))




def runtesseract(pngs, model):

    for file in pngs:

        tessercmd = '''
        tesseract \
        {filename} \
        {outname} \
        -l {model} \
        --psm 7
        '''.format(filename=file, outname=file[:-4]+'--tess'+'-'+model, model=model)

        subprocess_cmd(tessercmd)


def runocropy(fromdir, model):

    tessercmd = '''
    ocrd-cis-ocropy-rec -f {fromdir} -m {model}
    '''.format(fromdir=fromdir, model=model)

    subprocess_cmd(tessercmd)


def runcalamari(pngs, models):

    filestr = ' '.join(pngs)
    modelstr = ' '.join(models)


    calamaricmd = '''
    calamari-predict \
     --checkpoint {models} \
     --files {files}
    '''.format(models=modelstr, files=filestr)
    subprocess_cmd(calamaricmd)




def main():
    path = '/mnt/c/Users/chris/Documents/projects/OCR-D/daten/gt/lines/'
    os.chdir(path)

    #models

    #tesseractmodels
    tmodel1 = 'deu_frak'
    tmodel2 = 'eng'
    tmodel3 = tmodel1 + '+' + tmodel2

    #ocropymodels
    omodel1 = 'fraktur.pyrnn.gz'
    omodel2 = 'en-default.pyrnn.gz'

    #calamarimodels
    cmodel1= "/mnt/c/Users/chris/Documents/projects/OCR-D/daten/calamari_models-master/antiqua_historical/4.ckpt"
    cmodel2= "/mnt/c/Users/chris/Documents/projects/OCR-D/daten/calamari_models-master/antiqua_modern/4.ckpt"
    cmodel3= "/mnt/c/Users/chris/Documents/projects/OCR-D/daten/calamari_models-master/fraktur_19th_century/4.ckpt"
    cmodel4= "/mnt/c/Users/chris/Documents/projects/OCR-D/daten/calamari_models-master/fraktur_historical/4.ckpt"

    cmodels = [cmodel1, cmodel2, cmodel3, cmodel4]

    #models = args[1]

    print(path)
    _, dirs, _ = os.walk(path).__next__()

    print(dirs)
    pngs = []
    for dir in dirs:
        root, _, files = os.walk(path + dir).__next__()

        for file in files:
            if '.png' in file[-4:]:

                image = Image.open(file)
                w, _ = image.size

                # (h, w = image.shape)
                if w > 5000:
                    print("final image too long: %d", w)
                    continue

                pngs.append(root+'/'+file)


    runtesseract(pngs, tmodel1)
    runtesseract(pngs, tmodel2)
    runtesseract(pngs, tmodel3)


    runocropy(path, omodel1)
    runocropy(path, omodel2)


    runcalamari(pngs, cmodels)



main()
