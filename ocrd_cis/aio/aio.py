import os, json
import shutil
import subprocess
from zipfile import ZipFile

'''
All in One Tool for:
    Unpacking GT-Files
    Adding them to a Workspace
    Recognizing the Text from Images with different models
    Aligning
'''



def unpack(fromdir, todir):
    #extract all zips into temp dir

    path, dirs, files = os.walk(fromdir).__next__()
    for file in files:
        filedir = os.path.join(fromdir, file)
        with ZipFile(filedir, 'r') as myzip:
            myzip.extractall(todir)


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True) #, executable='/bin/bash')
    out, err = process.communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))




def addtoworkspace(wsdir, gtdir):
    #path to workspace
    if wsdir[-1] != '/': wsdir += '/'

    if not os.path.exists(wsdir):
        os.makedirs(wsdir)

    #workspace id (optional)
    #wsid = argv[3]

    tempdir = gtdir + 'temp'
    fileprefix = 'file://'


    #unpack zip files into temp file
    unpack(gtdir, tempdir)

    os.chdir(wsdir)

    initcmd = 'ocrd workspace init {}'.format(wsdir)
    subprocess_cmd(initcmd)

    #setidcmd = 'ocrd workspace set-id {}'.format(wsid)
    #subprocess_cmd(setidcmd)


    #walk through unpacked zipfiles and add tifs and xmls to workspace
    path, dirs, files = os.walk(tempdir).__next__()
    for d in dirs:
        filedir = os.path.join(tempdir, d, d)
        path2, dirs2, tiffiles = os.walk(filedir).__next__()

        for tif in tiffiles:
            if tif[-4:] == '.tif':
                filename = tif[:-4]
                
                tifdir = os.path.join(filedir, tif)
                xmldir = os.path.join(filedir, 'page', filename + '.xml') 


                #add tif image to workspace
                filegrp = 'OCR-D-IMG'
                mimetype = 'image/tif'
                fileid = filegrp + '-' + filename
                grpid = fileid
                imgcmd = '''ocrd workspace add \
                --file-grp {filegrp} \
                --file-id {fileid} \
                --group-id {grpid} \
                --mimetype {mimetype} \
                {fdir}'''.format(filegrp=filegrp, fileid=fileid, grpid=grpid, mimetype=mimetype, fdir=tifdir)

                subprocess_cmd(imgcmd)


                #add xml to workspace
                filegrp = 'OCR-D-GT'
                mimetype = 'application/vnd.prima.page+xml'
                fileid = filegrp + '-' + filename
                grpid = fileid
                xmlcmd = '''ocrd workspace add \
                --file-grp {filegrp} \
                --file-id {fileid} \
                --group-id {grpid} \
                --mimetype {mimetype} \
                {fdir}'''.format(filegrp=filegrp, fileid=fileid, grpid=grpid, mimetype=mimetype, fdir=xmldir)
                subprocess_cmd(xmlcmd)


                #rename filepaths in xml into file-urls
                sedcmd = '''
                sed -i {fname}.xml -e 's#imageFilename="{tif}"#imageFilename="{fdir}"#'
                '''.format(fname=wsdir+'OCR-D-GT/'+filename, tif=tif, fdir=fileprefix+wsdir+'OCR-D-IMG/'+tif)
                subprocess_cmd(sedcmd)


    shutil.rmtree(tempdir)


def runtesserocr(wsdir,configdir):

    #add xml to workspace
    filegrp = 'OCR-D-GT'
    tesserocrcmd = '''
    ocrd-tesserocr-recognize \
    --input-file-grp OCR-D-GT \
    --output-file-grp OCR-D-TESSER \
    --mets {mets}/mets.xml \
    --parameter {parameter}
    '''.format(mets = wsdir, parameter = configdir)

    subprocess_cmd(tesserocrcmd)


def runocropy(wsdir,configdir):

    with open(configdir) as f:
        config = json.load(f)

    model = config['model']

    ocropycmd = '''
    ocrd-cis-ocropy-recognize \
    --input-file-grp OCR-D-GT \
    --output-file-grp OCR-D-OCROPY-{model} \
    --mets {mets}/mets.xml \
    --parameter {parameter}
    '''.format(model = model, mets = wsdir, parameter = configdir)

    subprocess_cmd(ocropycmd)


def runalligner(wsdir,configdir,model1,model2):
    print('run aligner')
    allingercmd = '''
    ocrd-cis-align \
    --input-file-grp 'OCR-D-GT,OCR-D-TESSER,OCR-D-OCROPY-{model1},OCR-D-OCROPY-{model2}' \
    --output-file-grp 'OCR-D-GT+OCR-D-TESSER+OCR-D-OCROPY-{model1}+OCR-D-OCROPY-{model2}' \
    --mets {mets}/mets.xml \
    --parameter {parameter}
    '''.format(model1 = model1, model2=model2, mets = wsdir, parameter = configdir)
    subprocess_cmd(allingercmd)



def getValidInput(actualfolder):
    inp = input()
    if inp == 'y':
        return actualfolder + '/workspace'
    elif inp == 'n':
        print('enter a valid path for your workspace')
        return input()
    else:
        print('y/n ?')
        return getValidInput(actualfolder)





class AllInOne():

    actualfolder = os.getcwd()
    print(actualfolder)
    print('create workspace in same path? y/n:')
    workspacepath = getValidInput(actualfolder)

    #create Workspace
    addtoworkspace(workspacepath, actualfolder)


    #recognize Text with Tesserocr
    configdir = '/mnt/c/Users/chris/Documents/projects/OCR-D/daten/config/deu-frak.json'
    runtesserocr(workspacepath, configdir)


    #recognize Text with Ocropy model 1
    configdir = '/mnt/c/Users/chris/Documents/projects/OCR-D/daten/config/ocropy2.json'
    with open(configdir) as f:
        config = json.load(f)
    model1 = config['model']    
    runocropy(workspacepath, configdir)

    #recognize Text with Ocropy model 2
    configdir = '/mnt/c/Users/chris/Documents/projects/OCR-D/daten/config/ocropy.json'
    with open(configdir) as f:
        config = json.load(f)
    model2 = config['model']    
    runocropy(workspacepath, configdir)


    configdir = '/mnt/c/Users/chris/Documents/projects/OCR-D/daten/config/align.json'


    #rename filepaths in xml into file-urls
    # import re

    # newText = ''
    # with open(workspacepath+'/mets.xml', 'rt') as f:
    
    #     text=f.read()
    #     newText = text
    #     filepath = ''

    #     for line in text:
    #         idline = re.match(r".*ID=\"(?P<fileid>OCR-D-GT.+)\">", line)
    #         replaceline = re.match(r"(?P<first>.*xlink:href=\".+workspace/)(?P<middle>.+OCR.D.GT.*)(?P<end>.xml.+)", line)

    #         if idline:
    #             filepath = idline.group('fileid')
    #             print(filepath)
    #         if replaceline:
    #             newline = replaceline.group('first') + filepath + replaceline.group('end')
    #             print(newline)
    #             newtext = newText.replace(line, newline)
    
    # with open(workspacepath+'/mets.xml', "w") as f:
    #     f.write(newText)

    
    runalligner(workspacepath,configdir,model1,model2)
