import os, json, sys
import shutil
import subprocess
from ocrd.utils import getLogger
from zipfile import ZipFile

'''
All in One Tool for:
    Unpacking GT-Files
    Adding them to a Workspace
    Recognizing the Text from Images with different models
    Aligning
'''


log = getLogger('cis.Processor.AIO')

def unpack(fromdir, todir):
    #extract all zips into temp dir
    log.debug("unpacking {from} {to}".format(form=fromdir, to=todir))
    path, dirs, files = os.walk(fromdir).__next__()
    for file in files:
        if '.zip' in file:
            filedir = os.path.join(fromdir, file)
            with ZipFile(filedir, 'r') as myzip:
                myzip.extractall(todir)


def subprocess_cmd(command):
    log.debug("running {command}".format(command=command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True) #, executable='/bin/bash')
    out, err = process.communicate(command.encode('utf-8'))
    log.info(out.decode('utf-8'))

def wgetGT():
    log.info('updating zip file into current folder')
    gtlink = 'http://www.ocr-d.de/sites/all/GTDaten/IndexGT.html'
    wgetcmd = '''
    wget -r -np -l1 -nd -N -A zip -erobots=off {link}
    '''.format(link = gtlink)
    subprocess_cmd(wgetcmd)

def printStats(gtdir):
    path, dirs, files = os.walk(gtdir).__next__()
    books, pages = 0, 0
    for file in files:
        if '.zip' in file:
            books += 1
            with ZipFile(file, 'r') as zip:
                zipinfo = zip.namelist()
                for elem in zipinfo:
                    if '.tif' in elem:
                        pages +=1
    log.info('files: ' + str(books) + ' - pages: ' + str(pages))

def addtoworkspace(wsdir, gtdir):
    #path to workspace
    if wsdir[-1] != '/': wsdir += '/'

    if not os.path.exists(wsdir):
        os.makedirs(wsdir)

    #workspace id (optional)
    #wsid = argv[3]

    tempdir = gtdir + '/temp'
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
        if not os.path.exists(filedir):
            filedir = os.path.join(tempdir, d)

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
    logger
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
    log.info('run aligner')
    allingercmd = '''
    ocrd-cis-align \
    --input-file-grp 'OCR-D-GT,OCR-D-TESSER,OCR-D-OCROPY-{model1},OCR-D-OCROPY-{model2}' \
    --output-file-grp 'OCR-D-GT+OCR-D-TESSER+OCR-D-OCROPY-{model1}+OCR-D-OCROPY-{model2}' \
    --mets {mets}/mets.xml \
    --parameter {parameter}
    '''.format(model1 = model1, model2=model2, mets = wsdir, parameter = configdir)
    subprocess_cmd(allingercmd)

def AllInOne(actualfolder, parameterfile):

    os.chdir(actualfolder)

    if parameterfile == None:
        log.error('A Parameterfile is mandatory')
    with open(parameterfile) as f:
        parameter = json.load(f)

    try:
        tesserpar = parameter['tesserparampath']
        ocropar1 = parameter['ocropyparampath1']
        ocropar2 = parameter['ocropyparampath2']
        alignpar = parameter['alignparampath']
    except(KeyError):
        log.error('The parameter file is not complete')
        sys.exit(1)


    #wget gt zip files (only downloads new zip files)
    wgetGT()

    printStats(actualfolder)

    workspacepath = actualfolder + '/workspace'

    #create Workspace
    addtoworkspace(workspacepath, actualfolder)


    #recognize Text with Tesserocr
    runtesserocr(workspacepath, tesserpar)


    #recognize Text with Ocropy model 1
    with open(ocropar1) as f:
        config = json.load(f)
    model1 = config['model']
    runocropy(workspacepath, ocropar1)

    #recognize Text with Ocropy model 2
    with open(ocropar2) as f:
        config = json.load(f)
    model2 = config['model']
    runocropy(workspacepath, ocropar2)

    runalligner(workspacepath,alignpar,model1,model2)
