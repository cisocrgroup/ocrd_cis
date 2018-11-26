import re
import os
import json
import sys
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
    '''extract all zips into temp dir'''
    path, dirs, files = os.walk(fromdir).__next__()
    for file in files:
        if '.zip' in file:
            filedir = os.path.join(fromdir, file)
            resdir = os.path.join(todir, file)
            if os.path.isdir(resdir[0:-4]):
                print("{dir} exists - skipping...".format(dir=resdir[0:-4]))
                continue
            with ZipFile(filedir, 'r') as myzip:
                print("unpacking {file} to {todir}"
                      .format(file=file, todir=todir))
                myzip.extractall(todir)


def subprocess_cmd(command, want=0):
    print(re.sub("""\\s+""", " ", "running {command}".format(command=command)))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    returncode = process.wait()
    if returncode != want:
        raise Exception("invalid returncode for {cmd}: {c}"
                        .format(cmd=command, c=returncode))


def wgetGT():
    print('updating zip file into current folder')
    gtlink = 'http://www.ocr-d.de/sites/all/GTDaten/IndexGT.html'
    wgetcmd = '''
    wget -r -np -l1 -nd -N -A zip -erobots=off {link}
    '''.format(link=gtlink)
    subprocess_cmd(wgetcmd, want=8)


def getbaseStats(gtdir):
    path, dirs, files = os.walk(gtdir).__next__()
    books, pages = 0, 0
    for file in files:
        if '.zip' in file:
            books += 1
            with ZipFile(file, 'r') as zip:
                zipinfo = zip.namelist()
                for elem in zipinfo:
                    if '.tif' in elem:
                        pages += 1
    return('files: ' + str(books) + ' - pages: ' + str(pages))


def find_page_xml_file(bdir, img):
    """search for a matching xml file in bdir"""
    for root, dirs, files in os.walk(bdir):
        if "alto" in root:
            continue
        if "page" not in root:
            continue
        for file in files:
            if file[-4:] != '.xml':
                print("not an xml file: continue")
                continue
            if img in file:
                return os.path.join(bdir, root, file)
    return None


def addtoworkspace(wsdir, gtdir):
    # path to workspace
    if wsdir[-1] != '/':
        wsdir += '/'

    if not os.path.exists(wsdir):
        os.makedirs(wsdir)

    # workspace id (optional)
    # wsid = argv[3]

    tempdir = gtdir + '/temp'
    fileprefix = 'file://'

    # unpack zip files into temp file
    unpack(gtdir, tempdir)

    os.chdir(wsdir)

    initcmd = 'ocrd workspace init {}'.format(wsdir)
    subprocess_cmd(initcmd)

    # setidcmd = 'ocrd workspace set-id {}'.format(wsid)
    # subprocess_cmd(setidcmd)

    # walk through unpacked zipfiles and add tifs and xmls to workspace
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
                xmldir = find_page_xml_file(filedir, filename)
                if xmldir is None or not os.path.exists(xmldir):
                    raise Exception("cannot find page xml for {tif}".
                                    format(tif=tif))

                # xmldir = os.path.join(filedir, 'page', filename + '.xml')

                # add tif image to workspace
                filegrp = 'OCR-D-IMG'
                mimetype = 'image/tif'
                fileid = filegrp + '-' + filename
                grpid = fileid
                imgcmd = '''ocrd workspace add \
                --file-grp {filegrp} \
                --file-id {fileid} \
                --group-id {grpid} \
                --mimetype {mimetype} \
                {fdir}'''.format(filegrp=filegrp, fileid=fileid, grpid=grpid,
                                 mimetype=mimetype, fdir=tifdir)
                subprocess_cmd(imgcmd)

                # add xml to workspace
                filegrp = 'OCR-D-GT'
                mimetype = 'application/vnd.prima.page+xml'
                fileid = filegrp + '-' + filename
                grpid = fileid
                xmlcmd = '''ocrd workspace add \
                --file-grp {filegrp} \
                --file-id {fileid} \
                --group-id {grpid} \
                --mimetype {mimetype} \
                {fdir}'''.format(filegrp=filegrp, fileid=fileid, grpid=grpid,
                                 mimetype=mimetype, fdir=xmldir)
                subprocess_cmd(xmlcmd)


                #rename filepaths in xml into file-urls
                sedcmd = '''
                sed -i {fname}.xml -e 's#imageFilename="{tif}"#imageFilename="{fdir}"#'
                '''.format(fname=wsdir+'OCR-D-GT/'+filename, tif=tif,
                           fdir=fileprefix+wsdir+'OCR-D-IMG/'+tif)
                subprocess_cmd(sedcmd)

    shutil.rmtree(tempdir)


def get_ocrd_model(configfile):
    """Read model parameter from configfile and return it."""
    with open(configfile) as f:
        config = json.load(f)
    return config['model']


def runtesserocr(wsdir, configdir):
    """ Run tesseract with a model and return the new output-file-grp"""
    model = get_ocrd_model(configdir)
    output_file_group = 'OCRD-D-TESSER-{model}'.format(model)
    tesserocrcmd = '''
    ocrd-tesserocr-recognize \
    --input-file-grp OCR-D-GT \
    --output-file-grp {ofg} \
    --mets {mets}/mets.xml \
    --parameter {parameter}
    '''.format(mets=wsdir, parameter=configdir, ofg=output_file_group)
    subprocess_cmd(tesserocrcmd)
    return output_file_group


def runocropy(wsdir, configdir):
    """ Run ocropy with a model and return the new output-file-grp"""
    model = get_ocrd_model(configdir)
    output_file_group = 'OCRD-D-OCORPY-{model}'.format(model)
    ocropycmd = '''
    ocrd-cis-ocropy-recognize \
    --input-file-grp OCR-D-GT \
    --output-file-grp {ofg} \
    --mets {mets}/mets.xml \
    --parameter {parameter}
    '''.format(mets=wsdir, parameter=configdir, ofg=output_file_group)
    subprocess_cmd(ocropycmd)
    return output_file_group


def runprofiler(wsdir, configdir, masterocr):
    pass


def runalligner(wsdir, configdir, models):
    input_file_group = 'OCR-D-GT,' + ','.join(models)
    print('run aligner')
    allingercmd = '''
    ocrd-cis-align \
    --input-file-grp '{ifg}' \
    --output-file-grp 'OCR-D-ALIGN' \
    --mets {mets}/mets.xml \
    --parameter {parameter}
    '''.format(ifg=input_file_group, mets=wsdir, parameter=configdir)
    subprocess_cmd(allingercmd)


def getstats(wsdir):
    inputfilegrp = 'OCR-D-ALIGN'
    statscmd = '''
    ocrd-cis-stats \
    --input-file-grp '{inpgrp}' \
    --mets {mets}/mets.xml
    '''.format(inpgrp=inputfilegrp, mets=wsdir)
    subprocess_cmd(statscmd)


def AllInOne(actualfolder, parameterfile):

    os.chdir(actualfolder)

    if parameterfile is None:
        print('A Parameterfile is mandatory')
    with open(parameterfile) as f:
        parameter = json.load(f)

    # try:
    #     tesserpar = parameter['tesserparampath']
    #     ocropar1 = parameter['ocropyparampath1']
    #     ocropar2 = parameter['ocropyparampath2']
    #     alignpar = parameter['alignparampath']
    # except(KeyError):
    #     print('parameter file is not complete')
    #     sys.exit(1)
    # wget gt zip files (only downloads new zip files)
    wgetGT()

    basestats = getbaseStats(actualfolder)

    workspacepath = actualfolder + '/workspace'

    # create Workspace
    addtoworkspace(workspacepath, actualfolder)

    models = list()
    for ocr in parameter['ocr']:
        if ocr['type'] == 'tesseract':
            ofg = runtesserocr(workspacepath, ocr['path'])
            models.append(ofg)
        elif ocr['type'] == 'ocorpy':
            ofg = runocropy(workspacepath, ocr['path'])
            models.append(ofg)
        else:
            raise Exception('invalid ocr type: {typ}'.format(typ=ocr['type']))

    runalligner(workspacepath, alignpar, models)

    print(basestats)
    getstats(workspacepath)
