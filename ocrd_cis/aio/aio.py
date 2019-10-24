import re
import os
import json
import shutil
import subprocess
from zipfile import ZipFile
from collections import defaultdict
import glob
from ocrd_cis import JavaTrain
from ocrd_cis import JavaEvalDLE
from ocrd_cis import JavaEvalRRDM

import string

from ocrd_utils import getLogger
from ocrd.model.ocrd_page_generateds import parse


'''
All in One Tool for:
    Unpacking GT-Files
    Adding them to a Workspace
    Recognizing the Text from Images with different models
    Aligning
'''

OCRD_IMG_FGROUP = 'OCR-D-IMG'
OCRD_GT_FGROUP = 'OCR-D-GT'


def unpack(fromdir, todir):
    '''extract all zips into temp dir'''
    n = 1
    _, _, files = os.walk(fromdir).__next__()

    #for checking if zips are already unpacked
    if os.path.exists(todir):
        _, dirs, _ = os.walk(todir).__next__()
    else:
        dirs = []

    for file in files:
        if '.zip' in file and file not in dirs:
            if n == 0:
                break
            n -= 1
            # skip -- missing page dir or corrupt xml
            exclude = ['anton_locusGalat_1800.zip', 'dorn_uppedat_1507.zip']
            if file in exclude:
                continue
            filedir = os.path.join(fromdir, file)
            resdir = os.path.join(todir, file)
            if os.path.isdir(resdir[0:-4]):
                print("{dir} exists - skipping...".format(dir=resdir[0:-4]))
                continue
            with ZipFile(filedir, 'r') as myzip:
                print("unpacking {file} to {todir}"
                      .format(file=file, todir=todir))
                myzip.extractall(todir)


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


def subprocess_ret(command, want=0):
    print(re.sub("""\\s+""", " ", "running {command}".format(command=command)))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, _ = process.communicate(command.encode('utf-8'))
    returncode = process.wait()
    if returncode != want:
        raise Exception("invalid returncode for {cmd}: {c}"
                        .format(cmd=command, c=returncode))
    return out.decode('utf-8')


def wgetGT():
    print('updating zip file into current folder')
    gtlink = 'http://www.ocr-d.de/sites/all/GTDaten/IndexGT.html'
    wgetcmd = '''
    wget -r -np -l1 -nd -N -A zip -erobots=off {link}
    '''.format(link=gtlink)
    subprocess_cmd(wgetcmd, want=8)


def getbaseStats(gtdir):
    """get base stats: number of projects and pages)"""
    _, _, files = os.walk(gtdir).__next__()
    books, pages = 0, 0
    for file in files:
        if '.zip' in file:
            books += 1
            with ZipFile(file, 'r') as _zip:
                zipinfo = _zip.namelist()
                for elem in zipinfo:
                    if '.tif' in elem:
                        pages += 1
    return('files: {books} - pages: {pages}'.format(
        books=books, pages=pages))


def find_page_xml_file(bdir, img):
    """search for a matching xml file in bdir"""
    for root, _, files in os.walk(bdir):
        if "alto" in root:
            continue
        if "page" not in root:
            continue
        for file in files:
            if file[-4:] != '.xml':
                continue
            if img in file:
                return os.path.join(bdir, root, file)
    return None


def getLF(wsdir, input_file_group):
    """get predominant language and fonttype of input file group"""

    langcmd = '''
    ocrd-cis-lang \
    --input-file-grp {ifg} \
    --mets {mets}/mets.xml \
    '''.format(mets=wsdir, ifg=input_file_group)

    [lang, font] = subprocess_ret(langcmd).strip().split('\n')

    return lang, font

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

    old_cwd = os.getcwd()
    os.chdir(wsdir)

    _, _, files = os.walk(wsdir).__next__()
    if 'mets.xml' not in files:
        initcmd = 'ocrd workspace init {}'.format(wsdir)
        subprocess_cmd(initcmd)


    # walk through unpacked zipfiles and add tifs and xmls to workspace
    _, dirs, _ = os.walk(tempdir).__next__()
    _, wsdirs, _ = os.walk(wsdir).__next__()

    for d in dirs:
        filedir = os.path.join(tempdir, d, d)
        if not os.path.exists(filedir):
            filedir = os.path.join(tempdir, d)

        _, _, tiffiles = os.walk(filedir).__next__()

        for tif in tiffiles:
            if tif[-4:] == '.tif':
                filename = tif[:-4]

                tifdir = os.path.join(filedir, tif)
                xmldir = find_page_xml_file(filedir, filename)

                if xmldir is None or not os.path.exists(xmldir):
                    print(
                        "Warning: cannot find page xml file for {tif}".format(tif=tif))
                    continue
                    # raise Exception("cannot find page xml for {tif}".
                    #                 format(tif=tif))
                xmlfname = xmldir[xmldir.rfind('/')+1:-4]
                if filename != xmlfname:
                    tif = xmlfname+'.tif'
                    os.rename(tifdir, os.path.join(filedir, tif))
                    filename = xmlfname
                    tifdir = os.path.join(filedir, tif)

                # xmldir = os.path.join(filedir, 'page', filename + '.xml')

                # add tif image to workspace
                filegrp = 'OCR-D-IMG-' + d
                mimetype = 'image/tif'
                fileid = filegrp + '-' + filename
                imgcmd = '''ocrd workspace add \
                --file-grp {filegrp} \
                --file-id {fileid} \
                --mimetype {mimetype} \
                {fdir}'''.format(filegrp=filegrp, fileid=fileid,
                                 mimetype=mimetype, fdir=tifdir)
                if filegrp not in wsdirs:
                    subprocess_cmd(imgcmd)

                # add xml to workspace
                filegrp = 'OCR-D-GT-' + d
                mimetype = 'application/vnd.prima.page+xml'
                fileid = filegrp + '-' + filename
                xmlcmd = '''ocrd workspace add \
                --file-grp {filegrp} \
                --file-id {fileid} \
                --mimetype {mimetype} \
                {fdir}'''.format(filegrp=filegrp, fileid=fileid,
                                 mimetype=mimetype, fdir=xmldir)
                if filegrp not in wsdirs:
                    subprocess_cmd(xmlcmd)

                # rename filepaths in xml into file-urls
                sedcmd = '''
                sed -i {fname}.xml -e 's#imageFilename="[^"]*"#imageFilename="{fdir}"#'
                '''.format(fname=wsdir+'OCR-D-GT-'+d+'/'+filename,
                           fdir=fileprefix+wsdir+'OCR-D-IMG-'+d+'/'+tif)
                subprocess_cmd(sedcmd)

    shutil.rmtree(tempdir)
    os.chdir(old_cwd)
    return dirs


def get_ocrd_model(configfile):
    """Read model parameter from a ocr configfile and return it."""
    print("opening {cnf} [{cwd}]".
          format(cnf=configfile, cwd=os.getcwd()))
    with open(configfile) as f:
        config = json.load(f)
    return config['model']


def runtesserocr(wsdir, configdir, fgrpdict):
    """ Run tesseract with a model and return the new output-file-grp"""
    print('runtesserocr({wsdir}, {cnf}, {fgrp}'.format(
        wsdir=wsdir, cnf=configdir, fgrp=fgrpdict))
    model = get_ocrd_model(configdir)
    for fgrp in fgrpdict:

        input_file_group = 'OCR-D-GT-{fgrp}'.format(fgrp=fgrp)
        output_file_group = 'OCR-D-TESSER-{model}-{fgrp}'.format(
            model=model, fgrp=fgrp)
        fgrpdict[fgrp].append(output_file_group)

        tesserocrcmd = '''
        ocrd-tesserocr-recognize \
        --input-file-grp {ifg} \
        --output-file-grp {ofg} \
        --mets {mets}/mets.xml \
        --parameter {parameter}
        '''.format(mets=wsdir, parameter=configdir,
                   ifg=input_file_group, ofg=output_file_group)

        _, wsdirs, _ = os.walk(wsdir).__next__()
        if output_file_group not in wsdirs:
            subprocess_cmd(tesserocrcmd)

    return fgrpdict


def runocropy(wsdir, configdir, fgrpdict):
    """ Run ocropy with a model and return the new output-file-grp"""
    model = get_ocrd_model(configdir)

    for fgrp in fgrpdict:

        input_file_group = 'OCR-D-GT-{fgrp}'.format(fgrp=fgrp)
        output_file_group = 'OCR-D-OCROPY-{model}-{fgrp}'.format(
            model=model, fgrp=fgrp)
        fgrpdict[fgrp].append(output_file_group)

        ocropycmd = '''
        ocrd-cis-ocropy-recognize \
        --input-file-grp {ifg} \
        --output-file-grp {ofg} \
        --mets {mets}/mets.xml \
        --parameter {parameter}
        '''.format(mets=wsdir, parameter=configdir,
                   ifg=input_file_group, ofg=output_file_group)

        _, wsdirs, _ = os.walk(wsdir).__next__()
        if output_file_group not in wsdirs:
            subprocess_cmd(ocropycmd)

    return fgrpdict

def runcutter(wsdir, configdir, fgrpdict):
    for fgrp in fgrpdict:
        input_file_group = 'OCR-D-GT-{fgrp}'.format(fgrp=fgrp)
        cuttercmd = '''
        ocrd-cis-cutter \
        --input-file-grp {ifg} \
        --mets {mets}/mets.xml \
        --parameter {parameter}
        '''.format(mets=wsdir, parameter=configdir,
                   ifg=input_file_group)
        subprocess_cmd(cuttercmd)


def runcalamari(wsdir, configdir, fgrpdict):
    with open(configdir) as f:
        data = json.load(f)
    linesdir = data['linesdir']
    models = []
    models.append(data['model1'])
    models.append(data['model2'])
    models.append(data['model3'])
    models.append(data['model4'])

    root, _, files = os.walk(linesdir).__next__()

    pngs = []
    for file in files:
        if '.png' in file:
            pngs.append(root + '/' + file)

    filestr = ' '.join(pngs)
    modelstr = ' '.join(models)


    calamaricmd = '''
    calamari-predict \
     --checkpoint {models} \
     --files {files} \
     --extended_prediction_data
    '''.format(models=modelstr, files=filestr)
    subprocess_cmd(calamaricmd)


    for fgrp in fgrpdict:

        input_file_group = 'OCR-D-GT-{fgrp}'.format(fgrp=fgrp)
        output_file_group = 'OCR-D-Calamari-{fgrp}'.format(fgrp=fgrp)
        fgrpdict[fgrp].append(output_file_group)

        importercmd = '''
        ocrd-cis-importer \
        --input-file-grp {ifg} \
        --output-file-grp {ofg} \
        --mets {mets}/mets.xml \
        --parameter {parameter}
        '''.format(mets=wsdir, parameter=configdir,
                   ifg=input_file_group, ofg=output_file_group)



        _, wsdirs, _ = os.walk(wsdir).__next__()
        if output_file_group not in wsdirs:
            subprocess_cmd(importercmd)




def runalligner(wsdir, configdir, fgrpdict):
    alignfilegrps = []
    for fgrp in fgrpdict:
        input_file_group = ','.join(fgrpdict[fgrp])
        output_file_group = 'OCR-D-ALIGN-{fgrp}'.format(fgrp=fgrp)
        alignfilegrps.append(output_file_group)
        alignercmd = '''
        ocrd-cis-align \
        --input-file-grp '{ifg}' \
        --output-file-grp '{ofg}' \
        --mets {mets}/mets.xml \
        --parameter {parameter} \
        --log-level DEBUG
        '''.format(ifg=input_file_group, ofg=output_file_group,
                   mets=wsdir, parameter=configdir)

        _, wsdirs, _ = os.walk(wsdir).__next__()
        if output_file_group not in wsdirs:
            subprocess_cmd(alignercmd)
    return alignfilegrps


def runprofiler(wsdir, configdir, input_file_grp):
    output_file_grp = input_file_grp.replace(
        'OCR-D-ALIGN-', 'OCR-D-PROFILE-')
    profilercmd = '''
        ocrd-cis-profile \
        --input-file-grp '{ifg}' \
        --output-file-grp '{ofg}' \
        --mets {mets}/mets.xml \
        --parameter {parameter} \
        --log-level DEBUG
        '''.format(ifg=input_file_grp, ofg=output_file_grp,
                   mets=wsdir, parameter=configdir)
    subprocess_cmd(profilercmd)
    return output_file_grp


def getstats(wsdir, alignfilegrps):
    stats = defaultdict(float)
    for fgrp in alignfilegrps:
        statscmd = '''
        ocrd-cis-stats \
        --input-file-grp '{inpgrp}' \
        --mets {mets}/mets.xml
        '''.format(inpgrp=fgrp, mets=wsdir)

        out = subprocess_ret(statscmd).strip()

        jout = json.loads(out.replace("'", '"'))

        for k, v in jout.items():
            stats[k] += v

    return stats


def getFileLanguage(workspace, filegroup, stopwordspath, index=0):

    fgrppath = workspace + '/' + filegroup
    _, _, files = os.walk(fgrppath).__next__()

    inputfiles = []
    for file in files:
        if 'xml' in file:
            inputfiles.append(file)

    fgrp = defaultdict(int)
    for input_file in inputfiles:

        alignurl = fgrppath + '/' + input_file
        pcgts = parse(alignurl, True)
        page = pcgts.get_Page()
        regions = page.get_TextRegion()

        pagetext = ''
        for region in regions:
            pagetext += region.get_TextEquiv()[index].Unicode + ' '

        lang = detect_language(pagetext, stopwordspath)
        fgrp[lang] += 1

    return max(fgrp, key=lambda k: fgrp[k])


def tokenize(text):
    remove_punct = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    remove_digits = str.maketrans('', '', string.digits)

    tokenList = text.replace('\n', ' ').replace('\r', '').translate(remove_digits)\
        .translate(remove_punct).lower().strip().split()
    return tokenList

def detect_language(text, stopwordspath):
    languages_ratios = {}
    words = tokenize(text)
    words_set = set(words)

    with open(stopwordspath, encoding='utf-8') as f:
        languages = json.load(f)

    for language in languages:
            stopwords_set = set(languages[language])
            common_elements = words_set.intersection(stopwords_set)
            languages_ratios[language] = len(common_elements)

    most_rated_language = max(languages_ratios, key=languages_ratios.get)

    return most_rated_language


def runeval(parameter, workspace, evaldir):
    mimetype = 'application/vnd.prima.page+xml'
    for pagexml in glob.glob(os.path.join(evaldir, "*.xml")):
        fileid = os.path.basename(pagexml)[0:-4]
        cmd = '''ocrd workspace add \
        --file-grp {filegrp} \
        --file-id {fileid} \
        --mimetype {mimetype} \
        {fdir}'''.format(filegrp='OCR-D-GT-EVAL', fileid=fileid,
                         mimetype=mimetype, fdir=pagexml)
        subprocess_cmd(cmd)

    fgrpdict = {'OCR-D-GT-EVAL': []}
    for ocr in parameter['ocr']:
        if ocr['type'] == 'tesseract':
            fgrpdict = runtesserocr(workspace, ocr['path'], fgrpdict)
        elif ocr['type'] == 'ocropy':
            fgrpdict = runocropy(workspace, ocr['path'], fgrpdict)
        else:
            raise Exception('invalid ocr type: {typ}'.format(typ=ocr['type']))
    fgrpdict['OCR-D-GT-EVAL'].append('OCR-D-GT-EVAL')
    alignfgrps = runalligner(workspace, parameter['alignparampath'], fgrpdict)

    update_profiler_configuration(
        parameter['profilerparampath'], "dynamiclex", "")
    profilerfgrps = []
    for fg in alignfgrps:
        ofg = runprofiler(workspace, parameter['profilerparampath'], fg)
        profilerfgrps.append(ofg)

    with open(parameter['evalparampath']) as f:
        evalparameter = json.load(f)

    for fg in profilerfgrps:
        p = JavaEvalDLE(
            evalparameter['cisOcrJar'],
            os.path.join(workspace, 'mets.xml'),
            [fg],
            parameter['evalparampath'],
            'DEBUG')
        p.run('')

    dynamiclex = evalparameter['dleTraining']['dynamicLexicon']
    update_profiler_configuration(
        parameter['profilerparampath'], "dynamiclex", dynamiclex)
    profilerfgrps = []
    for fg in alignfgrps:
        ofg = runprofiler(workspace, parameter['profilerparampath'], fg)
        profilerfgrps.append(ofg)

    for fg in profilerfgrps:
        p = JavaEvalRRDM(
            evalparameter['cisOcrJar'],
            os.path.join(workspace, 'mets.xml'),
            [fg],
            parameter['evalparampath'],
            'DEBUG')
        p.run('')

def update_profiler_configuration(path, key, val):
    with open(path) as f:
        params = json.load(f)
    params[key] = val
    with open(path, 'w') as f:
        json.dump(params, f)

def AllInOne(actualfolder, parameterfile, verbose, download):

    log = getLogger('AllInOne')
    if verbose:
        import logging
        log.setLevel(logging.DEBUG)


    os.chdir(actualfolder)


    if parameterfile is None:
        print('A Parameterfile is mandatory')
    with open(parameterfile) as f:
        parameter = json.load(f)

    # wget gt zip files (only downloads new zip files)
    if download:
        log.info("downloading missing files...")
        wgetGT()
    else:
        print('\ncontinuing without downloading missing files\n'
              'if you want to download all files automatically use the argument "-l"\n')

    basestats = getbaseStats(actualfolder)

    workspacepath = actualfolder + '/workspace'

    # create Workspace
    projects = addtoworkspace(workspacepath, actualfolder)

    fgrpdict = dict()
    for p in projects:
        fgrpdict[p] = []


    #runcutter(workspacepath, parameter['cutterparampath'], fgrpdict)
    #runcalamari(workspacepath, parameter['importerparampath'], fgrpdict)


    for ocr in parameter['ocr']:
        if ocr['type'] == 'tesseract':
            fgrpdict = runtesserocr(workspacepath, ocr['path'], fgrpdict)
        elif ocr['type'] == 'ocropy':
            fgrpdict = runocropy(workspacepath, ocr['path'], fgrpdict)
        else:
            raise Exception('invalid ocr type: {typ}'.format(typ=ocr['type']))

    for p in projects:
        gt_file_group = 'OCR-D-GT-{fgrp}'.format(fgrp=p)
        fgrpdict[p].append(gt_file_group)

    alignpar = parameter['alignparampath']

    # liste aller alignierten file-groups
    alignfgrps = runalligner(workspacepath, alignpar, fgrpdict)

    print(json.dumps(parameter))
    profilerfgrps = []
    for fg in alignfgrps:
        ofg = runprofiler(workspacepath, parameter['profilerparampath'], fg)
        profilerfgrps.append(ofg)

    print(basestats)

    stats = getstats(workspacepath, alignfgrps)
    gtstats = stats["gt"]
    for k, v in stats.items():
        if k != "gt":
            print(k + ' : ' + str(1-v/gtstats))

    # train on aligned profiler file groups
    if parameter['train']:
        with open(parameter['trainparampath']) as f:
            trainparameter = json.load(f)
        p = JavaTrain(
            trainparameter['jar'],
            os.path.join(workspacepath, 'mets.xml'),
            profilerfgrps,
            parameter['trainparampath'],
            'DEBUG')
        p.run('')
    runeval(parameter, workspacepath, parameter['evaldir'])



    # eval
