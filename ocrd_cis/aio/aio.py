import re
import os
import json
import shutil
import subprocess
from zipfile import ZipFile
from collections import defaultdict

import string

from ocrd.utils import getLogger
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
    for file in files:
        if '.zip' in file:
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
    """remove unneded whitepsace from command strings"""
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

    initcmd = 'ocrd workspace init {}'.format(wsdir)
    subprocess_cmd(initcmd)

    # setidcmd = 'ocrd workspace set-id {}'.format(wsid)
    # subprocess_cmd(setidcmd)

    # walk through unpacked zipfiles and add tifs and xmls to workspace
    _, dirs, _ = os.walk(tempdir).__next__()
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
        subprocess_cmd(ocropycmd)

    return fgrpdict


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


def getFileLanguage(workspace, filegroup, index=0):

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

        lang = detect_language(pagetext)
        fgrp[lang] += 1

    return max(fgrp, key=lambda k: fgrp[k])


def tokenize(text):
    remove_punct = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    remove_digits = str.maketrans('', '', string.digits)

    tokenList = text.replace('\n', ' ').replace('\r', '').translate(remove_digits)\
        .translate(remove_punct).lower().strip().split()
    return tokenList

def detect_language(text):
    languages_ratios = {}
    words = tokenize(text)
    words_set = set(words)


    germanstopwords = 'aber, als, am, an, auch, auf, aus, bei, bin, bis, bist, da, dadurch, daher, darum, ' \
                      'das, daß, dass, dein, deine, dem, den, der, des, dessen, deshalb, die, dies, dieser, ' \
                      'dieses, doch, dort, du, durch, ein, eine, einem, einen, einer, eines, er, es, euer, ' \
                      'eure, für, hatte, hatten, hattest, hattet, hier, hinter, ich, ihr, ihre, im, in, ist, ' \
                      'ja, jede, jedem, jeden, jeder, jedes, jener, jenes, jetzt, kann, kannst, können, könnt, ' \
                      'machen, mein, meine, mit, muß, mußt, musst, müssen, müßt, nach, nachdem, nein, nicht, ' \
                      'nun, oder, seid, sein, seine, sich, sie, sind, soll, sollen, sollst, sollt, sonst, ' \
                      'soweit, sowie, und, unser, unsere, unter, vom, von, vor, wann, warum, was, weiter, ' \
                      'weitere, wenn, wer, werde, werden, werdet, weshalb, wie, wieder, wieso, wir, wird, ' \
                      'wirst, wo, woher, wohin, zu, zum, zur, über'

    englishstopwords = "a, about, above, after, again, against, all, am, an, and, any, are, aren't, as, at, " \
                       "be, because, been, before, being, below, between, both, but, by, can't, cannot, could, " \
                       "couldn't, did, didn't, do, does, doesn't, doing, don't, down, during, each, few, for, " \
                       "from, further, had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll, he's, " \
                       "her, here, here's, hers, herself, him, himself, his, how, how's, i, i'd, i'll, i'm, " \
                       "i've, if, in, into, is, isn't, it, it's, its, itself, let's, me, more, most, mustn't, " \
                       "my, myself, no, nor, not, of, off, on, once, only, or, other, ought, our, ours, ourselves, " \
                       "out, over, own, same, shan't, she, she'd, she'll, she's, should, shouldn't, so, some, such, " \
                       "than, that, that's, the, their, theirs, them, themselves, then, there, there's, these, they, " \
                       "they'd, they'll, they're, they've, this, those, through, to, too, under, until, up, very, " \
                       "was, wasn't, we, we'd, we'll, we're, we've, were, weren't, what, what's, when, when's, " \
                       "where, where's, which, while, who, who's, whom, why, why's, with, won't, would, wouldn't, " \
                       "you, you'd, you'll, you're, you've, your, yours, yourself, yourselves"

    latinstopwords = 'ab, ac, ad, adhic, aliqui, aliquis, an, ante, apud, at, atque, aut, autem, cum, cur, de, ' \
                     'deinde, dum, ego, enim, ergo, es, est, et, etiam, etsi, ex, fio, haud, hic, iam, idem, ' \
                     'igitur, ille, in, infra, inter, interim, ipse, is, ita, magis, modo, mox, nam, ne, nec, ' \
                     'necque, neque, nisi, non, nos, o, ob, per, possum, post, pro, quae, quam, quare, qui, ' \
                     'quia, quicumque, quidem, quilibet, quis, quisnam, quisquam, quisque, quisquis, quo, ' \
                     'quoniam, sed, si, sic, sive, sub, sui, sum, super, suus, tam, tamen, trans, tu, tum, ' \
                     'ubi, uel, uero'

    languages = {'german':germanstopwords, 'englisch':englishstopwords, 'latin':latinstopwords}

    for language in languages:
            stopwords_set = set(languages[language])
            common_elements = words_set.intersection(stopwords_set)
            languages_ratios[language] = len(common_elements)

    stopwords_set = set(latinstopwords.split(', '))
    common_elements = words_set.intersection(stopwords_set)
    languages_ratios['latin'] = len(common_elements)

    most_rated_language = max(languages_ratios, key=languages_ratios.get)

    return most_rated_language




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
