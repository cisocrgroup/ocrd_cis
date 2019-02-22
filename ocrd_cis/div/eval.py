import os
from PIL import Image
from Levenshtein import distance


path = '/mnt/c/Users/chris/Documents/projects/OCR-D/daten/gt/lines/'

total_gtcount = 0

total_dc = 0

total_do1 = 0
total_do2 = 0

total_do3 = 0
total_do4 = 0
total_do5 = 0
total_do6 = 0
total_do7 = 0

total_dt1 = 0
total_dt2 = 0
total_dt3 = 0

###
total_gtwordcount = 0
total_dcfull = 0

total_do1full = 0
total_do2full = 0

total_do3full = 0
total_do4full = 0
total_do5full = 0
total_do6full = 0
total_do7full = 0

total_dt1full = 0
total_dt2full = 0
total_dt3full = 0

_, dirs, _ = os.walk(path).__next__()

for dir in dirs:
    root, _, files = os.walk(path + dir).__next__()
    pngs = []

    for file in files:
        if '.png' in file[-4:]:

            image = Image.open(root + '/' + file)
            w, _ = image.size

            # (h, w = image.shape)
            if w > 5000:
                print("final image too long: %d", w)
                continue

            pngs.append(root + '/' + file[:-4])

    gtcount = 0

    dc = 0

    do1 = 0
    do2 = 0

    do3 = 0
    do4 = 0
    do5 = 0
    do6 = 0
    do7 = 0

    dt1 = 0
    dt2 = 0
    dt3 = 0

    ###
    gtwordcount = 0
    dcfull = 0

    do1full = 0
    do2full = 0

    do3full = 0
    do4full = 0
    do5full = 0
    do6full = 0
    do7full = 0

    dt1full = 0
    dt2full = 0
    dt3full = 0


    for f in pngs:
        gtfile = f + '.txt'

        calamarifile = f + '.pred.txt'
        ocropyfile1 = f + '--ocropy-en-default.pyrnn.gz.txt'
        ocropyfile2 = f + '--ocropy-fraktur.pyrnn.gz.txt'

        ocropyfile3 = f + '--ocropy-incunabula-00184000.pyrnn.gz.txt'
        ocropyfile4 = f + '--ocropy-latin1-00081000.pyrnn.gz.txt'
        ocropyfile5 = f + '--ocropy-latin2-00069000.pyrnn.gz.txt'
        ocropyfile6 = f + '--ocropy-ridges1-00085000.pyrnn.gz.txt'
        ocropyfile7 = f + '--ocropy-ridges2-00062000.pyrnn.gz.txt'

        tesseract1 = f + '--tess-eng.txt'
        tesseract2 = f + '--tess-deu_frak.txt'
        tesseract3 = f + '--tess-deu_frak+eng.txt'


        try:
            with open(gtfile) as fi:
                gtline = fi.readline()
                gtlen = len(gtline)
                gtcount += gtlen
                gtlinewords = gtline.split()
                gtwordcount += len(gtlinewords)

            with open(calamarifile) as fi:
                cpred = fi.readline()
                cpredwords = cpred.split()


            with open(ocropyfile1) as fi:
                opred1 = fi.readline()
                opred1words = opred1.split()
            with open(ocropyfile2) as fi:
                opred2 = fi.readline()
                opred2words = opred2.split()
            with open(ocropyfile3) as fi:
                opred3 = fi.readline()
                opred3words = opred3.split()
            with open(ocropyfile4) as fi:
                opred4 = fi.readline()
                opred4words = opred4.split()
            with open(ocropyfile5) as fi:
                opred5 = fi.readline()
                opred5words = opred5.split()
            with open(ocropyfile6) as fi:
                opred6 = fi.readline()
                opred6words = opred6.split()
            with open(ocropyfile7) as fi:
                opred7 = fi.readline()
                opred7words = opred7.split()


            with open(tesseract1) as fi:
                tpred1 = fi.readline()
                tpred1words = tpred1.split()
            with open(tesseract2) as fi:
                tpred2 = fi.readline()
                tpred2words = tpred2.split()
            with open(tesseract3) as fi:
                tpred3 = fi.readline()
                tpred3words = tpred3.split()
        except(FileNotFoundError):
            print('needed file not found for:')
            print(f)
            continue



        dc += distance(gtline, cpred)

        do1 += distance(gtline, opred1)
        do2 += distance(gtline, opred2)

        do3 += distance(gtline, opred3)
        do4 += distance(gtline, opred4)
        do5 += distance(gtline, opred5)
        do6 += distance(gtline, opred6)
        do7 += distance(gtline, opred7)

        dt1 += distance(gtline, tpred1)
        dt2 += distance(gtline, tpred2)
        dt3 += distance(gtline, tpred3)



        for w in gtlinewords:
            if w in tpred1words:
                dt1full += 1
            if w in tpred2words:
                dt2full += 1
            if w in tpred3words:
                dt3full += 1
            if w in cpredwords:
                dcfull += 1
            if w in opred1words:
                do1full += 1
            if w in opred2words:
                do2full += 1
            if w in opred3words:
                do3full += 1
            if w in opred4words:
                do4full += 1
            if w in opred5words:
                do5full += 1
            if w in opred6words:
                do6full += 1
            if w in opred7words:
                do7full += 1


    accc = 1-dc/gtcount

    acco1 = 1-do1/gtcount
    acco2 = 1-do2/gtcount

    acco3 = 1-do3/gtcount
    acco4 = 1-do4/gtcount
    acco5 = 1-do5/gtcount
    acco6 = 1-do6/gtcount
    acco7 = 1-do7/gtcount

    acct1 = 1-dt1/gtcount
    acct2 = 1-dt2/gtcount
    acct3 = 1-dt3/gtcount


    with open(root + '/' + '0000_eval.txt', 'w+') as fo:

        fo.write('#token errors made by models, lower is better\n')
        fo.write('total_gt_tokens: ' + str(gtcount) + '\n')
        total_gtcount += gtcount

        fo.write('calamari: ' + str(dc) + '\n')
        total_dc += dc

        fo.write('tesser_eng: ' + str(dt1) + '\n')
        total_dt1 += dt1
        fo.write('tesser_deu_frak: ' + str(dt2) + '\n')
        total_dt2 += dt2
        fo.write('tesser_deu_frak+eng: ' + str(dt3) + '\n')
        total_dt3 += dt3

        fo.write('ocropy_en-default: ' + str(do1) + '\n')
        total_do1 += do1
        fo.write('ocropy_fraktur:' + str(do2) + '\n')
        total_do2 += do2

        fo.write('ocropy_incunabula' + str(do3) + '\n')
        total_do3 += do3
        fo.write('ocropy_latin1' + str(do4) + '\n')
        total_do4 += do4
        fo.write('ocropy_latin2' + str(do5) + '\n')
        total_do5 += do5
        fo.write('ocropy_ridges1' + str(do6) + '\n')
        total_do6 += do6
        fo.write('ocropy_ridges2' + str(do7) + '\n')
        total_do7 += do7


        fo.write('\n#full words recognized by models, higher is better\n')
        fo.write('total_gt_words: ' + str(gtwordcount) + '\n')
        total_gtwordcount += gtwordcount

        fo.write('calamari: ' + str(dcfull) + '\n')
        total_dcfull += dcfull

        fo.write('tesser_eng: ' + str(dt1full) + '\n')
        total_dt2full += dt1full
        fo.write('tesser_deu_frak: ' + str(dt2full) + '\n')
        total_dt2full += dt2full
        fo.write('tesser_deu_frak+eng: ' + str(dt3full) + '\n')
        total_dt3full += dt3full

        fo.write('ocropy_en-default: ' + str(do1full) + '\n')
        total_do1 += do1full
        fo.write('ocropy_fraktur:' + str(do2full) + '\n')
        total_do2 += do2full

        fo.write('ocropy_incunabula' + str(do3full) + '\n')
        total_do3 += do3full
        fo.write('ocropy_latin1' + str(do4full) + '\n')
        total_do4 += do4full
        fo.write('ocropy_latin2' + str(do5full) + '\n')
        total_do5 += do5full
        fo.write('ocropy_ridges1' + str(do6full) + '\n')
        total_do6 += do6full
        fo.write('ocropy_ridges2' + str(do7full) + '\n')
        total_do7 += do7full


        fo.write('\n#accuracy achieved by models, higher is better\n')

        fo.write('calamari: ' + str(accc) + '\n')

        fo.write('tesser_eng: ' + str(acct1) + '\n')
        fo.write('tesser_deu_frak: ' + str(acct2) + '\n')
        fo.write('tesser_deu_frak+eng: ' + str(acct3) + '\n')

        fo.write('ocropy_en-default: ' + str(acco1) + '\n')
        fo.write('ocropy_fraktur:' + str(acco2) + '\n')

        fo.write('ocropy_incunabula' + str(acco3) + '\n')
        fo.write('ocropy_latin1' + str(acco4) + '\n')
        fo.write('ocropy_latin2' + str(acco5) + '\n')
        fo.write('ocropy_ridges1' + str(acco6) + '\n')
        fo.write('ocropy_ridges2' + str(acco7) + '\n')


#total counts

total_accc = 1-total_dc/total_gtcount

total_acco1 = 1-total_do1/total_gtcount
total_acco2 = 1-total_do2/total_gtcount

total_acco3 = 1-total_do3/total_gtcount
total_acco4 = 1-total_do4/total_gtcount
total_acco5 = 1-total_do5/total_gtcount
total_acco6 = 1-total_do6/total_gtcount
total_acco7 = 1-total_do7/total_gtcount

total_acct1 = 1-total_dt1/total_gtcount
total_acct2 = 1-total_dt2/total_gtcount
total_acct3 = 1-total_dt3/total_gtcount
with open(path + '0000_eval.txt', 'w+') as fo:
    fo.write('#token errors made by models, lower is better\n')
    fo.write('total_gt_tokens: ' + str(total_gtcount) + '\n')

    fo.write('calamari: ' + str(total_dc) + '\n')

    fo.write('tesser_eng: ' + str(total_dt1) + '\n')
    fo.write('tesser_deu_frak: ' + str(total_dt2) + '\n')
    fo.write('tesser_deu_frak+eng: ' + str(total_dt3) + '\n')

    fo.write('ocropy_en-default: ' + str(total_do1) + '\n')
    fo.write('ocropy_fraktur:' + str(total_do2) + '\n')

    fo.write('ocropy_incunabula' + str(total_do3) + '\n')
    fo.write('ocropy_latin1' + str(total_do4) + '\n')
    fo.write('ocropy_latin2' + str(total_do5) + '\n')
    fo.write('ocropy_ridges1' + str(total_do6) + '\n')
    fo.write('ocropy_ridges2' + str(total_do7) + '\n')

    fo.write('\n#full words recognized by models, higher is better\n')
    fo.write('total_gt_words: ' + str(total_gtwordcount) + '\n')

    fo.write('calamari: ' + str(total_dcfull) + '\n')

    fo.write('tesser_eng: ' + str(total_dt1full) + '\n')
    fo.write('tesser_deu_frak: ' + str(total_dt2full) + '\n')
    fo.write('tesser_deu_frak+eng: ' + str(total_dt3full) + '\n')

    fo.write('ocropy_en-default: ' + str(total_do1full) + '\n')
    fo.write('ocropy_fraktur:' + str(total_do2full) + '\n')

    fo.write('ocropy_incunabula' + str(total_do3full) + '\n')
    fo.write('ocropy_latin1' + str(total_do4full) + '\n')
    fo.write('ocropy_latin2' + str(total_do5full) + '\n')
    fo.write('ocropy_ridges1' + str(total_do6full) + '\n')
    fo.write('ocropy_ridges2' + str(total_do7full) + '\n')

    fo.write('\n#accuracy achieved by models, higher is better\n')

    fo.write('calamari: ' + str(total_accc) + '\n')

    fo.write('tesser_eng: ' + str(total_acct1) + '\n')
    fo.write('tesser_deu_frak: ' + str(total_acct2) + '\n')
    fo.write('tesser_deu_frak+eng: ' + str(total_acct3) + '\n')

    fo.write('ocropy_en-default: ' + str(total_acco1) + '\n')
    fo.write('ocropy_fraktur:' + str(total_acco2) + '\n')

    fo.write('ocropy_incunabula' + str(total_acco3) + '\n')
    fo.write('ocropy_latin1' + str(total_acco4) + '\n')
    fo.write('ocropy_latin2' + str(total_acco5) + '\n')
    fo.write('ocropy_ridges1' + str(total_acco6) + '\n')
    fo.write('ocropy_ridges2' + str(total_acco7) + '\n')

print('done')