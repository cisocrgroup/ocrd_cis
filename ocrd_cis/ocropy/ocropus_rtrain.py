#!/usr/bin/env python

import re, traceback, sys
from PIL import Image

import random as pyrandom
import numpy as np
np.seterr(divide='raise',over='raise',invalid='raise',under='ignore')

import matplotlib.pyplot as plt


from ocrd_cis.ocropy import ocrolib
from ocrd_cis.ocropy.ocrolib import lstm, lineest



def rtrain(inputs, load, output, ntrain):

    #defaultvalues
    #extra blank padding to the left and right of text line, default: 16
    pad = 16

    #set the default height for line estimation, default: 48
    height=48 

    # Number of LSTM state units, default: 100
    hiddensize=100

    #LSTM learning rate, default: 1e-4
    lrate=0.0001
    
    quiet=False

    #LSTM save frequency, default: 1000
    savefreq=1000

    #manually set the number of already learned lines, which influences the naming and stopping condition, default: -1
    start=-1

    #strip the model before saving
    strip=True

    argclstm=False


    def resize_keep_ratio(image, baseheight):
        baseheight = 48
        hpercent = (baseheight / float(image.size[1]))
        wsize = int((float(image.size[0] * float(hpercent))))
        image = image.resize((wsize, baseheight), Image.ANTIALIAS)
        return image

    # make sure an output file has been set
    if output is None:
        print("you must give an output file with %d in it, or a prefix")
        sys.exit(0)

    if not "%" in output:
        if argclstm:
            oname = output+"-%08d.h5"
        else:
            oname = output+"-%08d.pyrnn"
    else:
        oname = output



    # load the line normalizer
    lnorm = lineest.CenterNormalizer()
    lnorm.setHeight(height)


    print("# using default codec")
    charset = sorted(list(set(list(lstm.ascii_labels) + list(ocrolib.chars.default))))

    charset = [""," ","~",]+[c for c in charset if c not in [" ","~"]]
    print("# charset size", len(charset), end=' ')
    if len(charset)<200:
        print("[" + "".join(charset) + "]")
    else:
        s = "".join(charset)
        print("[" + s[:20], "...", s[-20:] + "]")
    codec = lstm.Codec().init(charset)


    # Load an existing network or construct a new one
    # Somewhat convoluted logic for dealing with old style Python
    # modules and new style C++ LSTM networks.

    def save_lstm(fname,network):
        if argclstm:
            network.lstm.save(fname)
        else:
            if strip:
                network.clear_log()
                for x in network.walk(): x.preSave()
            ocrolib.save_object(fname,network)
            if strip:
                for x in network.walk(): x.postLoad()


    def load_lstm(fname):
        if argclstm:
            network = lstm.SeqRecognizer(height,hiddensize,
                codec=codec,
                normalize=lstm.normalize_nfkc)
            import clstm
            mylstm = clstm.make_BIDILSTM()
            mylstm.init(network.No,hiddensize,network.Ni)
            mylstm.load(fname)
            network.lstm = clstm.CNetwork(mylstm)
            return network
        else:
            network = ocrolib.load_object(last_save)
            network.upgrade()
            for x in network.walk(): x.postLoad()
            return network

    if load:
        print("# loading", load)
        last_save = load
        network = load_lstm(load)
    else:
        last_save = None
        network = lstm.SeqRecognizer(height,hiddensize,
            codec=codec,
            normalize=lstm.normalize_nfkc)
        if argclstm:
            import clstm
            mylstm = clstm.make_BIDILSTM()
            mylstm.init(network.No,hiddensize,network.Ni)
            network.lstm = clstm.CNetwork(mylstm)

    if getattr(network,"lnorm",None) is None:
        network.lnorm = lnorm

    network.upgrade()
    if network.last_trial%100==99: network.last_trial += 1
    print("# last_trial", network.last_trial)


    # set up the learning rate
    network.setLearningRate(lrate,0.9)

    start = start if start>=0 else network.last_trial

    for trial in range(start,ntrain):
        network.last_trial = trial+1

        do_update = 1

        fname = pyrandom.sample(inputs,1)[0]
        # print(inputs)
        # fname = inputs[0]
        # print(fname)
        
        base,_ = ocrolib.allsplitext(fname)

        line = ocrolib.read_image_gray(fname)
        transcript = ocrolib.read_text(base+".gt.txt")



        network.lnorm.measure(np.amax(line)-line)
        line = network.lnorm.normalize(line,cval=np.amax(line))

        if line.size<10 or np.amax(line)==np.amin(line):
            print("EMPTY-INPUT")
            continue
        line = line * 1.0/np.amax(line)
        line = np.amax(line)-line
        line = line.T
        if pad>0:
            w = line.shape[1]
            line = np.vstack([np.zeros((pad,w)),line,np.zeros((pad,w))])
        cs = np.array(codec.encode(transcript),'i')
        try:
            pcs = network.trainSequence(line,cs,update=do_update,key=fname)
        except FloatingPointError as e:
            print("# oops, got FloatingPointError", e)
            traceback.print_exc()
            network = load_lstm(last_save)
            continue
        except lstm.RangeError as e:
            continue
        pred = "".join(codec.decode(pcs))
        acs = lstm.translate_back(network.aligned)
        gta = "".join(codec.decode(acs))
        if not quiet:
            print("%d %.2f %s" % (trial, network.error, line.shape), fname)
            print("   TRU:", repr(transcript))
            print("   ALN:", repr(gta[:len(transcript)+5]))
            print("   OUT:", repr(pred[:len(transcript)+5]))

        pred = re.sub(' ','_',pred)
        gta = re.sub(' ','_',gta)

        if (trial+1)%savefreq==0:
            ofile = oname%(trial+1)+".gz"
            print("# saving", ofile)
            save_lstm(ofile,network)
            last_save = ofile
