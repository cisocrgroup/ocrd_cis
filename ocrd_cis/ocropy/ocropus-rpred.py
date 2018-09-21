#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import traceback
import codecs
import os.path
import argparse
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import measurements

import ocrolib
from ocrolib import lstm
from ocrolib.exceptions import FileNotFound, OcropusException

parser = argparse.ArgumentParser("apply an RNN recognizer")


# line dewarping (usually contained in model)
parser.add_argument("-e","--nolineest",action="store_true",
                    help="target line height (overrides recognizer)")
parser.add_argument("-l","--height",default=-1,type=int,
                    help="target line height (overrides recognizer)")

# recognition
parser.add_argument('-m','--model',default="en-default.pyrnn.gz",
                    help="line recognition model")
parser.add_argument("-p","--pad",default=16,type=int,
                    help="extra blank padding to the left and right of text line")



# debugging
parser.add_argument("-Q","--parallel",type=int,default=1,
                    help="number of parallel processes to use, default: %(default)s")

# input files
parser.add_argument("files",nargs="+",
                    help="input files; glob and @ expansion performed")
args = parser.parse_args()


def print_info(*objs):
    print("INFO: ", *objs, file=sys.stdout)


def print_error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)




# compute the list of files to be classified

if len(args.files)<1:
    parser.print_help()
    sys.exit(0)


inputs = ocrolib.glob_all(args.files)

# disable parallelism when anything is being displayed

# load the network used for classification

try:
    network = ocrolib.load_object(args.model,verbose=1)
    for x in network.walk(): x.postLoad()
    for x in network.walk():
        if isinstance(x,lstm.LSTM):
            x.allocate(5000)
except FileNotFound:
    print_error("")
    print_error("Cannot find OCR model file:" + args.model)
    print_error("Download a model and put it into:" + ocrolib.default.modeldir)
    print_error("(Or override the location with OCROPUS_DATA.)")
    print_error("")
    sys.exit(1)

# get the line normalizer from the loaded network, or optionally
# let the user override it (this is not very useful)

lnorm = getattr(network,"lnorm",None)

if args.height>0:
    lnorm.setHeight(args.height)


# process one file

def process1(arg):
    (trial,fname) = arg
    base,_ = ocrolib.allsplitext(fname)
    line = ocrolib.read_image_gray(fname)
    raw_line = line.copy()
    if np.prod(line.shape)==0: return None
    if np.amax(line)==np.amin(line): return None



    if not args.nolineest:
        assert "dew.png" not in fname,"don't dewarp dewarped images"
        temp = np.amax(line)-line
        temp = temp*1.0/np.amax(temp)
        lnorm.measure(temp)
        line = lnorm.normalize(line,cval=np.amax(line))
    else:
        assert "dew.png" in fname,"only apply to dewarped images"

    line = lstm.prepare_line(line,args.pad)
    pred = network.predictString(line).encode("utf-8")
    output = '<ocropy>' + pred + '</ocropy>'
    print (output)



def safe_process1(arg):
    trial,fname = arg
    try:
        return process1(arg)
    except IOError as e:
        if ocrolib.trace: traceback.print_exc()
        print_info(fname+":"+e)
    except ocrolib.OcropusException as e:
        if e.trace: traceback.print_exc()
        print_info(fname+":"+e)
    except:
        traceback.print_exc()
        return None

if args.parallel==0:
    result = []
    for trial,fname in enumerate(inputs):
        result.append(process1((trial,fname)))
elif args.parallel==1:
    result = []
    for trial,fname in enumerate(inputs):
        result.append(safe_process1((trial,fname)))
else:
    pool = Pool(processes=args.parallel)
    result = []
    for r in pool.imap_unordered(safe_process1,enumerate(inputs)):
        result.append(r)



