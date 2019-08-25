from __future__ import absolute_import

import os.path
import sys
import io
import warnings
import logging

import numpy as np
from scipy.ndimage import measurements, filters, interpolation, morphology
from scipy import stats
from PIL import Image, ImageDraw, ImageStat

from ocrd_cis.ocropy import ocrolib
from ocrd_cis.ocropy.ocrolib import lineest, morph, psegutils, sl

from ocrd_models import OcrdExif
from ocrd_utils import getLogger, xywh_from_points, polygon_from_points

LOG = getLogger('') # to be refined by importer

# method similar to ocrolib.read_image_gray
def pil2array(image):
    """Convert image to floating point grayscale array.
    
    Given a PIL.Image instance (of any colorspace),
    convert to a grayscale Numpy array
    (with 1.0 for white and 0.0 for black).
    """
    assert isinstance(image, Image.Image), "not a PIL.Image"
    array = ocrolib.pil2array(image)
    if array.dtype == np.uint8:
        array = array / 255.0
    if array.dtype == np.int8:
        array = array / 127.0
    elif array.dtype == np.uint16:
        array = array / 65536.0
    elif array.dtype == np.int16:
        array = array / 32767.0
    elif np.issubdtype(array.dtype, np.floating):
        pass
    else:
        raise Exception("unknown image type: " + array.dtype)
    if array.ndim == 3:
        array = np.mean(array, 2)
    return array

def array2pil(array):
    """Convert floating point grayscale array to an image.
    
    Given a grayscale Numpy array
    (with 1.0 for white and 0.0 for black),
    convert to a (grayscale) PIL.Image instance.
    """
    assert isinstance(array, np.ndarray), "not a numpy array"
    array = np.array(255.0 * array, np.uint8)
    return ocrolib.array2pil(array)

# from ocropy-nlbin, but keeping exact size
def estimate_local_whitelevel(image, zoom=0.5, perc=80, range_=20):
    '''flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    range for filters, default: %(default)s
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = interpolation.zoom(image, zoom, mode='nearest')
        m = filters.percentile_filter(m, perc, size=(range_, 2))
        m = filters.percentile_filter(m, perc, size=(2, range_))
        m = interpolation.zoom(m, 1. / zoom)
    #w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    #flat = np.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)
    # we want to get exactly the same size as before:
    h, w = image.shape
    h0, w0 = m.shape
    m = np.pad(m, ((max(0, h-h0),0), (max(0, w-w0),0)), 'edge')[:h, :w]
    flat = np.nan_to_num(np.clip(image - m + 1, 0, 1))
    return flat

# from ocropy-nlbin, but with threshold on variance
def estimate_skew_angle(image, angles):
    estimates = np.zeros_like(angles)
    for i, a in enumerate(angles):
        v = np.mean(interpolation.rotate(image, a, order=0, mode='constant'), axis=1)
        v = np.var(v)
        estimates[i] = v
    # only return the angle of the largest entropy,
    # if it is considerably larger than the average:
    if np.amax(estimates) / np.mean(estimates) > 1.5:
        return angles[np.argmax(estimates)]
    else:
        return 0

# from ocropy-nlbin, but with reshape=True
def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    '''estimate skew angle and rotate'''
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1) # border ignore
    flat = np.amax(flat) - flat
    #flat -= np.amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    ma = maxskew
    ms = int(2 * maxskew * skewsteps)
    angles = np.linspace(-ma, ma, ms + 1)
    # if d0 > d1 and d0 * d1 > 300:
    #     angles = np.concatenate([angles,
    #                              angles + 90])
    angle = estimate_skew_angle(est, angles)
    # we must allow reshape/expand to avoid loosing information in the corners
    # (but this also means that consumers of the AlternativeImage must
    #  offset coordinates by half the increased width/height besides
    #  correcting for rotation in the coordinates):
    flat = interpolation.rotate(flat, angle, mode='constant', reshape=True)
    flat = np.amax(flat) - flat
    return flat, angle

# from ocropy-nlbin
def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90):
    '''estimate low and high thresholds.
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    '''
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    if escale > 0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est - filters.gaussian_filter(est, e * 20.0)
        v = filters.gaussian_filter(v ** 2, e * 20.0) ** 0.5
        v = (v > 0.3 * np.amax(v))
        v = morphology.binary_dilation(v, structure=np.ones((int(e * 50), 1)))
        v = morphology.binary_dilation(v, structure=np.ones((1, int(e * 50))))
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(), lo)
    hi = stats.scoreatpercentile(est.ravel(), hi)
    return lo, hi

# from ocropy-nlbin process1, but reshape when rotating, and catch NaN
def binarize(image, 
             perc=90, # percentage for local whitelevel filters (ocropy: 80)
             range=10, # range (in pixels) for local whitelevel filters (ocropy: 20)
             zoom=1.0, # zoom factor for local whitelevel filters (ocropy: 0.5)
             escale=1.0, # exp scale for region mask during threshold estimation
             bignore=0.1, # ignore this much of the border for threshold estimation
             threshold=0.5, # final binarization threshold (ocropy: 0.5)
             lo=5, # percentile for black estimation (ocropy: 5)
             hi=90, # percentile for white estimation (ocropy: 90)
             maxskew=2, # maximum angle (in degrees) for skew estimation
             skewsteps=8, # steps per degree
             nrm=False): # output grayscale normalized
    extreme = (np.sum(image < 0.05) + np.sum(image > 0.95)) * 1.0 / np.prod(image.shape)
    if extreme > 0.95:
        comment = "no-normalization"
        flat = image
    else:
        comment = ""
        # if not, we need to flatten it by estimating the local whitelevel
        flat = estimate_local_whitelevel(image, zoom, perc, range)
    if maxskew > 0:
        flat, angle = estimate_skew(flat, bignore, maxskew, skewsteps)
    else:
        angle = 0
    lo, hi = estimate_thresholds(flat, bignore, escale, lo, hi)
    if np.isnan(lo) or np.isnan(hi):
        LOG.warning("cannot estimate binarization thresholds (is the image empty?)")
    else:
        # normalize the image:
        flat -= lo
        if hi - lo:
            flat /= (hi - lo)
    flat = np.clip(flat, 0, 1)
    bin = 1 * (flat > threshold)
    LOG.debug("binarization: lo-hi (%.2f %.2f) angle %4.1f %s", lo, hi, angle, comment)
    return flat if nrm else bin, angle

# inspired by OLD/ocropus-lattices --borderclean
def borderclean(array, margin=4):
    """Remove components that are only contained within the margin.
    
    Given a grayscale-normalized image as Numpy array `array`, 
    find and remove those black components that do not exceed
    the top/bottom margins.
    (This can be used to get rid of ascenders and descenders from
    neighbouring regions/lines, which cannot be found by resegment()
    within the region/line.
    
    This should happen before deskewing, because the latter increases
    the margins by an unpredictable amount.)
    
    Return a Numpy array (with 0 for black and 1 for white).
    """
    binary = np.array(array <= ocrolib.midrange(array), np.uint8)
    return np.maximum(borderclean_bin(binary, margin=margin), array)

def borderclean_bin(binary, margin=4):
    """Remove components that are only contained within the margin.
    
    Given a binarized, inverted image as Numpy array `binary`, 
    find those foreground components that do not exceed
    the top/bottom margins, and return a mask to remove them.
    (This can be used to get rid of ascenders and descenders from
    neighbouring regions/lines, which cannot be found by resegment()
    within the region/line.
    
    This should happen before deskewing, because the latter increases
    the margins by an unpredictable amount.)
    
    Return a Numpy array (with 0 for stay and 1 for remove).
    """
    h, w = binary.shape
    h1, w1 = min(margin, h//2 - 4), min(margin, w//2 - 4)
    h2, w2 = -h1 if h1 else h, -w1 if w1 else w
    mask = np.zeros(binary.shape, 'i')
    mask[h1:h2, w1:w2] = 1
    # use only the vertical margins (to which recognition is sensitive):
    # mask[h1:h2, :] = 1
    _, ncomps_before = morph.label(binary)
    binary = morph.keep_marked(binary, mask)
    _, ncomps_after = morph.label(binary)
    LOG.debug('black components before/after borderclean (margin=%d): %d/%d',
              margin, ncomps_before, ncomps_after)
    return (1 - binary)*(1 - mask)

# from ocropus-rpred, but with zoom parameter
def check_line(binary, zoom=1.0):
    """Validate binary as a plausible text line image.
    
    Given a binarized, inverted image as Numpy array `binary`
    (with 0 for white and 1 for black),
    check the array has the right dimensions, is in fact inverted,
    and does not have too few or too many connected black components.
    
    Returns an error report, or None if valid.
    """
    if len(binary.shape)==3: return "input image is color image %s"%(binary.shape,)
    if np.mean(binary)<np.median(binary): return "image may be inverted"
    h,w = binary.shape
    if h<20/zoom: return "image not tall enough for a text line %s"%(binary.shape,)
    if h>200/zoom: return "image too tall for a text line %s"%(binary.shape,)
    #if w<1.5*h: return "line too short %s"%(binary.shape,)
    if w<1.5*h and w<32/zoom: return "line too short %s"%(binary.shape,)
    if w>4000/zoom: return "line too long %s"%(binary.shape,)
    ratio = w*1.0/h
    _, ncomps = measurements.label(binary)
    lo = int(0.5*ratio+0.5)
    hi = int(4*ratio)+1
    if ncomps<lo: return "too few connected components (got %d, wanted >=%d)"%(ncomps,lo)
    #if ncomps>hi*ratio: return "too many connected components (got %d, wanted <=%d)"%(ncomps,hi)
    if ncomps>hi*ratio and ncomps>10: return "too many connected components (got %d, wanted <=%d)"%(ncomps,hi)
    return None

# inspired by ocropus-gpageseg check_page
def check_region(binary, zoom=1.0):
    """Validate binary as a plausible text region image.
    
    Given a binarized, inverted image as Numpy array `binary`
    (with 0 for white and 1 for black),
    check the array has the right dimensions, is in fact inverted,
    and does not have too few or too many connected black components.
    
    Returns an error report, or None if valid.
    """
    if len(binary.shape)==3: return "input image is color image %s"%(binary.shape,)
    if np.mean(binary)<np.median(binary): return "image may be inverted"
    h,w = binary.shape
    if h<60/zoom: return "image not tall enough for a region image %s"%(binary.shape,)
    if h>5000/zoom: return "image too tall for a region image %s"%(binary.shape,)
    if w<100/zoom: return "image too narrow for a region image %s"%(binary.shape,)
    if w>5000/zoom: return "line too wide for a region image %s"%(binary.shape,)
    slots = int(w*h*1.0/(30*30)*zoom*zoom)
    _,ncomps = measurements.label(binary)
    if ncomps<5: return "too few connected components for a region image (got %d)"%(ncomps,)
    if ncomps>slots*2 and ncomps>10: return "too many connnected components for a region image (%d > %d)"%(ncomps,slots)
    return None

# from ocropus-gpageseg, but with zoom parameter
def check_page(binary, zoom=1.0):
    """Validate binary as a plausible printed text page image.
    
    Given a binarized, inverted image as Numpy array `binary`
    (with 0 for white and 1 for black),
    check the array has the right dimensions, is in fact inverted,
    and does not have too few or too many connected black components.
    
    Returns an error report, or None if valid.
    """
    if len(binary.shape)==3: return "input image is color image %s"%(binary.shape,)
    if np.mean(binary)<np.median(binary): return "image may be inverted"
    h,w = binary.shape
    if h<600/zoom: return "image not tall enough for a page image %s"%(binary.shape,)
    if h>10000/zoom: return "image too tall for a page image %s"%(binary.shape,)
    if w<600/zoom: return "image too narrow for a page image %s"%(binary.shape,)
    if w>10000/zoom: return "line too wide for a page image %s"%(binary.shape,)
    slots = int(w*h*1.0/(30*30)*zoom*zoom)
    _,ncomps = measurements.label(binary)
    if ncomps<10: return "too few connected components for a page image (got %d)"%(ncomps,)
    if ncomps>slots and ncomps>10: return "too many connnected components for a page image (%d > %d)"%(ncomps,slots)
    return None

def odd(num):
    return num + (num+1)%2

# from ocropus-gpageseg
def DSAVE(title,array, interactive=False):
    logging.getLogger('matplotlib').setLevel(logging.WARNING) # workaround
    from matplotlib import pyplot as plt
    from matplotlib import patches as mpatches
    from tempfile import mkstemp
    if type(array)==list:
        # 3 inputs, one for each RGB channel
        assert len(array)==3
        array = np.transpose(np.array(array),[1,2,0])
    if interactive:
        plt.imshow(array.astype('float'))
        plt.legend(handles=[mpatches.Patch(label=title)])
        plt.show()
    else:
        _,fname = mkstemp(suffix=title+".png")
        plt.imsave(fname,array.astype('float'))
        LOG.debug('DSAVE %s', fname)

# from ocropus-gpageseg, but with extra height criterion
def remove_hlines(binary,scale,maxsize=10):
    labels,_ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if (sl.width(b)>maxsize*scale and 
            sl.height(b)<scale):
            labels[b][labels[b]==i+1] = 0
    result = np.array(labels!=0,'B')
    #DSAVE('hlines', binary-result)
    return result

# from ocropus-gpageseg, but with different thresholds, and remove only connected components fully inside seps
def compute_separators_morph(binary,scale, sepwiden=10, maxseps=0):
    """Finds vertical black lines corresponding to column separators."""
    # FIXME: make zoomable
    d0 = int(max(5,scale/4))
    d1 = int(max(5,scale))+ sepwiden
    thick = morph.r_dilation(binary,(d0,d1))
    # 5 instead of 10, because black colseps can also be discontinous:
    vert = morph.rb_opening(thick,(5*scale,1))
    vert = morph.r_erosion(vert,(d0//2, sepwiden))
    vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2* maxseps)
    # 8 instead of 20, because black colseps can also be discontinous:
    vert = morph.select_regions(vert,sl.dim0,min=8*scale,nbest=maxseps)
    # reduce to the connected components properly contained in seps:
    vert = morph.propagate_labels(binary, vert+1, conflict=0)-1
    return vert

# from ocropus-gpagseg
def compute_colseps_conv(binary,scale=1.0, csminheight=10, maxcolseps=2):
    """Find column separators by convolution and
    thresholding."""
    # FIXME: make zoomable
    h,w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = filters.gaussian_filter(1.0*binary,(scale,scale*0.5))
    smoothed = filters.uniform_filter(smoothed,(5.0*scale,1))
    thresh = (smoothed<np.amax(smoothed)*0.1)
    #DSAVE("1thresh",thresh)
    # find column edges by filtering
    grad = filters.gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
    grad = filters.uniform_filter(grad,(10.0*scale,1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad>0.5*np.amax(grad))
    #DSAVE("2grad",grad)
    # combine edges and whitespace
    seps = np.minimum(thresh,filters.maximum_filter(grad,(int(scale),int(5*scale))))
    seps = filters.maximum_filter(seps,(int(2*scale),1))
    #DSAVE("3seps",seps)
    # select only the biggest column separators
    seps = morph.select_regions(seps,sl.dim0,min=csminheight*scale,nbest=maxcolseps)
    #DSAVE("4seps",seps)
    return seps

# from ocropus-gpageseg, but without apply_mask (i.e. input file)
def compute_colseps(binary,scale, maxcolseps=3, maxseps=0):
    """Computes column separators either from vertical black lines or whitespace."""
    LOG.debug("considering at most %g whitespace column separators", maxcolseps)
    colseps = compute_colseps_conv(binary,scale, maxcolseps=maxcolseps)
    #DSAVE("colwsseps",0.7*colseps+0.3*binary)
    if maxseps > 0:
        LOG.debug("considering at most %g black column separators", maxseps)
        seps = compute_separators_morph(binary,scale/2, maxseps=maxseps)
        #DSAVE("colseps", [binary, colseps, seps])
        #colseps = compute_colseps_morph(binary,scale)
        colseps = np.maximum(colseps,seps)
        binary = np.minimum(binary,1-seps)
    #binary,colseps = apply_mask(binary,colseps)
    return colseps,binary

# from ocropus-gpageseg, but with smaller box minsize and less horizontal blur
def compute_gradmaps(binary,scale,
                     usegauss=False,vscale=1.0,hscale=1.0):
    # use gradient filtering to find baselines
    # do not use the default boxmap scale thresholds (0.5,4),
    # because we will have remainders of (possibly rotated and)
    # chopped lines at the region boundaries,
    # which we want to regard as neighbouring full lines
    # when estimating segmentation to re-segment (and mask) lines:    
    boxmap = psegutils.compute_boxmap(binary,scale, threshold=(0.1,4))
    #DSAVE("boxmap",boxmap)
    cleaned = boxmap*binary
    #DSAVE("cleaned",cleaned)
    if usegauss:
        # this uses Gaussians
        grad = filters.gaussian_filter(
            1.0*cleaned,
            (vscale*0.3*scale,
             hscale*scale),
            #hscale*6*scale),
            order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = filters.gaussian_filter(
            1.0*cleaned,
            (max(4,vscale*0.3*scale),
             hscale*scale),
            order=(1,0))
        grad = filters.uniform_filter(
            grad,
            (vscale,hscale*scale))
            #(vscale,hscale*6*scale))
    #DSAVE("grad", grad)
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    #DSAVE("bottom+top+boxmap", [0.5*boxmap + binary, bottom, top])
    return bottom,top,boxmap

# from ocropus-gpageseg, but with robust switch
def compute_line_seeds(binary,bottom,top,colseps,scale,
                       threshold=0.2,vscale=1.0,
                       # more robust top/bottom transition rules:
                       robust=True):
    """Based on gradient maps, compute candidates for baselines and xheights.
    Then, mark the regions between the two as a line seed. Finally, label
    all connected regions (starting with 1, with 0 as background)."""
    # FIXME: make zoomable
    vrange = int(vscale*scale)
    # find (more or less) horizontal lines along the maximum gradient,
    # where it is above (squared) threshold and not crossing columns:
    bmarked = filters.maximum_filter(
        # mark position of maximum gradient every `vrange` pixels:
        bottom==filters.maximum_filter(bottom,(vrange,0)),
        # blur by 2 pixels, then retain only large gradients:
        (2,2)) * (bottom>threshold*np.amax(bottom)*threshold) *(1-colseps)
    tmarked = filters.maximum_filter(
        # mark position of maximum gradient every `vrange` pixels:
        top==filters.maximum_filter(top,(vrange,0)),
        # blur by 2 pixels, then retain only large gradients:
        (2,2)) * (top>threshold*np.amax(top)*threshold/2) *(1-colseps)
    if robust:
        bmarked = filters.maximum_filter(bmarked,(1,scale//2))
    tmarked = filters.maximum_filter(tmarked,(1,scale//2))
    #tmarked = filters.maximum_filter(tmarked,(1,20))
    seeds = np.zeros(binary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        # sort both kinds of mark from bottom to top (i.e. inverse y position)
        transitions = sorted([(y,1) for y in psegutils.find(bmarked[:,x])] +
                             [(y,0) for y in psegutils.find(tmarked[:,x])])[::-1]
        if robust:
            l = 0
            while l < len(transitions):
                y0, s0 = transitions[l]
                if s0: # bmarked?
                    y1 = max(0, y0 - delta) # project seed from bottom
                    if l+1 < len(transitions) and transitions[l+1][0] > y1:
                        y1 = transitions[l+1][0] # fill with seed to next mark
                    seeds[y1:y0, x] = 1
                else: # tmarked?
                    y1 = y0 + delta # project seed from top
                    if l > 0 and transitions[l-1][0] < y1:
                        y1 = transitions[l-1][0] # fill with seed to next mark
                    seeds[y0:y1, x] = 1
                l += 1
        else:
            transitions += [(0,0)]
            for l in range(len(transitions)-1):
                y0,s0 = transitions[l]
                if s0==0:
                    continue # keep looking for next bottom
                seeds[y0-delta:y0,x] = 1 # project seed from bottom
                y1,s1 = transitions[l+1]
                if s1==0 and (y0-y1)<5*scale: # why 5?
                    # consistent next top?
                    seeds[y1:y0,x] = 1 # fill with seed completely
    if not robust:
        # commented to avoid smearing into neighbouring line components at as/descenders
        # (horizontal consistency will achieved by hmerge and spread):
        seeds = filters.maximum_filter(seeds,(1,int(1+scale)))
    seeds = seeds*(1-colseps)
    #DSAVE("lineseeds unlabelled",[0.4*seeds+0.5*binary, bmarked, tmarked])
    seeds, nlabels = morph.label(seeds)
    for i in range(nlabels):
        ys, xs = np.nonzero(seeds == i+1)
        height = np.amax(ys) - np.amin(ys)
        if height > 2 * scale:
            LOG.warning('line %d has extreme height (%d vs %d)', i+1, height, scale)
    return seeds

def hmerge_line_seeds(seeds, colseps, threshold=0.2):
    """Relabel line seeds such that regions of coherent vertical
    intervals get the same label."""
    # merge labels horizontally to avoid splitting lines at long whitespace
    # (to prevent corners from becoming the largest label when spreading
    #  into the background; and make contiguous contours possible), but
    # ignore conflicts which affect only small fractions of either line
    # (avoiding merges for small vertical overlap):
    relabel = np.unique(seeds)
    labels = np.delete(relabel, 0) # without background
    labels_count = dict([(label, np.sum(seeds == label)) for label in labels])
    overlap_count = dict([(label, (0, label)) for label in labels])
    for label in labels:
        # get maximum horizontal spread for current label:
        mask = filters.maximum_filter(seeds == label, (1, seeds.shape[1]))
        # get overlap between other labels and mask (but discount colseps):
        candidates, counts = np.unique(seeds * mask * (1-colseps), return_counts=True)
        for candidate, count in zip(candidates, counts):
            if (candidate and candidate != label and
                overlap_count[candidate][0] < count):
                overlap_count[candidate] = (count, label)
    for label in labels:
        # get candidate with largest overlap:
        count, candidate = overlap_count[label]
        if (candidate != label and
            count / labels_count[label] > threshold):
            # the new label could have been relabelled already:
            new_label = relabel[candidate]
            # assign label to (new assignment for) candidate:
            relabel[label] = new_label
            # re-assign labels already relabelled to label:
            relabel[relabel == label] = new_label
            # fill the horizontal background between both regions:
            label_y, label_x = np.where(seeds == label)
            new_label_y, new_label_x = np.where(seeds == new_label)
            for y in np.intersect1d(label_y, new_label_y):
                x_min = label_x[label_y == y][0]
                x_max = label_x[label_y == y][-1]
                new_x_min = new_label_x[new_label_y == y][0]
                new_x_max = new_label_x[new_label_y == y][-1]
                if x_max < new_x_min:
                    seeds[y, x_max:new_x_min] = label
                if new_x_max < x_min:
                    seeds[y, new_x_max:x_min] = label
    # apply re-assignments:
    seeds = relabel[seeds]
    #DSAVE("lineseeds_hmerged", seeds)
    return seeds

# like ocropus-gpageseg compute_segmentation, but:
# - with fullpage switch and zoom parameter,
# - with twice the estimated scale,
# - with horizontal merge instead of blur,
# - with component majority for foreground
#   outside of seeds (instead of spread), and
# - without final unmasking
def compute_line_labels(array, fullpage=False, zoom=1.0, maxcolseps=2, maxseps=0, check=True):
    """Find text line segmentation within a region or page.
    
    Given a grayscale-normalized image as Numpy array `array`, compute
    a complete segmentation into text lines for it, avoiding any single
    horizontal splits. 
    If `fullpage` is True, then also find (and suppress) horizontal lines,
    and up to `maxcolseps` white-space and `maxseps` black column separators
    (both counted by connected components).
    
    Return a Numpy array of the background labels
    (not the foreground or the masked image).
    """
    if np.prod(array.shape) == 0:
        raise Exception('image dimensions are zero')
    if np.amax(array) == np.amin(array):
        raise Exception('image is blank')
    binary = np.array(array <= ocrolib.midrange(array), np.uint8)
    #DSAVE("binary",binary)
    if check:
        if fullpage:
            report = check_page(binary, zoom)
        else:
            report = check_region(binary, zoom)
        if report:
            raise Exception(report)
    scale = psegutils.estimate_scale(binary)
    # use larger scale so broken/blackletter fonts with their large capitals
    # are not cut into two lines or joined at ascenders/decsenders:
    scale *= 2
    LOG.debug('xheight: %d, scale: %d', binary.shape[0], scale)

    if fullpage:
        binary2 = remove_hlines(binary, scale, maxsize=3/zoom)
        hlines, binary = binary - binary2, binary2
    else:
        hlines = np.zeros_like(binary)
        
    bottom, top, boxmap = compute_gradmaps(binary, scale/2, usegauss=False,
                                           hscale=1.0/zoom, vscale=1.0/zoom)
    if fullpage:
        colseps, binary2 = compute_colseps(binary, scale, maxcolseps=maxcolseps, maxseps=maxseps)
        vlines, binary = binary - binary2, binary2
    else:
        colseps = np.zeros(binary.shape, np.uint8)
        vlines = np.zeros_like(binary)
    seeds = compute_line_seeds(binary, bottom, top, colseps, scale)
    #DSAVE("seeds",[boxmap,bottom,top])
    if not fullpage:
        seeds = hmerge_line_seeds(seeds, colseps)

    # assign the seeds labels to all component boxes
    # (boxes with conflicts will become background):
    llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
    # spread the seed labels to background and conflicts
    # (unassigned pixels will get the nearest label up to maxdist):
    spread = morph.spread_labels(seeds,maxdist=scale)
    #DSAVE('spread', spread + 0.6*binary)
    # background and conflict will get nearest spread label:
    llabels = np.where(llabels>0,llabels,spread)
    #DSAVE('llabels', llabels + 0.6*binary)
    # now improve the above procedure by ensuring that
    # no connected components are (unnecessarily) split;
    # the only connected components that must be split are
    # those conflicting in seeds (not those conflicting in spread);
    # those conflicting in spread should be given the label
    # with a majority of foreground pixels:
    llabels2 = morph.propagate_labels_majority(binary, llabels)
    llabels2 = np.where(seeds > 0, seeds, llabels2)
    seed_majority = morph.propagate_labels_majority(binary, seeds)
    seed_nonconflict = morph.propagate_labels(binary, seeds, conflict=0)
    seed_conflicts = seed_majority > seed_nonconflict
    # re-spread the component labels (more exact, and majority
    # leaders will not extrude):
    llabels = morph.spread_labels(np.where(seed_conflicts, llabels, llabels2), maxdist=scale)
    #DSAVE('llabels2', llabels + 0.6*binary)
    #segmentation = llabels*binary
    #return segmentation
    return llabels # , hlines, vlines

# from ocropus-gpageseg, but as separate step on the PIL.Image
def remove_noise(image, maxsize=8):
    array = pil2array(image)
    binary = np.array(array <= ocrolib.midrange(array), np.uint8)
    _, ncomps_before = morph.label(binary)
    clean = ocrolib.remove_noise(binary, maxsize)
    _, ncomps_after = morph.label(clean)
    LOG.debug('black components before/after denoising (maxsize=%d): %d/%d',
              maxsize, ncomps_before, ncomps_after)
    array = np.maximum(array, binary - clean)
    return array2pil(array)

# to be refactored into core (as function in ocrd_utils):
def polygon_mask(image, coordinates):
    """"Create a mask image of a polygon.
    
    Given a PIL.Image `image` (merely for dimensions), and
    a numpy array `polygon` of relative coordinates into the image,
    create a new image of the same size with black background, and
    fill everything inside the polygon hull with white.
    
    Return the new PIL.Image.
    """
    mask = Image.new('L', image.size, 0)
    if isinstance(coordinates, np.ndarray):
        coordinates = list(map(tuple, coordinates))
    ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=255)
    return mask

# to be refactored into core (as function in ocrd_utils):
def image_from_polygon(image, polygon):
    """"Mask an image with a polygon.
    
    Given a PIL.Image `image` and a numpy array `polygon`
    of relative coordinates into the image, put everything
    outside the polygon hull to the background. Since `image`
    is not necessarily binarized yet, determine the background
    from the median color (instead of white).
    
    Return a new PIL.Image.
    """
    mask = polygon_mask(image, polygon)
    # create a background image from its median color
    # (in case it has not been binarized yet):
    # array = np.asarray(image)
    # background = np.median(array, axis=[0, 1], keepdims=True)
    # array = np.broadcast_to(background.astype(np.uint8), array.shape)
    background = ImageStat.Stat(image).median[0]
    new_image = Image.new('L', image.size, background)
    new_image.paste(image, mask=mask)
    return new_image

# to be refactored into core (as function in ocrd_utils):
def crop_image(image, box=None):
    """"Crop an image to a rectangle, filling with background.
    
    Given a PIL.Image `image` and a list `box` of the bounding
    rectangle relative to the image, crop at the box coordinates,
    filling everything outside `image` with the background.
    (This covers the case where `box` indexes are negative or
    larger than `image` width/height. PIL.Image.crop would fill
    with black.) Since `image` is not necessarily binarized yet,
    determine the background from the median color (instead of
    white).
    
    Return a new PIL.Image.
    """
    # todo: perhaps we should issue a warning if we encounter this
    # (It should be invalid in PAGE-XML to extend beyond parents.)
    if not box:
        box = (0, 0, image.width, image.height)
    xywh = xywh_from_bbox(*box)
    background = ImageStat.Stat(image).median[0]
    new_image = Image.new(image.mode, (xywh['w'], xywh['h']),
                          background) # or 'white'
    new_image.paste(image, (-xywh['x'], -xywh['y']))
    return new_image

# to be refactored into core (as function in ocrd_utils):
def rotate_coordinates(polygon, angle, orig=np.array([0, 0])):
    """Apply a passive rotation transformation to the given coordinates.
    
    Given a numpy array `polygon` of points and a rotation `angle`,
    as well as a numpy array `orig` of the center of rotation,
    calculate the coordinate transform corresponding to the rotation
    of the underlying image by `angle` degrees at `center` by
    applying translation to the center, inverse rotation,
    and translation from the center.

    Return a numpy array of the resulting polygon.
    """
    angle = np.deg2rad(angle)
    cos = np.cos(angle)
    sin = np.sin(angle)
    # active rotation:  [[cos, -sin], [sin, cos]]
    # passive rotation: [[cos, sin], [-sin, cos]] (inverse)
    return orig + np.dot(polygon - orig, np.array([[cos, sin], [-sin, cos]]).transpose())

# to be refactored into core (as method of ocrd.workspace.Workspace):
def coordinates_of_segment(segment, parent_image, parent_xywh):
    """Extract the relative coordinates polygon of a PAGE segment element.
    
    Given a Region / TextLine / Word / Glyph `segment` and
    the PIL.Image of its parent Page / Region / TextLine / Word
    along with its bounding box, calculate the relative coordinates
    of the segment within the image. That is, shift all points from
    the offset of the parent, and (in case the parent was rotated,)
    rotate all points with the center of the image as origin.
    
    Return the rounded numpy array of the resulting polygon.
    """
    # get polygon:
    polygon = np.array(polygon_from_points(segment.get_Coords().points))
    # offset correction (shift coordinates to base of segment):
    polygon -= np.array([parent_xywh['x'], parent_xywh['y']])
    # angle correction (rotate coordinates if image has been rotated):
    if 'angle' in parent_xywh:
        polygon = rotate_coordinates(
            polygon, parent_xywh['angle'],
            orig=np.array([0.5 * parent_image.width,
                           0.5 * parent_image.height]))
    return np.round(polygon).astype(np.int32)

# to be refactored into core (as method of ocrd.workspace.Workspace):
def coordinates_for_segment(polygon, parent_image, parent_xywh):
    """Convert a relative coordinates polygon to absolute.
    
    Given a numpy array `polygon` of points, and a parent PIL.Image
    along with its bounding box to which the coordinates are relative,
    calculate the absolute coordinates within the page.
    That is, (in case the parent was rotated,) rotate all points in
    opposite direction with the center of the image as origin, then
    shift all points to the offset of the parent.
    
    Return the rounded numpy array of the resulting polygon.
    """
    # angle correction (unrotate coordinates if image has been rotated):
    if 'angle' in parent_xywh:
        polygon = rotate_coordinates(
            polygon, -parent_xywh['angle'],
            orig=np.array([0.5 * parent_image.width,
                           0.5 * parent_image.height]))
    # offset correction (shift coordinates from base of segment):
    polygon += np.array([parent_xywh['x'], parent_xywh['y']])
    return np.round(polygon).astype(np.int32)

# to be refactored into core (as method of ocrd.workspace.Workspace):
def image_from_page(workspace, page, page_id):
    """Extract the Page image from the workspace.
    
    Given a PageType object, `page`, extract its PIL.Image from
    AlternativeImage if it exists. Otherwise extract the PIL.Image
    from imageFilename and crop it if a Border exists. Otherwise
    just return it.
    
    When cropping, respect any orientation angle annotated for
    the page (from page-level deskewing) by rotating the
    cropped image, respectively.
    
    If the resulting page image is larger than the bounding box of
    `page`, pass down the page's box coordinates with an offset of
    half the width/height difference.
    
    Return the extracted image, and the absolute coordinates of
    the page's bounding box / border (for passing down), and
    an OcrdExif instance associated with the original image.
    """
    page_image = workspace.resolve_image_as_pil(page.imageFilename)
    page_image_info = OcrdExif(page_image)
    page_xywh = {'x': 0,
                 'y': 0,
                 'w': page_image.width,
                 'h': page_image.height}
    page_xywh['angle'] = -(page.get_orientation() or 0)
    # FIXME: remove PrintSpace here as soon as GT abides by the PAGE standard:
    border = page.get_Border() or page.get_PrintSpace()
    if border:
        page_points = border.get_Coords().points
        LOG.debug("Using explictly set page border '%s' for page '%s'",
                  page_points, page_id)
        page_xywh = xywh_from_points(page_points)
    
    alternative_image = page.get_AlternativeImage()
    if alternative_image:
        # (e.g. from page-level cropping, binarization, deskewing or despeckling)
        # assumes implicit cropping (i.e. page_xywh has been applied already)
        LOG.debug("Using AlternativeImage %d (%s) for page '%s'",
                  len(alternative_image), alternative_image[-1].get_comments(),
                  page_id)
        page_image = workspace.resolve_image_as_pil(
            alternative_image[-1].get_filename())
    elif border:
        # get polygon outline of page border:
        page_polygon = np.array(polygon_from_points(page_points))
        # create a mask from the page polygon:
        page_image = image_from_polygon(page_image, page_polygon)
        # recrop into page rectangle:
        page_image = crop_image(page_image,
            box=(page_xywh['x'],
                 page_xywh['y'],
                 page_xywh['x'] + page_xywh['w'],
                 page_xywh['y'] + page_xywh['h']))
        if page_xywh['angle']:
            LOG.info("About to rotate page '%s' by %.2f°",
                      page_id, page_xywh['angle'])
            page_image = page_image.rotate(page_xywh['angle'],
                                           expand=True,
                                           #resample=Image.BILINEAR,
                                           fillcolor='white')
    # subtract offset from any increase in binary region size over source:
    page_xywh['x'] -= round(0.5 * max(0, page_image.width  - page_xywh['w']))
    page_xywh['y'] -= round(0.5 * max(0, page_image.height - page_xywh['h']))
    return page_image, page_xywh, page_image_info

# to be refactored into core (as method of ocrd.workspace.Workspace):
def image_from_segment(workspace, segment, parent_image, parent_xywh):
    """Extract a segment image from its parent's image.
    
    Given a PIL.Image of the parent, `parent_image`, and
    its absolute coordinates, `parent_xywh`, and a PAGE
    segment (TextRegion / TextLine / Word / Glyph) object
    logically contained in it, `segment`, extract its PIL.Image
    from AlternativeImage (if it exists), or via cropping from
    `parent_image`.
    
    When cropping, respect any orientation angle annotated for
    the parent (from parent-level deskewing) by compensating the
    segment coordinates in an inverse transformation (translation
    to center, rotation, re-translation).
    Also, mind the difference between annotated and actual size
    of the parent (usually from deskewing), by a respective offset
    into the image. Cropping uses a polygon mask (not just the
    rectangle).
    
    When cropping, respect any orientation angle annotated for
    the segment (from segment-level deskewing) by rotating the
    cropped image, respectively.
    
    If the resulting segment image is larger than the bounding box of
    `segment`, pass down the segment's box coordinates with an offset
    of half the width/height difference.
    
    Return the extracted image, and the absolute coordinates of
    the segment's bounding box (for passing down).
    """
    segment_xywh = xywh_from_points(segment.get_Coords().points)
    if 'orientation' in segment.__dict__:
        # angle: PAGE orientation is defined clockwise,
        # whereas PIL/ndimage rotation is in mathematical direction:
        segment_xywh['angle'] = -(segment.get_orientation() or 0)
    alternative_image = segment.get_AlternativeImage()
    if alternative_image:
        # (e.g. from segment-level cropping, binarization, deskewing or despeckling)
        LOG.debug("Using AlternativeImage %d (%s) for segment '%s'",
                  len(alternative_image), alternative_image[-1].get_comments(),
                  segment.id)
        segment_image = workspace.resolve_image_as_pil(
            alternative_image[-1].get_filename())
    else:
        # get polygon outline of segment relative to parent image:
        segment_polygon = coordinates_of_segment(segment, parent_image, parent_xywh)
        # create a mask from the segment polygon:
        segment_image = image_from_polygon(parent_image, segment_polygon)
        # recrop into segment rectangle:
        segment_image = crop_image(segment_image,
            box=(segment_xywh['x'] - parent_xywh['x'],
                 segment_xywh['y'] - parent_xywh['y'],
                 segment_xywh['x'] - parent_xywh['x'] + segment_xywh['w'],
                 segment_xywh['y'] - parent_xywh['y'] + segment_xywh['h']))
        # note: We should mask overlapping neighbouring segments here,
        # but finding the right clipping rules can be difficult if operating
        # on the raw (non-binary) image data alone: for each intersection, it
        # must be decided which one of either segment or neighbour to assign,
        # e.g. an ImageRegion which properly contains our TextRegion should be
        # completely ignored, but an ImageRegion which is properly contained
        # in our TextRegion should be completely masked, while partial overlap
        # may be more difficult to decide. On the other hand, on the binary image,
        # we can use connected component analysis to mask foreground areas which
        # originate in the neighbouring regions. But that would introduce either
        # the assumption that the input has already been binarized, or a dependency
        # on some ad-hoc binarization method. Thus, it is preferable to use
        # a dedicated processor for this (which produces clipped AlternativeImage
        # or reduced polygon coordinates).
        if 'angle' in segment_xywh and segment_xywh['angle']:
            LOG.info("About to rotate segment '%s' by %.2f°",
                      segment.id, segment_xywh['angle'])
            segment_image = segment_image.rotate(segment_xywh['angle'],
                                                 expand=True,
                                                 #resample=Image.BILINEAR,
                                                 fillcolor='white')
    # subtract offset from any increase in binary region size over source:
    segment_xywh['x'] -= round(0.5 * max(0, segment_image.width  - segment_xywh['w']))
    segment_xywh['y'] -= round(0.5 * max(0, segment_image.height - segment_xywh['h']))
    return segment_image, segment_xywh

# to be refactored into core (as method of ocrd.workspace.Workspace):
def save_image_file(workspace, image,
                    file_id,
                    page_id=None,
                    file_grp='OCR-D-IMG', # or -BIN?
                    format='PNG',
                    force=True):
    """Store and reference an image as file into the workspace.
    
    Given a PIL.Image `image`, and an ID `file_id` to use in METS,
    store the image under the fileGrp `file_grp` and physical page
    `page_id` into the workspace (in a file name based on
    the `file_grp`, `file_id` and `format` extension).
    
    Return the (absolute) path of the created file.
    """
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=format)
    file_path = os.path.join(file_grp,
                             file_id + '.' + format.lower())
    out = workspace.add_file(
        ID=file_id,
        file_grp=file_grp,
        pageId=page_id,
        local_filename=file_path,
        mimetype='image/' + format.lower(),
        content=image_bytes.getvalue(),
        force=force)
    LOG.info('created file ID: %s, file_grp: %s, path: %s',
             file_id, file_grp, out.local_filename)
    return file_path

# to be refactored into core (as function in ocrd_utils):
def bbox_from_points(points):
    """Construct a numeric list representing a bounding box from polygon coordinates in page representation."""
    xys = [[int(p) for p in pair.split(',')] for pair in points.split(' ')]
    return bbox_from_polygon(xys)

# to be refactored into core (as function in ocrd_utils):
def points_from_bbox(minx, miny, maxx, maxy):
    """Construct polygon coordinates in page representation from a numeric list representing a bounding box."""
    return "%i,%i %i,%i %i,%i %i,%i" % (
        minx, miny, maxx, miny, maxx, maxy, minx, maxy)

# to be refactored into core (as function in ocrd_utils):
def xywh_from_bbox(minx, miny, maxx, maxy):
    """Convert a bounding box from a numeric list to a numeric dict representation."""
    return {
        'x': minx,
        'y': miny,
        'w': maxx - minx,
        'h': maxy - miny,
    }

# to be refactored into core (as function in ocrd_utils):
def bbox_from_xywh(xywh):
    """Convert a bounding box from a numeric dict to a numeric list representation."""
    return (
        xywh['x'],
        xywh['y'],
        xywh['x'] + xywh['w'],
        xywh['y'] + xywh['h']
    )

# to be refactored into core (as function in ocrd_utils):
def points_from_polygon(polygon):
    """Convert polygon coordinates from a numeric list representation to a page representation."""
    return " ".join("%i,%i" % (x, y) for x, y in polygon)

# to be refactored into core (as function in ocrd_utils):
def xywh_from_polygon(polygon):
    """Construct a numeric dict representing a bounding box from polygon coordinates in numeric list representation."""
    return xywh_from_bbox(*bbox_from_polygon(polygon))

# to be refactored into core (as function in ocrd_utils):
def polygon_from_xywh(xywh):
    """Construct polygon coordinates in numeric list representation from numeric dict representing a bounding box."""
    return polygon_from_bbox(*bbox_from_xywh(xywh))

# to be refactored into core (as function in ocrd_utils):
def bbox_from_polygon(polygon):
    """Construct a numeric list representing a bounding box from polygon coordinates in numeric list representation."""
    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0
    for xy in polygon:
        if xy[0] < minx:
            minx = xy[0]
        if xy[0] > maxx:
            maxx = xy[0]
        if xy[1] < miny:
            miny = xy[1]
        if xy[1] > maxy:
            maxy = xy[1]
    return minx, miny, maxx, maxy

# to be refactored into core (as function in ocrd_utils):
def polygon_from_bbox(minx, miny, maxx, maxy):
    """Construct polygon coordinates in numeric list representation from a numeric list representing a bounding box."""
    return [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]

# to be refactored into core (as function in ocrd_utils):
def polygon_from_x0y0x1y1(x0y0x1y1):
    """Construct polygon coordinates in numeric list representation from a string list representing a bounding box."""
    minx = int(x0y0x1y1[0])
    miny = int(x0y0x1y1[1])
    maxx = int(x0y0x1y1[2])
    maxy = int(x0y0x1y1[3])
    return [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]

def membername(class_, val):
    """Convert a member variable/constant into a member name string."""
    return next((k for k, v in class_.__dict__.items() if v == val), str(val))
