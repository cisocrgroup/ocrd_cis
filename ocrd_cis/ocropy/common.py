from __future__ import absolute_import

import os.path
import sys
import io
import warnings

import numpy as np
from scipy.ndimage import measurements, filters, interpolation, morphology
from scipy import stats
from PIL import Image, ImageDraw

from ocrd_cis.ocropy import ocrolib
from ocrd_cis.ocropy.ocrolib import lineest, morph, psegutils, sl

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

# from ocropy-nlbin
def estimate_skew_angle(image, angles):
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image, a, order=0, mode='constant'), axis=1)
        v = np.var(v)
        estimates.append((v, a))
    _, a = max(estimates)
    return a

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

# from ocropy-nlbin process1, but reshape when rotating
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
             skewsteps=8): # steps per degree
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
    # rescale the image to get the gray scale image
    flat -= lo
    flat /= (hi - lo)
    flat = np.clip(flat, 0, 1)
    bin = 1 * (flat > threshold)
    LOG.debug("binarization: lo-hi (%.2f %.2f) angle %4.1f %s", lo, hi, angle, comment)
    return bin, angle

# inspired by OLD/ocropus-lattices --borderclean
def borderclean(array, margin=4):
    """Remove components that are only contained within the margin.
    
    Given a grayscale-normalized image as Numpy array `array`, 
    find and remove those black components that do not exceed
    the top/bottom margins.
    (This can be used to get rid of ascenders and descenders from
    neighbouring regions/lines, which cannot be found by resegment().
    But this should happen before deskewing, because this increases
    the margins by an unpredictable amount.)
    
    Return a Numpy array.
    """
    binary = np.array(array <= ocrolib.midrange(array), np.uint8)
    h, w = array.shape
    h1, w1 = min(margin, h//2 - 4), min(margin, w//2 - 4)
    h2, w2 = -h1 if h1 else h, -w1 if w1 else w
    mask = np.zeros(array.shape, 'i')
    #mask[h1:h2, w1:w2] = 1
    # use only the vertical margins (to which recognition is sensitive):
    mask[h1:h2, :] = 1
    _, ncomps_before = morph.label(binary)
    binary = morph.keep_marked(binary, mask)
    _, ncomps_after = morph.label(binary)
    LOG.debug('black components before/after borderclean (margin=%d): %d/%d',
              margin, ncomps_before, ncomps_after)
    return np.maximum((1 - binary)*(1 - mask), array)

# from ocropus-rpred
def check_line(array):
    """Validate array as a plausible text line image.
    
    Given a binarized, inverted image as Numpy array `array`
    (with 0 for white and 1 for black),
    check the array has the right dimensions, is in fact inverted,
    and does not have too few or too many connected black components.
    
    Returns an error report, or None if valid.
    """
    if len(array.shape)==3: return "input image is color image %s"%(array.shape,)
    if np.mean(array)<np.median(array): return "image may be inverted"
    h,w = array.shape
    if h<20: return "image not tall enough for a text line %s"%(array.shape,)
    if h>200: return "image too tall for a text line %s"%(array.shape,)
    #if w<1.5*h: return "line too short %s"%(array.shape,)
    if w<1.5*h and w<32: return "line too short %s"%(array.shape,)
    if w>4000: return "line too long %s"%(array.shape,)
    ratio = w*1.0/h
    _,ncomps = measurements.label(array>np.mean(array))
    lo = int(0.5*ratio+0.5)
    hi = int(4*ratio)+1
    if ncomps<lo: return "too few connected components (got %d, wanted >=%d)"%(ncomps,lo)
    #if ncomps>hi*ratio: return "too many connected components (got %d, wanted <=%d)"%(ncomps,hi)
    if ncomps>hi*ratio and ncomps>10: return "too many connected components (got %d, wanted <=%d)"%(ncomps,hi)
    return None

# inspired by ocropus-gpageseg check_page
def check_region(array):
    """Validate array as a plausible text region image.
    
    Given a binarized, inverted image as Numpy array `array`
    (with 0 for white and 1 for black),
    check the array has the right dimensions, is in fact inverted,
    and does not have too few or too many connected black components.
    
    Returns an error report, or None if valid.
    """
    if len(array.shape)==3: return "input image is color image %s"%(array.shape,)
    if np.mean(array)<np.median(array): return "image may be inverted"
    h,w = array.shape
    if h<100: return "image not tall enough for a region image %s"%(array.shape,)
    if h>5000: return "image too tall for a region image %s"%(array.shape,)
    if w<100: return "image too narrow for a region image %s"%(array.shape,)
    if w>5000: return "line too wide for a region image %s"%(array.shape,)
    slots = int(w*h*1.0/(30*30))
    _,ncomps = measurements.label(array>np.mean(array))
    if ncomps<5: return "too few connected components for a region image (got %d)"%(ncomps,)
    if ncomps>slots*2 and ncomps>10: return "too many connnected components for a region image (%d > %d)"%(ncomps,slots)
    return None

# from ocropus-gpageseg
def check_page(array):
    """Validate array as a plausible printed text page image.
    
    Given a binarized, inverted image as Numpy array `array`
    (with 0 for white and 1 for black),
    check the array has the right dimensions, is in fact inverted,
    and does not have too few or too many connected black components.
    
    Returns an error report, or None if valid.
    """
    if len(array.shape)==3: return "input image is color image %s"%(array.shape,)
    if np.mean(array)<np.median(array): return "image may be inverted"
    h,w = array.shape
    if h<600: return "image not tall enough for a page image %s"%(array.shape,)
    if h>10000: return "image too tall for a page image %s"%(array.shape,)
    if w<600: return "image too narrow for a page image %s"%(array.shape,)
    if w>10000: return "line too wide for a page image %s"%(array.shape,)
    slots = int(w*h*1.0/(30*30))
    _,ncomps = measurements.label(array>np.mean(array))
    if ncomps<10: return "too few connected components for a page image (got %d)"%(ncomps,)
    if ncomps>slots and ncomps>10: return "too many connnected components for a page image (%d > %d)"%(ncomps,slots)
    return None

def odd(num):
    return num + (num+1)%2

# from ocropus-gpageseg
def DSAVE(title,array):
    from matplotlib import pyplot as plt
    from matplotlib import patches as mpatches
    if type(array)==list:
        assert len(array)==3
        array = np.transpose(np.array(array),[1,2,0])
    #plt.imshow(array.astype('float'))
    #plt.legend(handles=[mpatches.Patch(label=title)])
    #plt.show()
    fname = title+".png"
    plt.imsave(fname,array.astype('float'))

# from ocropus-gpageseg    
def compute_separators_morph(binary,scale, sepwiden=10, maxseps=0):
    """Finds vertical black lines corresponding to column separators."""
    d0 = int(max(5,scale/4))
    d1 = int(max(5,scale))+ sepwiden
    thick = morph.r_dilation(binary,(d0,d1))
    vert = morph.rb_opening(thick,(10*scale,1))
    vert = morph.r_erosion(vert,(d0//2, sepwiden))
    vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2* maxseps)
    vert = morph.select_regions(vert,sl.dim0,min=20*scale,nbest=maxseps)
    return vert

# from ocropus-gpagseg
def compute_colseps_conv(binary,scale=1.0, csminheight=10, maxcolseps=2):
    """Find column separators by convolution and
    thresholding."""
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
        seps = compute_separators_morph(binary,scale, maxseps=maxseps)
        #DSAVE("colseps",0.7*seps+0.3*binary)
        #colseps = compute_colseps_morph(binary,scale)
        colseps = np.maximum(colseps,seps)
        binary = np.minimum(binary,1-seps)
    #binary,colseps = apply_mask(binary,colseps)
    return colseps,binary

# from ocropus-gpageseg, but with additional option vsticky
def compute_gradmaps(binary,scale,
                     vsticky=False,
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
             hscale*6*scale),
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
            (vscale,hscale*6*scale))
        if vsticky:
            # make gradient stick at top and bottom
            # so chopped off lines will be used as well
            grad[0, :] = 0.01
            grad[-1, :] = -0.01
    #DSAVE("grad", grad)
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    #DSAVE("bottom+top+boxmap", [boxmap, bottom + 2*top, binary])
    return bottom,top,boxmap

# from ocropus-gpageseg
def compute_line_seeds(binary,bottom,top,colseps,scale,
                       threshold=0.2,vscale=1.0):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = threshold
    vrange = int(vscale*scale)
    bmarked = filters.maximum_filter(
        bottom==filters.maximum_filter(bottom,(vrange,0)),
        (2,2)) * (bottom>t*np.amax(bottom)*t) *(1-colseps)
    tmarked = filters.maximum_filter(
        top==filters.maximum_filter(top,(vrange,0)),
        (2,2)) * (top>t*np.amax(top)*t/2) *(1-colseps)
    tmarked = filters.maximum_filter(tmarked,(1,10))
    seeds = np.zeros(binary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y,1) for y in psegutils.find(bmarked[:,x])] +
                             [(y,0) for y in psegutils.find(tmarked[:,x])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = filters.maximum_filter(seeds,(1,int(1+scale)))
    seeds = seeds*(1-colseps)
    #DSAVE("lineseeds unlabelled",[seeds,0.3*tmarked+0.7*bmarked,binary])
    seeds, _ = morph.label(seeds)
    return seeds

def hmerge_line_seeds(seeds, colseps):
    # use full horizontal spread to avoid splitting lines at long whitespace
    # (but only indirectly as masks with a label conflict, so we can
    #  horizontally merge components, and at the same time avoid to
    #  horizontally enlarge components in general, which would
    #  allow corners to become the largest component later-on;
    #  moreover, ignore conflicts which affect only small fractions
    #  of either line, so a merge can be avoided for small vertical overlap):
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
        count, candidate = overlap_count[label]
        if count / labels_count[label] > 0.2:
            relabel[label] = relabel[candidate]
    seeds = relabel[seeds]
    #DSAVE("lineseeds hmerged", seeds)
    return seeds

# like ocropus-gpageseg compute_segmentation, but with fullpage switch, with larger scale, smaller hscale and horizontal merge, and without final unmasking
def compute_line_labels(array, fullpage=False, maxcolseps=2, maxseps=0):
    """Find text line segmentation within a region or page.
    
    Given a binarized image as Numpy array `array`, compute a complete segmentation
    into text lines, avoiding horizontal splits. If `fullpage` is True, then also
    find up to `maxcolseps` white-space and up to `maxseps` black column separators.
    
    Return the background label array (not the foreground or the masked image).
    """
    if np.prod(array.shape) == 0:
        raise Exception('image dimensions are zero')
    if np.amax(array) == np.amin(array):
        raise Exception('image is blank')
    binary = np.array(array <= ocrolib.midrange(array), np.uint8)
    #DSAVE("binary",binary)
    if fullpage:
        report = check_page(binary)
    else:
        report = check_region(binary)
    if report:
        raise Exception(report)
    scale = psegutils.estimate_scale(binary)
    # use larger scale so broken/blackletter fonts with their large capitals
    # are not cut into two lines:
    scale *= 2
    LOG.debug('xheight: %d, scale: %d', binary.shape[0], scale)
    bottom, top, boxmap = compute_gradmaps(binary, scale, usegauss=False,
                                           # make top and bottom sticky so chopped-off
                                           # lines are not merged with neighbours:
                                           vsticky=not fullpage,
                                           # use small scale so broken/blackletter fonts
                                           # (with their large capitals)
                                           # can be approximated with enough precision:
                                           hscale=0.2, vscale=0.5)
    if fullpage:
        colseps, _ = compute_colseps(binary, scale, maxcolseps=maxcolseps, maxseps=maxseps)
    else:
        colseps = np.zeros(binary.shape, np.uint8)
    seeds = compute_line_seeds(binary, bottom, top, colseps, scale)
    #DSAVE("seeds",[bottom,top,boxmap])
    seeds = hmerge_line_seeds(seeds, colseps)

    # spread the text line seeds to all the remaining
    # components
    # (boxes with overlapping seeds will become background:)
    llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
    spread = morph.spread_labels(seeds,maxdist=scale)
    #DSAVE('spread', spread)
    # (background and conflict will get nearest seed:)
    llabels = np.where(llabels>0,llabels,spread)
    #DSAVE('llabels', llabels)
    #segmentation = llabels*binary
    #return segmentation
    return llabels

# from ocropus-gpageseg, but as separate step on the PIL.Image
def remove_noise(image, maxsize=8):
    array = pil2array(image)
    binary = np.array(array <= ocrolib.midrange(array), np.uint8)
    _, ncomps_before = morph.label(binary)
    binary = ocrolib.remove_noise(binary, maxsize)
    _, ncomps_after = morph.label(binary)
    LOG.debug('black components before/after denoising (maxsize=%d): %d/%d',
              maxsize, ncomps_before, ncomps_after)
    return array2pil(1 - binary)

# to be refactored into core:
# inspired by ocrolib.compute_lines
def resegment(mask_image, labels):
    """Shrink a mask to the largest component of a segmentation.
    
    Given a PIL.Image of a mask and a label array for the region,
    find the label with the largest area within the mask,
    and reduce the mask to that.
    
    Return a PIL.Image.
    """
    mask = np.array(pil2array(mask_image), np.uint8)
    #DSAVE('mask', mask)
    #DSAVE('labels', labels)
    #DSAVE('mask*labels', mask*labels)
    objects = morph.find_objects(mask * labels)
    max_count, max_mask = 0, mask
    for slices in objects:
        if not slices: continue
        if (sl.dim1(slices) < 2 * 20 or
            sl.dim0(slices) < 20): continue
        # identify largest label in slice:
        count = np.bincount(labels[slices].flat)
        label = np.argmax(count)
        count = np.amax(count)
        if not count: continue
        if count > max_count:
            max_count = count
            #max_mask = labels == label
            # avoid increasing margin when recropping:
            max_mask = mask * (labels == label)
    #DSAVE('max_mask', max_mask)
    LOG.debug('black pixels before/after resegment (nlabels=%d): %d/%d',
              len(objects),
              np.count_nonzero(mask * labels),
              np.count_nonzero(max_mask))
    return array2pil(max_mask)

# to be refactored into core (as function in ocrd_utils):
def polygon_mask(image, coordinates):
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=255)
    return mask

# to be refactored into core (as function in ocrd_utils):
def rotate_polygon(coordinates, angle, orig={'x': 0, 'y': 0}):
    # if the region image has been rotated, we must also
    # rotate the coordinates of the line
    # (which relate to the top page image)
    # in the same direction but with inverse transformation
    # matrix (i.e. passive rotation), and
    # (since the region was rotated around its center,
    #  but our coordinates are now relative to the top left)
    # by first translating to center of region, then
    # rotating around that center, and translating back:
    # point := (point - region_center) * region_rotation + region_center
    # moreover, since rotation has reshaped/expanded the image,
    # the line coordinates must be offset by those additional pixels:
    # point := point + 0.5 * (new_region_size - old_region_size)
    angle = np.deg2rad(angle)
    # active rotation:  [[cos, -sin], [sin, cos]]
    # passive rotation: [[cos, sin], [-sin, cos]] (inverse)
    return [(orig['x']
             + (x - orig['x'])*np.cos(angle)
             + (y - orig['y'])*np.sin(angle),
             orig['y']
             - (x - orig['x'])*np.sin(angle)
             + (y - orig['y'])*np.cos(angle))
            for x, y in coordinates]

# to be refactored into core (as method of ocrd.workspace.Workspace):
def image_from_page(workspace, page,
                    page_image,
                    page_id):
    """Extract the Page image from the workspace.
    
    Given a PIL.Image of the page, `page_image`,
    and the Page object logically associated with it, `page`,
    extract its PIL.Image from AlternativeImage (if it exists),
    or via cropping from `page_image` (if a Border exists),
    or by just returning `page_image` (otherwise).
    
    When using AlternativeImage, if the resulting page image
    is larger than the annotated page, then pass down the page's
    box coordinates with an offset of half the width/height difference.
    
    Return the extracted image, and the page's box coordinates,
    relative to the source image (for passing down).
    """
    page_xywh = {'x': 0,
                 'y': 0,
                 'w': page_image.width,
                 'h': page_image.height}
    # FIXME: remove PrintSpace here as soon as GT abides by the PAGE standard:
    border = page.get_Border() or page.get_PrintSpace()
    if border and border.get_Coords():
        LOG.debug("Using explictly set page border '%s' for page '%s'",
                  border.get_Coords().points, page_id)
        page_xywh = xywh_from_points(border.get_Coords().points)
    
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
        page_image = page_image.crop(
            box=(page_xywh['x'],
                 page_xywh['y'],
                 page_xywh['x'] + page_xywh['w'],
                 page_xywh['y'] + page_xywh['h']))
        # FIXME: mask away all GraphicRegion, SeparatorRegion etc which
        # could overlay any text regions
    # subtract offset from any increase in binary region size over source:
    page_xywh['x'] -= 0.5 * max(0, page_image.width  - page_xywh['w'])
    page_xywh['y'] -= 0.5 * max(0, page_image.height - page_xywh['h'])
    return page_image, page_xywh

# to be refactored into core (as method of ocrd.workspace.Workspace):
def image_from_region(workspace, region,
                      page_image, page_xywh):
    """Extract the TextRegion image from a Page image.
    
    Given a PIL.Image of the page, `page_image`,
    and its coordinates relative to the border, `page_xywh`,
    and a TextRegion object logically contained in it, `region`,
    extract its PIL.Image from AlternativeImage (if it exists),
    or via cropping from `page_image`.
    
    When cropping, respect any angle annotated for the region
    (from deskewing) by rotating the cropped image, respectively.
    Regardless, if the resulting region image is larger than
    the annotated region, pass down the region's box coordinates
    with an offset of half the width/height difference.
    
    Return the extracted image, and the region's box coordinates,
    relative to the page image (for passing down).
    """
    region_xywh = xywh_from_points(region.get_Coords().points)
    # region angle: PAGE orientation is defined clockwise,
    # whereas PIL/ndimage rotation is in mathematical direction:
    region_xywh['angle'] = -(region.get_orientation() or 0)
    alternative_image = region.get_AlternativeImage()
    if alternative_image:
        # (e.g. from region-level cropping, binarization, deskewing or despeckling)
        LOG.debug("Using AlternativeImage %d (%s) for region '%s'",
                  len(alternative_image), alternative_image[-1].get_comments(),
                  region.id)
        region_image = workspace.resolve_image_as_pil(
            alternative_image[-1].get_filename())
    else:
        region_image = page_image.crop(
            box=(region_xywh['x'] - page_xywh['x'],
                 region_xywh['y'] - page_xywh['y'],
                 region_xywh['x'] - page_xywh['x'] + region_xywh['w'],
                 region_xywh['y'] - page_xywh['y'] + region_xywh['h']))
        # FIXME: mask any overlapping regions (esp. Separator/Noise/Image)
        # but we might need overlapping rules: e.g. an ImageRegion which
        # properly contains our TextRegion should be completely ignored, but
        # an ImageRegion which is properly contained in our TextRegion should
        # be completely masked, while partial overlap may be more difficult
        # to decide (use polygons?)
        if region_xywh['angle']:
            LOG.info("About to rotate region '%s' by %.2f°",
                      region.id, region_xywh['angle'])
            region_image = region_image.rotate(region_xywh['angle'],
                                               expand=True,
                                               #resample=Image.BILINEAR,
                                               fillcolor='white')
    # subtract offset from any increase in binary region size over source:
    region_xywh['x'] -= 0.5 * max(0, region_image.width  - region_xywh['w'])
    region_xywh['y'] -= 0.5 * max(0, region_image.height - region_xywh['h'])
    return region_image, region_xywh

# to be refactored into core (as method of ocrd.workspace.Workspace):
def image_from_line(workspace, line,
                    region_image, region_xywh,
                    segmentation=None):
    """Extract the TextLine image from a TextRegion image.
    
    Given a PIL.Image of the region, `region_image`,
    and its coordinates relative to the page, `region_xywh`,
    and a TextLine object logically contained in it, `line`,
    extract its PIL.Image from AlternativeImage (if it exists),
    or via cropping from `region_image`.
    
    When cropping, respect any angle annotated for the region
    (from deskewing) by compensating the line coordinates in
    an inverse transformation (translation to center, rotation,
    re-translation). Also, mind the difference between annotated
    and actual size of the region (usually from deskewing), by
    a respective offset into the image. Cropping uses a polygon
    mask (not just the rectangle).
    
    If passed an optional labelling for the region, `segmentation`,
    the mask is shrinked further to the largest overlapping line
    label, which avoids seeing ascenders from lines below, and
    descenders from lines above `line`.
    
    If the resulting line image is larger than the annotated line,
    pass down the line's box coordinates with an offset of half
    the width/height difference.
    
    Return the extracted image, and the line's box coordinates,
    relative to the region image (for passing down).
    """
    line_points = line.get_Coords().points
    line_xywh = xywh_from_points(line_points)
    line_polygon = [(x - region_xywh['x'],
                     y - region_xywh['y'])
                    for x, y in polygon_from_points(line_points)]
    alternative_image = line.get_AlternativeImage()
    if alternative_image:
        # (e.g. from line-level cropping, deskewing or despeckling)
        LOG.debug("Using AlternativeImage %d (%s) for line '%s'",
                  len(alternative_image), alternative_image[-1].get_comments(),
                  line.id)
        line_image = workspace.resolve_image_as_pil(
            alternative_image[-1].get_filename())
    else:
        # create a mask from the line polygon:
        line_polygon = rotate_polygon(line_polygon,
                                      region_xywh['angle'],
                                      orig={'x': 0.5 * region_image.width,
                                            'y': 0.5 * region_image.height})
        line_mask = polygon_mask(region_image, line_polygon)
        if isinstance(segmentation, np.ndarray):
            # modify mask from (ad-hoc) line segmentation of region
            # (shrink to largest label spread in that area):
            line_mask = resegment(line_mask, segmentation)
        # create a background image from its median color
        # (in case it has not been binarized yet):
        region_array = np.asarray(region_image)
        background = np.median(region_array, axis=[0, 1], keepdims=True)
        region_array = np.broadcast_to(background.astype(np.uint8), region_array.shape)
        line_image = Image.fromarray(region_array)
        line_image.paste(region_image, mask=line_mask)
        # recrop into a line:
        bbox = line_mask.getbbox()
        if bbox:
            left, upper, right, lower = bbox
            # keep upper/lower, regardless of h (no vertical padding)
            # pad left/right if target width w is larger:
            margin_x = (line_xywh['w'] - right + left) // 2
            left = max(0, left - margin_x)
            right = min(line_mask.width, left + line_xywh['w'])
        else:
            left = line_xywh['x'] - region_xywh['x']
            upper = line_xywh['y'] - region_xywh['y']
            right = left + line_xywh['w']
            lower = upper + line_xywh['h']
        line_image = line_image.crop(box=(left, upper, right, lower))
    # subtract offset from any increase in binary line size over source:
    line_xywh['x'] -= 0.5 * max(0, line_image.width  - line_xywh['w'])
    line_xywh['y'] -= 0.5 * max(0, line_image.height - line_xywh['h'])
    return line_image, line_xywh
                
# to be refactored into core (as method of ocrd.workspace.Workspace):
def image_from_word(workspace, word,
                    line_image, line_xywh):
    """Extract the Word image from a TextLine image.
    
    Given a PIL.Image of the line, `line_image`,
    and its coordinates relative to the region, `line_xywh`,
    and a Word object logically contained in it, `word`,
    extract its PIL.Image from AlternativeImage (if it exists),
    or via cropping from `line_image`.
    
    When cropping, mind the difference between annotated
    and actual size of the line (usually from deskewing), by
    a respective offset into the image. Cropping uses a polygon
    mask (not just the rectangle).
    
    If the resulting word image is larger than the annotated word,
    pass down the word's box coordinates with an offset of half
    the width/height difference.
    
    Return the extracted image, and the word's box coordinates,
    relative to the line image (for passing down).
    """
    word_points = word.get_Coords().points
    word_xywh = xywh_from_points(word_points)
    word_polygon = [(x - line_xywh['x'],
                     y - line_xywh['y'])
                    for x, y in polygon_from_points(word_points)]
    alternative_image = word.get_AlternativeImage()
    if alternative_image:
        # (e.g. from word-level cropping or binarization)
        LOG.debug("Using AlternativeImage %d (%s) for word '%s'",
                  len(alternative_image), alternative_image[-1].get_comments(),
                  word.id)
        word_image = workspace.resolve_image_as_pil(
            alternative_image[-1].get_filename())
    else:
        # create a mask from the word polygon:
        word_mask = polygon_mask(line_image, word_polygon)
        # create a background image from its median color
        # (in case it has not been binarized yet):
        line_array = np.asarray(line_image)
        background = np.median(line_array, axis=[0, 1], keepdims=True)
        line_array = np.broadcast_to(background.astype(np.uint8), line_array.shape)
        word_image = Image.fromarray(line_array)
        word_image.paste(line_image, mask=word_mask)
        # recrop into a line:
        bbox = word_mask.getbbox()
        if bbox:
            left, upper, right, lower = bbox
            # keep upper/lower, regardless of h (no vertical padding)
            # pad left/right if target width w is larger:
            margin_x = (word_xywh['w'] - right + left) // 2
            left = max(0, left - margin_x)
            right = min(word_mask.width, left + word_xywh['w'])
        else:
            left = word_xywh['x'] - line_xywh['x']
            upper = word_xywh['y'] - line_xywh['y']
            right = left + word_xywh['w']
            lower = upper + word_xywh['h']
        word_image = word_image.crop(box=(left, upper, right, lower))
    # subtract offset from any increase in binary line size over source:
    word_xywh['x'] -= 0.5 * max(0, word_image.width  - word_xywh['w'])
    word_xywh['y'] -= 0.5 * max(0, word_image.height - word_xywh['h'])
    return word_image, word_xywh

# to be refactored into core (as method of ocrd.workspace.Workspace):
def image_from_glyph(workspace, glyph,
                    word_image, word_xywh):
    """Extract the Glyph image from a Word image.
    
    Given a PIL.Image of the word, `word_image`,
    and its coordinates relative to the line, `word_xywh`,
    and a Glyph object logically contained in it, `glyph`,
    extract its PIL.Image from AlternativeImage (if it exists),
    or via cropping from `word_image`.
    
    When cropping, mind the difference between annotated
    and actual size of the word (usually from deskewing), by
    a respective offset into the image. Cropping uses a polygon
    mask (not just the rectangle).
    
    If the resulting glyph image is larger than the annotated glyph,
    pass down the glyph's box coordinates with an offset of half
    the width/height difference.
    
    Return the extracted image, and the glyph's box coordinates,
    relative to the word image (for passing down).
    """
    glyph_points = glyph.get_Coords().points
    glyph_xywh = xywh_from_points(glyph_points)
    glyph_polygon = [(x - word_xywh['x'],
                      y - word_xywh['y'])
                     for x, y in polygon_from_points(glyph_points)]
    alternative_image = glyph.get_AlternativeImage()
    if alternative_image:
        # (e.g. from glyph-level cropping or binarization)
        LOG.debug("Using AlternativeImage %d (%s) for glyph '%s'",
                  len(alternative_image), alternative_image[-1].get_comments(),
                  glyph.id)
        glyph_image = workspace.resolve_image_as_pil(
            alternative_image[-1].get_filename())
    else:
        # create a mask from the glyph polygon:
        glyph_mask = polygon_mask(word_image, glyph_polygon)
        # create a background image from its median color
        # (in case it has not been binarized yet):
        word_array = np.asarray(word_image)
        background = np.median(word_array, axis=[0, 1], keepdims=True)
        word_array = np.broadcast_to(background.astype(np.uint8), word_array.shape)
        glyph_image = Image.fromarray(word_array)
        glyph_image.paste(word_image, mask=glyph_mask)
        # recrop into a word:
        bbox = glyph_mask.getbbox()
        if bbox:
            left, upper, right, lower = bbox
            # keep upper/lower, regardless of h (no vertical padding)
            # pad left/right if target width w is larger:
            margin_x = (glyph_xywh['w'] - right + left) // 2
            left = max(0, left - margin_x)
            right = min(glyph_mask.width, left + glyph_xywh['w'])
        else:
            left = glyph_xywh['x'] - word_xywh['x']
            upper = glyph_xywh['y'] - word_xywh['y']
            right = left + glyph_xywh['w']
            lower = upper + glyph_xywh['h']
        glyph_image = glyph_image.crop(box=(left, upper, right, lower))
    # subtract offset from any increase in binary word size over source:
    glyph_xywh['x'] -= 0.5 * max(0, glyph_image.width  - glyph_xywh['w'])
    glyph_xywh['y'] -= 0.5 * max(0, glyph_image.height - glyph_xywh['h'])
    return glyph_image, glyph_xywh

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
    """Constructs a numeric list representing a bounding box from polygon coordinates in page representation."""
    xys = [[int(p) for p in pair.split(',')] for pair in points.split(' ')]
    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0
    for xy in xys:
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
def points_from_bbox(minx, miny, maxx, maxy):
    """Constructs polygon coordinates in page representation from a numeric list representing a bounding box."""
    return "%i,%i %i,%i %i,%i %i,%i" % (
        minx, miny, maxx, miny, maxx, maxy, minx, maxy)

# to be refactored into core (as function in ocrd_utils):
def xywh_from_bbox(minx, miny, maxx, maxy):
    """Converts a bounding box from a numeric list to a numeric dict representation."""
    return {
        'x': minx,
        'y': miny,
        'w': maxx - minx,
        'h': maxy - miny,
    }

# to be refactored into core (as function in ocrd_utils):
def bbox_from_xywh(xywh):
    """Converts a bounding box from a numeric dict to a numeric list representation."""
    return (
        xywh['x'],
        xywh['y'],
        xywh['x'] + xywh['w'],
        xywh['y'] + xywh['h']
    )

# to be refactored into core (as function in ocrd_utils):
def points_from_polygon(polygon):
    """Converts polygon coordinates from a numeric list representation to a page representation."""
    return " ".join("%i,%i" % (x, y) for x, y in polygon)

def membername(class_, val):
    return next((k for k, v in class_.__dict__.items() if v == val), str(val))
