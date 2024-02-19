from __future__ import absolute_import

import warnings
import logging

import numpy as np
from scipy.ndimage import measurements, filters, interpolation, morphology
from scipy import stats, signal
#from skimage.morphology import convex_hull_image
from skimage.morphology import medial_axis
import networkx as nx
from PIL import Image

from . import ocrolib
from .ocrolib import morph, psegutils, sl
# for decorators (type-checks etc):
from .ocrolib.toplevel import *

LOG = logging.getLogger('ocrolib') # to be refined by importer

# method similar to ocrolib.read_image_gray
@checks(Image.Image)
def pil2array(image, alpha=0):
    """Convert image to floating point grayscale array.

    Given a PIL.Image instance (of any colorspace),
    convert to a grayscale Numpy array
    (with 1.0 for white and 0.0 for black).

    If ``alpha`` is not zero, and an alpha channel is present,
    then preserve it (the array will now have 3 dimensions,
    with 2 coordinates in the last: luminance and transparency).
    """
    assert isinstance(image, Image.Image), "not a PIL.Image"
    array = ocrolib.pil2array(image, alpha=alpha)
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
        if array.shape[-1] == 3:
            array = np.mean(array, 2)
        elif array.shape[-1] == 4:
            alpha = array[:,:,3]
            color = np.mean(array[:,:,:3])
            array = np.dstack((color,alpha))
    return array

@checks(ANY(GRAYSCALE1,ABINARY2))
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
@checks(GRAYSCALE1)
def estimate_local_whitelevel(image, zoom=0.5, perc=80, range_=20):
    '''flatten/normalize image by estimating the local whitelevel
    ``zoom``: downscaling for page background estimation (smaller=faster)
    ``perc``: percentage for filters
    ``range_``: size of filters
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # calculate at reduced pixel density to save CPU time
        m = interpolation.zoom(image, zoom, mode='nearest')
        m = filters.percentile_filter(m, perc, size=(range_, 2))
        m = filters.percentile_filter(m, perc, size=(2, range_))
        m = interpolation.zoom(m, 1. / zoom)
    ##w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    ##flat = np.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)
    # we want to get exactly the same size as before:
    h, w = image.shape
    h0, w0 = m.shape
    m = np.pad(m, ((max(0, h-h0),0), (max(0, w-w0),0)), 'edge')[:h, :w]
    flat = np.nan_to_num(np.clip(image - m + 1, 0, 1))
    return flat

# from ocropy-nlbin, but with threshold on variance
@checks(GRAYSCALE1,ALL(ARRAY1,ARANGE(-90,90)))
def estimate_skew_angle(image, angles, min_factor=1.5):
    """Score deskewing angles for image by projection profiling.
    
    Given grayscale-normalized and inverted (fg=1.0) array ``image``,
    and array ``angles`` of angles (degrees counter-cw),
    rotate image by each candidate angle and calculate a score for it
    in the following way.
    
    Aggregate average luminance along pixel lines (horizontally),
    then determine their variance across pixel lines (vertically).
    (The variance will be maximal when largely-black lines within
     textlines and mostly-white lines between textlines are clearly
     separated, i.e. orthogonal.)
    
    Return the angle which achieves a maximum score, if its score
    is larger than the average scores by ``min_factor``, or zero
    otherwise.
    """
    # TODO: make zoomable, i.e. interpolate down to max 300 DPI to be faster
    # TODO: sweep through angles very coarse, then hill climbing for precision
    # TODO: try with shear (i.e. simply numpy shift) instead of true rotation
    # TODO: use square of difference instead of variance as projection score
    #       (more reliable with background noise or multi-column)
    # TODO: offer flip (90°) test (comparing length-normalized projection profiles)
    # TODO: offer mirror (180°, or + vs - 90°) test based on ascender/descender signal
    #       (Latin scripts have more ascenders e.g. bhdkltſ than descenders e.g. qgpyj)
    estimates = np.zeros_like(angles)
    varmax = 0
    for i, a in enumerate(angles):
        #rotated = interpolation.rotate(image, a, order=0, mode='constant')
        # *much* faster (by an order of magnitude):
        rotated = np.array(Image.fromarray(image).rotate(
            # (BILINEAR would be equivalent to above, but only 4x as fast)
            a, expand=True, resample=Image.NEAREST))
        v = np.mean(rotated, axis=1)
        v = np.var(v)
        estimates[i] = v
        if v > varmax:
            varmax = v
    # only return the angle of the largest entropy
    # if it is considerably larger than the average
    varavg = np.mean(estimates)
    LOG.debug('estimate skew angle: min(angle)=%.1f° max(angle)=%.1f° max(var)=%.2g avg(var)=%.2g',
              np.min(angles), np.max(angles), varmax, varavg)
    if varmax and varmax / varavg > min_factor:
        return angles[np.argmax(estimates)]
    else:
        return 0

# from ocropy-nlbin, but with reshape=True
@checks(GRAYSCALE1)
def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    '''estimate skew angle and rotate'''
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1) # border ignore
    # invert
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
    # we must allow reshape/expand to avoid losing information in the corners
    # (but this also means that consumers of the AlternativeImage must
    #  offset coordinates by half the increased width/height besides
    #  correcting for rotation in the coordinates):
    #flat = interpolation.rotate(flat, angle, mode='constant', reshape=True)
    # *much* faster:
    flat = np.array(Image.fromarray(flat).rotate(
            angle, expand=True, resample=Image.BICUBIC)) / 255.0
    # invert back
    flat = np.amax(flat) - flat
    return flat, angle

# from ocropy-nlbin
@checks(GRAYSCALE1)
def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90):
    '''estimate low and high thresholds.
    ``bignore``: ignore this much of the border/margin area for threshold estimation
    ``escale``: scale for estimating a mask over the text region
    ``lo``: percentile for black estimation
    ``hi``: percentile for white estimation
    Return a float tuple of low/black and high/white
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

# from ocropy-nlbin process1, but
# - reshape when rotating,
# - catch NaN,
# - add nrm switch
@checks(GRAYSCALE1)
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
    """Binarize/grayscale-normalize image via locally adaptive thresholding.
    
    Return a tuple:
    - if ``nrm``, then the grayscale-normalized float array (bg=1.0),
      otherwise the binarized integer array (bg=1 / fg=0)
    - if ``maxskew>0``, then the deskewing angle (in degrees counter-cw) applied,
      otherwise zero
    """
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
@checks(GRAYSCALE1)
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

@checks(ALL(ABINARY2,DARK))
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
    if np.prod(binary.shape)==0: return "image dimensions are zero"
    if len(binary.shape)==3: return "image is not monochrome %s"%(binary.shape,)
    if np.amax(binary)==np.amin(binary): return "image is blank"
    if np.mean(binary)<np.median(binary): return "image may be inverted"
    h,w = binary.shape
    if h<20/zoom: return "image not tall enough for a text line %s"%(binary.shape,)
    if h>200/zoom: return "image too tall for a text line %s"%(binary.shape,)
    ##if w<1.5*h: return "line too short %s"%(binary.shape,)
    if w<1.5*h and w<32/zoom: return "image too short for a line image %s"%(binary.shape,)
    if w>4000/zoom: return "image too long for a line image %s"%(binary.shape,)
    return None
    ratio = w*1.0/h
    _, ncomps = measurements.label(binary)
    lo = int(0.5*ratio+0.5)
    hi = int(4*ratio)+1
    if ncomps<lo: return "too few connected components (got %d, wanted >=%d)"%(ncomps,lo)
    ##if ncomps>hi*ratio: return "too many connected components (got %d, wanted <=%d)"%(ncomps,hi)
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
    if np.prod(binary.shape)==0: return "image dimensions are zero"
    if len(binary.shape)==3: return "image is not monochrome %s"%(binary.shape,)
    if np.amax(binary)==np.amin(binary): return "image is blank"
    if np.mean(binary)<np.median(binary): return "image may be inverted"
    h,w = binary.shape
    if h<45/zoom: return "image not tall enough for a region image %s"%(binary.shape,)
    if h>5000/zoom: return "image too tall for a region image %s"%(binary.shape,)
    if w<100/zoom: return "image too narrow for a region image %s"%(binary.shape,)
    if w>5000/zoom: return "image too wide for a region image %s"%(binary.shape,)
    return None
    # zoom factor (DPI relative) and 4 (against fragmentation from binarization)
    slots = int(w*h*1.0/(30*30)*zoom*zoom) * 4
    _,ncomps = measurements.label(binary)
    if ncomps<5: return "too few connected components for a region image (got %d)"%(ncomps,)
    if ncomps>slots and ncomps>10: return "too many connected components for a region image (%d > %d)"%(ncomps,slots)
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
    if np.prod(binary.shape)==0: return "image dimensions are zero"
    if len(binary.shape)==3: return "image not monochrome %s"%(binary.shape,)
    if np.amax(binary)==np.amin(binary): return "image is blank"
    if np.mean(binary)<np.median(binary): return "image may be inverted"
    h,w = binary.shape
    if h<600/zoom: return "image not tall enough for a page image %s"%(binary.shape,)
    if h>20000/zoom: return "image too tall for a page image %s"%(binary.shape,)
    if w<600/zoom: return "image too narrow for a page image %s"%(binary.shape,)
    if w>20000/zoom: return "image too wide for a page image %s"%(binary.shape,)
    return None
    # zoom factor (DPI relative) and 4 (against fragmentation from binarization)
    slots = int(w*h*1.0/(30*30)*zoom*zoom) * 4
    _,ncomps = measurements.label(binary)
    if ncomps<10: return "too few connected components for a page image (got %d)"%(ncomps,)
    if ncomps>slots and ncomps>10: return "too many connected components for a page image (%d > %d)"%(ncomps,slots)
    return None

def odd(num):
    return int(num) + int((num+1)%2)

# from ocropus-gpageseg, but with interactive switch
@disabled()
def DSAVE(title,array, interactive=False):
    """Plot all intermediate results for debugging.
    
    Comment the ``disabled`` decorator to activate.
    Change the default value of ``interactive`` to your needs:
    - True: step through all results interactively
      (quit plot window by pressing ``q`` to advance)
    - False: save all plots as PNG files under /tmp (or $TMPDIR)
      (call image viewer with file list based on date stamps)
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING) # workaround
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from matplotlib import patches as mpatches
    from tempfile import mkstemp
    from os import close
    import copy
    # set uniformly bright / maximally differentiating colors
    cmap = copy.copy(cm.rainbow) # default viridis is too dark on low end
    # use black for bg (not in the cmap)
    cmap.set_bad(color='black') # for background (normal)
    # allow calling with extra fg as 2nd array
    cmap.set_under(color='black') # for background
    cmap.set_over(color='white') # for foreground
    vmin, vmax = None, None
    # allow calling with 3 arrays (direct RGB channels)
    if type(array)==list:
        assert len(array) in [2,3]
        if len(array)==3:
            # 3 inputs, one for each RGB channel
            array = np.transpose(np.array(array),[1,2,0])
        else:
            array2 = array[1] # fg
            array = array[0] # labels
            vmin = 0 # under
            vmax = np.amax(array) # over
            array = array.copy()
            array[array2>0] = vmax+1 # fg
    else:
        vmin = 0
        vmax = np.amax(array)
    array = array.astype('float')
    array[array==0] = np.nan # bad (extra color)
    if interactive:
        # also return the last key pressed by user:
        result = None
        def on_press(event):
            nonlocal result
            if event.key not in ['q', 'ctrl+w']:
                result = event.key
        plt.connect('key_press_event', on_press)
        plt.imshow(array,vmin=vmin,vmax=vmax,interpolation='none',cmap=cmap)
        plt.legend(handles=[mpatches.Patch(label=title)])
        plt.tight_layout(pad=0)
        plt.show()
        plt.disconnect('key_press_event')
        return result
    else:
        fd, fname = mkstemp(suffix="_" + title + ".png")
        plt.imsave(fname,array,vmin=vmin,vmax=vmax,cmap=cmap)
        close(fd)
        LOG.debug('DSAVE %s', fname)

@checks(ABINARY2,NUMBER)
def compute_images(binary, scale, maximages=5):
    """Detects large connected foreground components that could be images.
    
    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``scale``, square root of average bbox area of characters
    - ``maximages``, maximum number of images to find
    (This could be drop-capitals, line drawings or photos.)
    
    Returns a same-size image label array.
    """
    if maximages == 0:
        return np.zeros_like(binary, int)
    images = binary
    # d0 = odd(max(2,scale/5))
    # d1 = odd(max(2,scale/8))
    # 1- close a little to reconnect components that have been
    #   noisily binarized
    #images = morph.rb_closing(images, (d0,d1))
    #DSAVE('images1_closed', images+0.6*binary)
    # 1- filter largest connected components
    images = morph.select_regions(images,sl.area,min=(4*scale)**2,nbest=2*maximages)
    DSAVE('images1_large', images+0.6*binary)
    if not images.any():
        return np.zeros_like(binary, int)
    # 2- open horizontally and vertically to suppress
    #    v/h-lines; these will be detected separately,
    #    and it is dangerous to combine them into one
    #    single frame, because then the hull polygon
    #    can cover/overlap large text/table parts which
    #    we cannot discern from the actual image anymore
    h_opened = morph.rb_opening(images, (1, odd(scale/2)))
    DSAVE('images2_h-opened', h_opened+0.6*binary)
    v_opened = morph.rb_opening(images, (odd(scale/2), 1))
    DSAVE('images2_v-opened', v_opened+0.6*binary)
    # 3- close whatever remains
    closed = morph.rb_closing(h_opened&v_opened, (odd(2*scale),odd(2*scale)))
    DSAVE('images3_closed', closed+0.6*binary)
    # 4- reconstruct the losses up to a certain distance
    #    to avoid creeping into pure h/v-lines again but still
    #    cover most of the large object
    #images = np.where(images, closed, 2)
    #images = morph.spread_labels(images, maxdist=scale) % 2 | closed
    images = morph.rb_reconstruction(closed, images, step=2, maxsteps=scale)
    DSAVE('images4_reconstructed', images+0.6*binary)
    # 5- select nbest
    images = morph.select_regions(images,sl.area,min=(4*scale)**2,nbest=maximages)
    DSAVE('images5_selected', images+0.6*binary)
    if not images.any():
        return np.zeros_like(binary, int)
    # 6- dilate a little to get a smooth contour without gaps
    dilated = morph.r_dilation(images, (odd(scale),odd(scale)))
    images = morph.propagate_labels_majority(binary, dilated+1)
    images = morph.spread_labels(images, maxdist=scale)==2
    images, _ = morph.label(images)
    DSAVE('images6_dilated', images+0.6*binary)
    # we could repeat reconstruct-dilate here...
    return images

@checks(ABINARY2,NUMBER)
def compute_seplines(binary, scale, maxseps=0):
    """Detects thin connected foreground components that could be separators.
    
    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``scale``, square root of average bbox area of characters
    - ``maxseps``, maximum number of separators to find
    (This could be horizontal, vertical or oblique, even slightly warped and discontinuous lines.)
    
    Returns a same-size separator label array.
    """
    # tries to find a compromise for the following issues,
    # potentially occurring in combination (or all at once):
    # - non-congiguous or broken lines (due to thin ink or low contrast)
    # - skewed, curved or warped lines (due to non-planar photography or irregular typography)
    # - very close or overlapping text (due to show-through or bad binarization)
    # - superimposed fg noise (due to bad binarization) that may connect text and non-text
    # - intersecting vertical and horizontal lines, even closed shapes (enclosing text)
    # - line-like glyphs (i.e. false positives)
    if maxseps == 0:
        return np.zeros_like(binary, int)
    skel, dist = medial_axis(binary, return_distance=True)
    DSAVE("medial-axis", [dist, skel])
    labels, nlabels = morph.label(skel)
    slices = [None] + morph.find_objects(labels)
    DSAVE("skel-labels", labels)
    # determine those components which could be separators
    # (filter by compactness, and by mean+variance of distances)
    sepmap = np.zeros(nlabels + 1, int)
    numsep = 0
    sepsizes = [0]
    sepslices = [None]
    sepdists = [0]
    for label in range(1, nlabels + 1):
        labelslice = slices[label]
        labelmask = labels == label
        labelsize = np.count_nonzero(labelmask) # sum of skel pixels, i.e. "inner length"
        labelarea = sl.area(labelslice)
        labelaspect = sl.aspect(labelslice)
        if labelaspect > 1:
            labelaspect = 1 / labelaspect
        labellength = np.hypot(*sl.dims(labelslice)) # length of bbox diagonal, i.e. "outer length"
        #LOG.debug("skel label %d has inner size %d and outer size %d", label, labelsize, labellength)
        if labelsize > 1.5 * labellength and labelaspect >= 0.1 and labelsize < 15 * scale: #and labelsize > 0.1 * labelarea
            # not long / straight, but very compact
            continue
        distances = dist[labelmask]
        avg_dist = np.median(distances) #np.mean(distances)
        std_dist = np.std(distances)
        # todo: empirical analysis of ideal thresholds
        if avg_dist > scale / 4 or std_dist/avg_dist > 0.7:
            continue
        #LOG.debug("skel label %d has dist %.1f±%.2f", label, avg_dist, std_dist)
        numsep += 1
        sepmap[label] = numsep
        sepsizes.append(labelsize)
        sepslices.append(labelslice)
        sepdists.append(avg_dist)
        if labelsize > 10 * scale and avg_dist > 0 and std_dist / avg_dist > 0.2:
            # try to split this large label up along neighbouring spans of similar distances:
            # (e.g. vlines that touch letters or images)
            # 1. get optimal (by variability) spans as bin intervals, then merge largest spans
            disthist, distedges = np.histogram(distances, bins='scott', density=True) # stone
            disthist *= np.diff(distedges) # get probability masses
            disthistlarge = disthist > 0.1
            if np.count_nonzero(disthistlarge) < 2:
                continue # only 1 large bin
            disthistlarge[-1] = True # ensure full interval
            distedges = distedges[1:][disthistlarge]
            disthist = np.cumsum(disthist)[disthistlarge]
            disthist = np.diff(disthist, prepend=0)
            distbin = np.digitize(distances, distedges, right=True)
            # 2. now find connected components within bins, but map all tiny components
            #    to a single label so they can be replaced by their neighbours later-on
            sublabels = np.zeros_like(labels)
            sublabels[labelmask] = distbin + 1
            DSAVE("sublabels", sublabels)
            sublabels2 = np.zeros_like(labels)
            sublabel = 1
            sublabelmap = [0, 1]
            for bin in range(len(distedges)):
                binmask = sublabels == bin + 1
                binlabels, nbinlabels = morph.label(binmask)
                _, binlabelcounts = np.unique(binlabels, return_counts=True)
                largemask = (binlabelcounts > 2 * scale)[binlabels]
                smallmask = (binlabelcounts <= 2 * scale)[binlabels]
                sublabels2[binmask & smallmask] = 1
                if not np.any(binmask & largemask):
                    continue
                sublabels2[binmask & largemask] = binlabels[binmask & largemask] + sublabel
                sublabel += nbinlabels
                sublabelmap.extend(nbinlabels*[bin + 1])
            if sublabel == 1:
                continue # only tiny sublabels here
            DSAVE("sublabels_connected", sublabels2)
            sublabelmap = np.array(sublabelmap)
            # 3. finally, replace tiny components by nearest components,
            #    and recombine survivors to bin labels
            smallmask = sublabels2 == 1
            sublabels2[smallmask] = 0
            sublabels2[smallmask] = morph.spread_labels(sublabels2)[smallmask]
            sublabels = sublabelmap[sublabels2]
            DSAVE("sublabels_final", sublabels)
            # now apply as multiple separators
            numsep -= 1
            sepmap[label] = 0
            slices[label] = None
            sepsizes = sepsizes[:-1]
            sepslices = sepslices[:-1]
            sepdists = sepdists[:-1]
            for sublabel in np.unique(sublabels[labelmask]):
                sublabelmask = sublabels == sublabel
                sublabelsize = np.count_nonzero(sublabelmask)
                sublabelslice = sublabelmask.nonzero()
                sublabelslice = sl.box(sublabelslice[0].min(),
                                       sublabelslice[0].max(),
                                       sublabelslice[1].min(),
                                       sublabelslice[1].max())
                subdistances = dist[sublabelmask]
                nlabels += 1
                numsep += 1
                sepmap = np.append(sepmap, numsep)
                labels[sublabelmask] = nlabels
                slices.append(sublabelslice)
                sepsizes.append(sublabelsize)
                sepslices.append(sublabelslice)
                sepdists.append(np.median(subdistances))
                #LOG.debug("adding sublabel %d as sep %d (size %d [%s])", sublabel, numsep, sublabelsize, str(sublabelslice))
    sepsizes = np.array(sepsizes)
    sepslices = np.array(sepslices)
    LOG.debug("detected %d separator candidates", numsep)
    DSAVE("seps-raw", sepmap[labels])
    # now dilate+erode to link neighbouring candidates,
    # but allow only such links which
    # - stay consistent regarding avg/std width
    # - do not enclose large areas in between
    # - do not "change direction" (roughly adds up their diagonals)
    # then combine mutual neighbourships to largest allowed partitions
    d0 = odd(max(1,scale/2))
    d1 = odd(max(1,scale/4))
    closed = morph.rb_closing(sepmap[labels] > 0, (d0,d1))
    DSAVE("seps-closed", [dist, closed])
    labels2, nlabels2 = morph.label(closed)
    corrs = morph.correspondences(sepmap[labels], labels2, return_counts=False).T
    corrmap = np.arange(numsep + 1)
    for sep2 in range(1, nlabels2 + 1):
        corrinds = corrs[:, 1] == sep2
        corrinds[corrs[:, 0] == 0] = False # ignore bg
        corrinds = corrinds.nonzero()[0]
        if len(corrinds) == 1:
            continue # nothing to link
        nonoverlapping = np.zeros((len(corrinds), len(corrinds)), dtype=bool)
        for i, indi in enumerate(corrinds[:-1]):
            sepi = corrs[indi, 0]
            labeli = np.flatnonzero(sepmap == sepi)[0]
            slicei = slices[labeli]
            lengthi = np.hypot(*sl.dims(slicei))
            areai = sl.area(slicei)
            for j, indj in enumerate(corrinds[i + 1:], i + 1):
                sepj = corrs[indj, 0]
                labelj = np.flatnonzero(sepmap == sepj)[0]
                slicej = slices[labelj]
                lengthj = np.hypot(*sl.dims(slicej))
                areaj = sl.area(slicej)
                union = sl.union(slicei, slicej)
                length = np.hypot(*sl.dims(union))
                if length < 0.9 * (lengthi + lengthj):
                    continue
                if sl.area(union) > 1.3 * (areai + areaj):
                    continue
                if not (0.8 < sepdists[sepi] / sepdists[sepj] < 1.2):
                    continue
                inter = sl.intersect(slicei, slicej)
                if (sl.empty(inter) or
                    (sl.area(inter) / areai < 0.2 and
                     sl.area(inter) / areaj < 0.2)):
                    nonoverlapping[i, j] = True
                    nonoverlapping[j, i] = True
        # find largest maximal clique (i.e. fully connected subgraphs)
        corrinds = corrinds[max(nx.find_cliques(nx.Graph(nonoverlapping)), key=len)]
        corrmap[corrs[corrinds, 0]] = corrs[corrinds[0], 0]
    _, corrmap = np.unique(corrmap, return_inverse=True) # make contiguous
    numsep = corrmap.max()
    LOG.debug("linked to %d separator candidates", numsep)
    def union(slices):
        if len(slices) > 1:
            return sl.union(slices[0], union(slices[1:]))
        return slices[0]
    for sep in range(1, numsep + 1):
        sepsizes[sep] = max(sepsizes[corrmap == sep]) # sum
        sepslices[sep] = union(sepslices[corrmap == sep])
    sepsizes = sepsizes[:numsep + 1]
    sepslices = sepslices[:numsep + 1]
    seplengths = np.array([np.hypot(*sl.dims(sepslice)) if sepslice else 0
                           for sepslice in sepslices])
    sepmap = corrmap[sepmap]
    DSAVE("seps-raw-linked", sepmap[labels])
    # order by size, filter minsize and filter top maxseps
    order = np.argsort(sepsizes)[::-1]
    # no more than maxseps and no smaller than scale
    minsize = np.flatnonzero((sepsizes[order] < scale) | (seplengths[order] < 3 * scale))
    if np.any(minsize):
        maxseps = min(maxseps, minsize[0])
    maxseps = min(maxseps, numsep)
    ordermap = np.zeros(numsep + 1, int)
    ordermap[order[:maxseps]] = np.arange(1, maxseps + 1)
    sepmap = ordermap[sepmap]
    DSAVE("sep-top", sepmap[labels])
    # spread into fg against other fg
    sepseeds = sepmap[labels]
    sepseeds = morph.spread_labels(sepseeds, maxdist=max(sepdists))
    sepseeds[~binary] = 0
    #labels = morph.propagate_labels_simple(binary, labels)
    #DSAVE("seps-top-spread-fg", sepseeds)
    # spread into bg against other fg
    sepseeds[binary & (sepseeds == 0)] = maxseps + 1
    seplabels = morph.spread_labels(sepseeds, maxdist=scale / 2)
    seplabels[seplabels == maxseps + 1] = 0
    DSAVE("seps-top-spread-bg", seplabels)
    return seplabels

# from ocropus-gpageseg, but with horizontal opening
@deprecated
def remove_hlines(binary,scale,maxsize=10):
    hlines = morph.select_regions(binary,sl.width,min=maxsize*scale)
    #DSAVE('hlines', hlines)
    # try to cut off text components that touch the lines
    h_open = morph.rb_opening(binary, (1, maxsize*scale))
    hlines = np.minimum(hlines, h_open)
    #DSAVE('hlines h-opened', hlines)
    return binary-hlines
        
# like remove_hlines, but much more robust (analoguous to compute_separators_morph)
@checks(ABINARY2,NUMBER)
def compute_hlines(binary, scale,
                   hlminwidth=10,
                   images=None):
    """Finds (and removes) horizontal black lines.

    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``scale``, square root of average bbox area of characters
    - ``hlminwidth``, minimum width (in ``scale`` multiples)
    (Minimum width for non-contiguous separators applies piece-wise.)
    - ``images``, an optional same-size int array as a non-text mask
      (to be ignored in fg)
    
    Returns a same-size bool array as a separator mask.
    """
    ## with zero horizontal dilation, hlines would need
    ## to be perfectly contiguous (i.e. without noise
    ## from binarization):
    d0 = odd(max(1,scale/8))
    d1 = odd(max(1,scale/4))
    # TODO This does not cope well with slightly sloped or heavily fragmented lines
    horiz = binary
    # 1- close horizontally a little to make warped or
    #    noisily binarized horizontal lines survive:
    horiz = morph.rb_closing(horiz, (d0,d1))
    DSAVE('hlines1_h-closed', horiz+0.6*binary)
    # 2- open horizontally to remove everything
    #    that is horizontally non-contiguous:
    opened = morph.rb_opening(horiz, (1, hlminwidth*scale//2))
    DSAVE('hlines2_h-opened', opened+0.6*binary)
    # 3- reconstruct the losses up to a certain distance
    #    to avoid creeping into overlapping glyphs but still
    #    cover most of the line even if not perfectly horizontal
    # (it would be fantastic if we could calculate the
    #  distance transform with stronger horizontal weights)
    #horiz = np.where(horiz, opened, 2)
    #horiz = morph.spread_labels(horiz, maxdist=d1) % 2 | opened
    horiz = morph.rb_reconstruction(opened, horiz, step=2, maxsteps=d1//2)
    DSAVE('hlines3_reconstructed', horiz+0.6*binary)
    if not horiz.any():
        return horiz > 0
    # 4- disregard parts from images; we don't want
    #    to compete/overlap with image objects too much,
    #    or waste our nbest on them
    if isinstance(images, np.ndarray):
        horiz = np.minimum(horiz,images==0)
        #horiz = morph.keep_marked(horiz, images==0)
        DSAVE('hlines5_noimages', horiz+0.6*binary)
    # 5- filter objects long enough:
    horiz = morph.select_regions(horiz, sl.width, min=hlminwidth*scale)
    DSAVE('hlines5_selected', horiz+0.6*binary)
    # 6- dilate vertically a little
    #    to get a smooth contour without gaps
    horiz = morph.r_dilation(horiz, (d0,odd(scale)))
    DSAVE('hlines6_v-dilated', horiz+0.6*binary)
    return horiz > 0

# from ocropus-gpageseg, but
# - much more robust for curved or non-contiguous lines
# - less intrusive w.r.t. adjacent characters
# - also based on reconstruction/seedfill
# - return label/contour-friendly mask (not just foreground)
@checks(ABINARY2,NUMBER)
def compute_separators_morph(binary, scale,
                             maxseps=2,
                             csminheight=10,
                             images=None):
    """Finds vertical black lines corresponding to column separators.

    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``scale``, square root of average bbox area of characters
    - ``maxseps``, maximum number of black separators to keep
    - ``csminheight``, minimum height (in ``scale`` multiples)
    (Non-contiguous separator lines count piece-wise. Equally,
     minimum height applies piece-wise.)
    - ``images``, an optional same-size int array as a non-text mask
      (to be ignored in fg)
    
    Returns a same-size bool array as separator mask.
    """
    if maxseps == 0:
        return binary == -1
    LOG.debug("considering at most %g black column separators", maxseps)
    ## no large vertical dilation here, because
    ## that would turn letters into lines; but
    ## with zero vertical share, vlines would need
    ## to be perfectly contiguous (i.e. without any
    ## noise from binarization etc):
    d0 = odd(max(1,scale/4))
    d1 = odd(max(1,scale/8))
    # TODO This does not cope well with slightly sloped or heavily fragmented lines
    vert = binary
    # 1- close vertically a little to make warped or
    #    noisily binarized vertical lines survive:
    vert = morph.rb_closing(vert, (d0,d1))
    DSAVE('colseps1_v-closed', vert+0.6*binary)
    # 2- open vertically to remove everything that
    #    is vertically non-contiguous:
    opened = morph.rb_opening(vert, (csminheight*scale//2, 1))
    DSAVE('colseps2_v-opened', opened+0.6*binary)
    # 3- reconstruct the losses up to a certain distance
    #    to avoid creeping into overlapping glyphs but still
    #    cover most of the line even if not perfectly vertical
    # (it would be fantastic if we could calculate the
    #  distance transform with stronger vertical weights)
    #vert = np.where(vert, opened, 2)
    #vert = morph.spread_labels(vert, maxdist=d1) % 2 | opened
    vert = morph.rb_reconstruction(opened, vert, step=2, maxsteps=d1//2)
    DSAVE('colseps3_reconstructed', vert+0.6*binary)
    if not vert.any():
        return vert > 0
    # 4- disregard parts from images; we don't want
    #    to compete/overlap with image objects too much,
    #    or waste our nbest on them
    if isinstance(images, np.ndarray):
        vert = np.minimum(vert,images==0)
        #vert = morph.keep_marked(vert, images==0)
        DSAVE('colseps4_noimages', vert+0.6*binary)
    # 5- select the n widest and highest segments
    #    above certain thresholds to be vertical lines:
    ## min=1 instead of 3, because lines can be 1 pixel thin
    vert = morph.select_regions(vert,sl.dim1,min=1,nbest=4* maxseps)
    vert = morph.select_regions(vert,sl.dim0,min=csminheight*scale,nbest=maxseps)
    DSAVE('colseps5_selected', vert+0.6*binary)
    # 6- dilate horizontally a little
    #    to get a smooth contour without gaps
    vert = morph.r_dilation(vert, (odd(scale),d1))
    DSAVE('colseps6_h-dilated', vert+0.6*binary)
    return vert > 0

# from ocropus-gpagseg, but
# - vertically dilate gradient edges _before_ (not after)
#   combining with thresholded whitespace
@checks(ABINARY2)
def compute_colseps_conv(binary, scale=1.0, csminheight=10, maxcolseps=2):
    """Find vertical whitespace corresponding to column separators by convolution and thresholding.
    
    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``scale``, square root of average bbox area of characters
    - ``csminheight``, minimum height (in ``scale`` multiples)
    - ``maxcolseps``, maximum number of separators to keep
    (Minimum height for non-contiguous separators applies piece-wise.)
    
    Returns a same-size bool array as separator mask.
    """
    if maxcolseps == 0:
        return binary == -1
    LOG.debug("considering at most %g whitespace column separators", maxcolseps)
    # find vertical whitespace by thresholding
    smoothed = filters.gaussian_filter(1.0*binary,(scale,scale*0.5))
    #smoothed = filters.uniform_filter(smoothed,(5.0*scale,1))
    # avoid blurring small/protruding glyphs below threshold
    smoothed = np.maximum(smoothed, filters.uniform_filter(smoothed,(5.0*scale,1)))
    thresh = (smoothed<np.amax(smoothed)*0.1)
    # note: maximum is unreliable
    #thresh = (smoothed<np.median(smoothed)*0.4) # 0.7
    # but median is also unreliable (depends on how much background there is)
    # maybe best use hist, bins = np.histogram(smoothed); bins[scipy.signal.find_peaks(-hist)[0]]
    DSAVE("colwsseps1_thresh",thresh+binary*0.6)
    # find column edges by filtering
    grad = filters.gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
    grad = filters.uniform_filter(grad,(10.0*scale,1)) # csminheight
    DSAVE("colwsseps2_grad-raw",grad)
    grad = grad > np.minimum(0.5 * np.amax(grad), np.percentile(grad, 99.5))
    DSAVE("colwsseps2_grad",grad)
    # combine dilated edges and whitespace
    seps = np.minimum(thresh,filters.maximum_filter(grad,(odd(10*scale),odd(5*scale))))
    DSAVE("colwsseps3_seps",seps+binary*0.6)
    # select only the biggest column separators
    seps = morph.select_regions(seps,sl.dim0,min=csminheight*scale,nbest=maxcolseps)
    DSAVE("colwsseps4_selected",seps+binary*0.6)
    return seps > 0

# from ocropus-gpageseg, but without apply_mask (i.e. input file)
@deprecated # it's better to remove vlines _before_ finding bg column seps, and to find column seps on a h/v-line free and boxmap cleaned binary
@checks(ABINARY2,NUMBER)
def compute_colseps(binary, scale, maxcolseps=3, maxseps=0, csminheight=7):
    """Computes column separators either from vertical black lines or whitespace.

    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``scale``, square root of average bbox area of characters
    - ``maxcolseps``, maximum number of white separators to keep
    - ``maxseps``, maximum number of black separators to keep
    - ``csminheight``, minimum height (in ``scale`` multiples)
    (Non-contiguous separator lines count piece-wise. Equally,
     minimum height applies piece-wise.)
    
    Returns a tuple:
    - same-size int array as (combined) separator mask,
    - same-size int array of foreground without black separators.
    """
    colseps = compute_colseps_conv(binary, scale,
                                   maxcolseps=maxcolseps,
                                   csminheight=csminheight)
    DSAVE("colwsseps",0.7*colseps+0.3*binary)
    if maxseps > 0:
        seps = compute_separators_morph(binary, scale,
                                        maxseps=maxseps,
                                        csminheight=csminheight)
        DSAVE("colseps", [binary, colseps, seps])
        #colseps = compute_colseps_morph(binary,scale)
        colseps = np.maximum(colseps,seps)
        binary = np.minimum(binary,1-seps)
    #binary,colseps = apply_mask(binary,colseps)
    return colseps,binary

# from ocropus-gpageseg, but with smaller box minsize and less horizontal blur
@checks(ABINARY2,NUMBER)
def compute_gradmaps(binary, scale,
                     usegauss=False,
                     fullpage=False,
                     vscale=1.0, hscale=1.0):
    # use gradient filtering to find baselines
    # default ocropy min,max scale filter: (0.5,4)
    if fullpage:
        # on complete pages, there is a good chance we also see
        # a wider range of glyph sizes (capitals, headings, paragraphs, footnotes)
        # or even large non-text blobs; all of those would contribute
        # to an *overestimation* of the scale by the median method;
        # so we might want a smaller lower boundary
        threshold = (0.5,4)
    else:
        # within regions/blocks, we will have remainders of
        # (possibly rotated and chopped) text lines at the
        # region boundaries, which we want to regard as
        # neighbouring independent/full lines (especially
        # during resegmentation)
        # so we could use a smaller minimum threshold
        threshold = (0.5,4)
    boxmap = psegutils.compute_boxmap(binary, scale, threshold=threshold)
    DSAVE("boxmap",boxmap)
    cleaned = boxmap*binary
    DSAVE("boxmap-cleaned",cleaned)
    # find vertical edges
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
            ##(vscale,hscale*6*scale))
    DSAVE("gradmap", grad)
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    DSAVE("bottom+top+boxmap", [0.5*boxmap + 0.5*binary, bottom, top])
    return bottom,top,boxmap

# from ocropus-gpageseg, but
# - with robust mode (can be disabled):
#   improved state transitions between bottom and top marks,
#   and avoid bleeding into next line with horizontal dilation
# - respect colseps early (during bottom/top marking)
# - default vscale=2 to better handle broken fonts
@checks(ABINARY2,AFLOAT2,AFLOAT2,ABINARY2,NUMBER)
def compute_line_seeds(binary,bottom,top,colseps,scale,
                       threshold=0.2,
                       # use larger scale so broken/blackletter fonts with their large capitals
                       # are not cut into two lines or joined at ascenders/decsenders:
                       vscale=2.0,
                       # more robust top/bottom transition rules:
                       robust=True):
    """Based on gradient maps, compute candidates for baselines and xheights.
    Then, mark the regions between the two as a line seed. Finally, label
    all connected regions (starting with 1, with 0 as background)."""
    vrange = odd(vscale*scale)
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
        bmarked = filters.maximum_filter(bmarked,(1,odd(scale))) *(1-colseps)
    tmarked = filters.maximum_filter(tmarked,(1,odd(scale))) *(1-colseps)
    ##tmarked = filters.maximum_filter(tmarked,(1,20))
    # why not just np.diff(bmarked-tmarked, axis=0, append=0) > 0 ?
    seeds = np.zeros(binary.shape,'i')
    delta = max(3,int(scale))
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
    DSAVE("lineseeds+bmarked+tmarked",[0.4*seeds+0.6*binary, bmarked, tmarked])
    if robust:
        # try to separate lines that already touch:
        seeds = morph.rb_opening(seeds, (odd(scale/2), odd(scale)))
    else:
        # this will smear into neighbouring line components at as/descenders
        # (but horizontal consistency is now achieved by hmerge and spread):
        seeds = filters.maximum_filter(seeds,(1,odd(1+scale)))
    DSAVE("lineseeds", seeds+0.6*binary)
    # interrupt by column separators before labelling:
    seeds = seeds*(1-colseps)
    DSAVE("lineseeds-colseps", seeds+0.6*binary)
    seeds, nlabels = morph.label(seeds)
    DSAVE("lineseeds_labelled", [seeds,binary])
    return seeds

@checks(ABINARY2,SEGMENTATION,NUMBER)
def hmerge_line_seeds(binary, seeds, scale, threshold=0.8, seps=None):
    """Relabel line seeds such that regions of coherent vertical
    intervals get the same label, and join them morphologically."""
    # merge labels horizontally to avoid splitting lines at long whitespace
    # (ensuring contiguous contours), but
    # ignore conflicts which affect only small fractions of either line
    # (avoiding merges for small vertical overlap):
    labels = np.unique(seeds * (binary > 0)) # without empty foreground
    labels = labels[labels > 0] # without background
    seeds[~np.isin(seeds, labels, assume_unique=True)] = 0
    #DSAVE("hmerge0_nonempty", seeds)
    if len(labels) < 2:
        return seeds
    objects = measurements.find_objects(seeds)
    centers = measurements.center_of_mass(binary, seeds, labels)
    relabel = np.arange(np.max(seeds)+1, dtype=seeds.dtype)
    # FIXME: get incidence of y overlaps to avoid full inner loops
    LOG.debug('checking %d non-empty line seeds for overlaps', len(labels))
    def h_compatible(obj1, obj2, center1, center2):
        if not (obj2[0].start < center1[0] < obj2[0].stop):
            return False
        if not (obj1[0].start < center2[0] < obj1[0].stop):
            return False
        if (obj2[1].start < center1[1] < obj2[1].stop):
            return False
        if (obj1[1].start < center2[1] < obj1[1].stop):
            return False
        return True
    for label in labels:
        seed = seeds == label
        if not seed.any():
            continue
        #DSAVE('hmerge1_seed', seed)
        # close to fill holes from underestimated scale
        seed = morph.rb_closing(seed, (scale, scale))
        #DSAVE('hmerge2_closed', seed)
        # not really necessary (seed does not contain ascenders/descenders):
        # # open horizontally to remove extruding ascenders/descenders
        # seed = morph.rb_opening(seed, (1, 3*scale))
        # DSAVE('hmerge3_h-opened', seed)
        if not seed.any():
            continue
        obj = measurements.find_objects(seed)[0]
        if obj is None:
            continue
        seed[obj[0], 0:seed.shape[1]] = 1
        #DSAVE('hmerge4_h-closed', seed)
        # get overlaps
        for label2 in labels:
            if label == label2 or relabel[label] == label2:
                continue
            obj2 = objects[label2-1]
            if not obj2:
                continue
            if not sl.yoverlaps(obj, obj2):
                continue
            center = centers[labels.searchsorted(label)]
            bbox = objects[label-1]
            if not all(h_compatible(bbox, bbox2, center, center2)
                       for bbox2, center2 in [(objects[i-1], centers[labels.searchsorted(i)])
                                              for i in np.nonzero(relabel == relabel[label2])[0]]):
                LOG.debug('ignoring h-overlap between %d and %d (not mutually centric)', label, label2)
                continue
            seed2 = seeds == label2
            count = np.count_nonzero(seed2 * seed)
            total = np.count_nonzero(seed2)
            if count < threshold * total:
                LOG.debug('ignoring h-overlap between %d and %d (only %d of %d)', label, label2, count, total)
                continue
            label1_y, label1_x = np.where(seeds == label)
            label2_y, label2_x = np.where(seed2)
            shared_y = np.intersect1d(label1_y, label2_y)
            gap = np.zeros_like(seed2, bool)
            for y in shared_y:
                can_x_min = label2_x[label2_y == y][0]
                can_x_max = label2_x[label2_y == y][-1]
                new_x_min = label1_x[label1_y == y][0]
                new_x_max = label1_x[label1_y == y][-1]
                if can_x_max < new_x_min:
                    if (seps is None or
                        not seps[y, can_x_max:new_x_min].any()):
                        gap[y, can_x_max:new_x_min] = True
                if new_x_max < can_x_min:
                    if (seps is None or
                        not seps[y, new_x_max:can_x_min].any()):
                        gap[y, new_x_max:can_x_min] = True
            if not gap.any() or gap.max(axis=1).sum() / len(shared_y) < 0.1:
                LOG.debug('ignoring h-overlap between %d and %d (blocked by seps)', label, label2)
                continue
            # find y with shortest gap
            gapwidth = gap.sum(axis=1)
            gapwidth[gapwidth==0] = seed.shape[1]
            mingap = gapwidth < gapwidth.min() + 4
            # make contiguous
            mingap = mingap.nonzero()[0]
            gap[0:mingap[0]] = False
            gap[mingap[-1]:] = False
            LOG.debug('hmerging %d with %d', label2, label)
            # fill the horizontal background between both regions:
            seeds[gap] = label
            # the new label could have been relabelled already:
            new_label = relabel[label]
            # assign candidate to (new assignment for) label:
            relabel[label2] = new_label
            # re-assign labels already relabelled to candidate:
            relabel[relabel == label2] = new_label
    # apply re-assignments:
    seeds = relabel[seeds]
    # DSAVE("hmerge5_connected", seeds)
    return seeds
        
# from ocropus-gpageseg, but:
# - with fullpage switch
#   (opt-in for separator line and column detection),
# - with external separator mask
#   (opt-in for separator line pass-through)
# - with zoom parameter
#   (make fixed dimension params relative to pixel density,
#    instead of blind 300 DPI assumption)
# - with improved separator line and column detection
# - with separator detection _before_ column detection
# - with separator suppression _after_ large component filtering
# - with more robust line seed estimation,
# - with horizontal merge instead of blur,
# - with component majority for foreground
#   outside of seeds (instead of spread),
#   except for components with seed conflict
#   (which must be split anyway)
# - with tighter polygonal spread around foreground
# - with spread of line labels against separator labels
# - with baseline extraction
# - return bg line and sep labels intead of just fg line labels
# - return baseline coords, too
@checks(ABINARY2)
def compute_segmentation(binary,
                         zoom=1.0,
                         fullpage=False,
                         seps=None,
                         maxcolseps=2,
                         csminheight=4,
                         maxseps=0,
                         maximages=0,
                         spread_dist=None,
                         rl=False,
                         bt=False):
    """Find text line segmentation within a region or page.

    Given a binarized (and inverted) image as Numpy array ``image``, compute
    a complete segmentation of it into text lines as a label array.

    If ``fullpage`` is false, then
    - avoid single-line horizontal splits, and
    - ignore any foreground in ``seps``.

    If ``fullpage`` is true, then
    - allow all horizontal splits, and search
    - for up to ``maxcolseps`` multi-line vertical whitespaces
      (as column separators, counted piece-wise) of at least
      ``csminheight`` multiples of ``scale``,
    - for up to ``maxseps`` black separator lines (horizontal, vertical
      or oblique; counted piece-wise),
    - for anything in ``seps`` if given,
    then suppress these non-text components and return them separately.
    
    Labels will be projected ("spread") from the foreground to the
    surrounding background within ``spread_dist`` distance (or half
    the estimated scale).
    
    Respect the given reading order:
    - ``rl``, whether to sort text lines in reverse order (from right to left),
    - ``bt``, whether to sort text lines in reverse order (from bottom to top).

    Return a tuple of:
    - Numpy array of the textline background labels
      (not the foreground or the masked image;
       foreground may remain unlabelled for
       separators and other non-text like small
       noise, or large drop-capitals / images),
    - list of Numpy arrays of baseline coordinates [y, x points in lr order]
    - Numpy array of foreground separator lines mask,
    - Numpy array of large/non-text foreground component mask,
    - Numpy array of vertical background separators mask,
    - the estimated scale (i.e. median sqrt bbox area of glyph components).
    """
    # TODO generalize to multi-scale (with `scale` as group array instead of float)
    DSAVE("input_binary",binary)
    LOG.debug('estimating glyph scale')
    scale = psegutils.estimate_scale(binary, zoom)
    LOG.debug('height: %d, zoom: %.2f, scale: %d', binary.shape[0], zoom, scale)

    if fullpage:
        LOG.debug('detecting images')
        images = compute_images(binary, scale, maximages=maximages)
        LOG.debug('detecting separators')
        #hlines = compute_hlines(binary, scale, hlminwidth=hlminwidth, images=images)
        #vlines = compute_separators_morph(binary, scale, csminheight=csminheight, maxseps=maxseps, images=images)
        slines = compute_seplines(binary, scale, maxseps=maxseps)
        binary = np.minimum(binary, 1 - (slines > 0))
        binary = np.minimum(binary, 1 - (images > 0))
    else:
        slines = np.zeros_like(binary, np.uint8)
        images = np.zeros_like(binary, np.uint8)
    if seps is not None and not seps.all():
        # suppress separators/images for line estimation
        # (unless it encompasses the full image for some reason)
        binary = (1-seps) * binary

    LOG.debug('computing gradient map')
    bottom, top, boxmap = compute_gradmaps(binary, scale,
                                           usegauss=False,
                                           fullpage=fullpage)
    if fullpage:
        LOG.debug('finding whitespace column separators')
        colseps = compute_colseps_conv(binary, scale,
                                       maxcolseps=maxcolseps,
                                       csminheight=csminheight)
        DSAVE("colseps",0.7*colseps+0.3*binary)
        # get a larger (closed) mask of all separators
        # (both bg boundary and fg line seps, detected
        # and passed in) to separate line/column labels
        sepmask = np.maximum(slines > 0, images > 0)
        sepmask = np.maximum(sepmask, colseps)
        if seps is not None:
            sepmask = np.maximum(sepmask, seps)
        sepmask = morph.r_closing(sepmask, (scale, scale))
        DSAVE("sepmask",0.7*sepmask+0.3*binary)
    else:
        colseps = np.zeros(binary.shape, np.uint8)
        sepmask = np.zeros(binary.shape, np.uint8)

    LOG.debug('computing line seeds')
    seeds = compute_line_seeds(binary, bottom, top, sepmask, scale)
    if fullpage:
        # filter labels that have only noise fg (e.g. from split)
        invalid = np.setdiff1d(seeds.flatten(), (seeds*binary*boxmap).flatten())
        relabel = np.arange(np.max(seeds)+1)
        relabel[invalid] = 0
        seeds = relabel[seeds]
        DSAVE("lineseeds_filtered", [seeds,binary])
    else:
        seeds = hmerge_line_seeds(binary, seeds, scale, seps=sepmask)

    LOG.debug('spreading seed labels')
    # spread labels from seeds to bg, but watch fg,
    # voting for majority on bg conflicts,
    # but splitting on seed conflicts
    llabels = morph.propagate_labels_majority(binary, seeds)
    llabels2 = morph.propagate_labels(binary, seeds, conflict=0)
    conflicts = llabels > llabels2
    llabels = np.where(conflicts, seeds, llabels)
    # capture diacritics (isolated components above seeds)
    seeds2 = interpolation.shift(seeds, (-scale, 0), order=0, prefilter=False)
    seeds2 = np.where(seeds, seeds, seeds2)
    DSAVE('lineseeds_cap', [seeds2,binary])
    llabels2 = morph.propagate_labels_simple(binary, seeds2)
    llabels = np.where(llabels, llabels, llabels2)
    # (protect sepmask as a temporary label)
    seplabel = np.max(seeds)+1
    llabels[sepmask>0] = seplabel
    spread = morph.spread_labels(llabels, maxdist=spread_dist or scale/2)
    DSAVE('lineseeds_spread', [spread,binary])
    llabels2 = morph.propagate_labels_majority(binary, spread)
    llabels = np.where(seeds, seeds, llabels2)
    llabels[sepmask>0] = seplabel
    llabels = morph.spread_labels(llabels, maxdist=spread_dist or scale/2)
    llabels[llabels==seplabel] = 0
    DSAVE('llabels', [llabels,binary])
    
    LOG.debug('sorting labels by reading order')
    llabels = morph.reading_order(llabels,rl,bt)[llabels]
    DSAVE('llabels_ordered', llabels)

    #segmentation = llabels*binary
    #return segmentation
    blines = compute_baselines(bottom, top, llabels, scale)
    return llabels, blines, slines, images, colseps, scale

@checks(AFLOAT2,AFLOAT2,SEGMENTATION,NUMBER)
def compute_baselines(bottom, top, linelabels, scale, method='bottom'):
    """Get the coordinates of baselines running along each bottom gradient peak."""
    seeds = linelabels > 0
    # smooth bottom+top maps horizontally for centerline estimation
    bot = filters.gaussian_filter(bottom, (scale*0.25,scale), mode='constant')
    top = filters.gaussian_filter(top, (scale*0.25,scale), mode='constant')
    # idea: center is where bottom and top gradient meet in the middle
    # (but between top and bottom, not between bottom and top)
    # - calculation via numpy == or isclose is too fragile numerically:
    #clines = np.isclose(top, bottom, rtol=0.5) & (np.diff(top - bottom, axis=0, append=0) < 0)
    # - calculation via zero crossing of bop-bottom is more robust,
    #   but needs post-processing for lines with much larger height than scale
    if method == 'center':
        blines = (np.diff(np.sign(top - bottom), axis=0, append=0) < 0) & seeds
        #DSAVE('centerlines', blines)
    # - calculation via peak gradient
    elif method == 'bottom':
        bot1d = np.diff(bot, axis=0, append=0)
        bot1d = np.diff(np.sign(bot1d), axis=0, append=0) < 0
        bot1d &= bot > 0
        DSAVE('bot1d', bot1d)
        blines = bot1d
    baselabels, nbaselabels = morph.label(blines)
    baseslices = [(slice(0,0),slice(0,0))] + morph.find_objects(baselabels)
    # if multiple labels per seed, ignore the ones above others
    # (can happen due to mis-estimation of scale)
    corrs = morph.correspondences(linelabels, baselabels).T
    labelmap = {}
    DSAVE('baselines-raw', baselabels)
    for line in np.unique(linelabels):
        if not line: continue # ignore bg line
        corrinds = corrs[:, 0] == line
        corrinds[corrs[:, 1] == 0] = False # ignore bg baseline
        if not np.any(corrinds): continue
        corrinds = corrinds.nonzero()[0]
        if len(corrinds) == 1:
            labelmap.setdefault(line, list()).append(corrs[corrinds[0], 1])
            continue
        nonoverlapping = ~np.eye(len(corrinds), dtype=bool)
        for i, indi in enumerate(corrinds[:-1]):
            baselabeli = corrs[indi, 1]
            baseslicei = baseslices[baselabeli]
            for j, indj in enumerate(corrinds[i + 1:], i + 1):
                baselabelj = corrs[indj, 1]
                baseslicej = baseslices[baselabelj]
                if sl.xoverlaps(baseslicei, baseslicej):
                    nonoverlapping[i, j] = False
                    nonoverlapping[j, i] = False
        # find all maximal cliques in the graph (i.e. all fully connected subgraphs)
        # and then pick the partition with the largest sum of pixels at its nodes
        def pathlen(path):
            return sum(corrs[corrinds[path], 2])
        corrinds = corrinds[max(nx.find_cliques(nx.Graph(nonoverlapping)), key=pathlen)]
        labelmap.setdefault(line, list()).extend(corrs[corrinds, 1])
    basepoints = []
    for line in np.unique(linelabels):
        if line not in labelmap: continue
        linemask = linelabels == line
        points = []
        for label in labelmap[line]:
            points.extend(list(zip(*np.where((baselabels == label) & linemask))))
        basepoints.append(points)
    return basepoints

# from ocropus-gpageseg, but
# - on both foreground and background,
# - as separate step on the PIL.Image
@checks(Image.Image)
def remove_noise(pil_image, maxsize=8):
    array = pil2array(pil_image)
    binary = np.array(array <= ocrolib.midrange(array), np.uint8)
    # TODO we should use opening/closing against fg/bg noise instead pixel counting
    clean_bg = ocrolib.remove_noise(binary, maxsize)
    clean_fg = ocrolib.remove_noise(1 - binary, maxsize)
    if LOG.getEffectiveLevel() <= logging.DEBUG:
        _, ncomps_before = morph.label(binary)
        _, ncomps_after = morph.label(clean_bg)
        LOG.debug('components before/after black denoising (maxsize=%d): %d/%d',
                  maxsize, ncomps_before, ncomps_after)
        _, ncomps_after = morph.label(1 - clean_fg)
        LOG.debug('components before/after white denoising (maxsize=%d): %d/%d',
                  maxsize, ncomps_before, ncomps_after)
    array = np.maximum(array, binary - clean_bg) # cleaned bg becomes white
    array = np.minimum(array, binary + clean_fg) # cleaned fg becomes black
    return array2pil(array)

@checks(ABINARY2,SEGMENTATION)
def lines2regions(binary, llabels,
                  rlabels=None,
                  sepmask=None,
                  prefer_vertical=None,
                  rl=False, bt=False,
                  min_line=4.0,
                  gap_height=0.01,
                  gap_width=1.5,
                  scale=None, zoom=1.0):
    """Aggregate text lines to text regions by running hybrid recursive X-Y cut.
    
    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
      (including text lines, separators, images etc)
    - ``llabels``, a segmentation of the page into adjacent textlines
      (including locally correct reading order)
    - (optionally) ``rlabels``, an initial solution as a (possibly partial)
      segmentation of the page into region labels (assignment of line labels);
      these regions will stay grouped together, but will be re-assigned labels
      in the order of cutting/partitioning
    - (optionally) ``sepmask``, a mask array of fg or bg separators;
      it is applied before, but also during recursive X-Y cut:
      In each iteration's box, if sepmask creates enough partitions
      (not empty in fg and not significantly crossing line labels),
      and horizontal or vertical cuts are not very prominent already,
      or don't offer as many valid slices as there would be partitions,
      then use those partitions instead, passing down a mask to apply
      for each partition; partitions can also be re-partitioned, just
      not on the very next level of recursion
    - (optionally) ``prefer_vertical``, whether to prefer
      vertical cuts (into columns) over horizontal cuts (into rows)
      when the choice is not straightforward; set this to
      - True, when the page can be expected to be dominated by columns
      - False, when the page has table semantics
      - None, when geometry alone should decide each time
       (i.e. direction with widest and lowest gaps)
    - ``rl``, whether to sort vertical cuts/partitions in reverse order
      (from right to left),
    - ``bt``, whether to sort horizontal cuts/partitions in reverse order
      (from bottom to top),
    - ``min_line``, minimum number of fg pixels (in multiples
      of ``scale``) for a line label to be regarded as significant
      w.r.t. block segmentation (i.e. to not split across regions)
    - ``gap_height``, largest minimum pixel average in
       the horizontal or vertical profiles to still be regarded as
       a gap (needs to be larger when foreground noise is present;
       reduce to avoid mistaking text for noise)
    - ``gap_width``, smallest width (in multiples of scale)
       of a valley in the horizontal or vertical profiles
       to still be regarded as a gap (needs to be smaller
       when foreground noise is present; increase to avoid
       mistaking inter-line as paragraph gaps and
       inter-word as inter-column gaps)
    - ``scale``, square root of the average bbox area of characters
      (as determined during line segmentation)
    
    Split the image recursively into horizontal or vertical slices.
    For each slice (at some stack depth), find gaps running completely
    across the binary (foreground) of that slice, either vertically or
    horizontally. Gaps must have a certain minimum width (dependent on
    the scale) and height (dependent on the level of noise), but also
    a certain distance between each other (also dependent on the
    scale, corresponding to the number of lines).  However, gaps must
    not cut (significant parts of) any single line label.
    
    Usually, gaps in one direction are much more prominent than in the
    other, and certain gaps within one direction more so than others.
    Therefore, always choosing the most prominent gap(s) is already
    sufficient to guarantee some alternation between horizontal and
    vertical direction. But for cases where the difference is rather
    small, ``prefer_vertical`` may be set (True for multiple columns, or
    False for multiple rows) to swing the decision (otherwise the better
    direction wins).
    
    Then recursively enter each slice between the chosen cuts.
    Iterate the slices top-down (unless ``bt``) and left-right (unless
    ``rl``).
    
    Alternatively, find a partitioning using ``sepmask`` instead of
    gaps. If the separators split the current slice into multiple
    independent parts, each of which has a number of line labels (in a
    significant foreground share), and which does not cut any single
    line label, then instead of slicing, sort the partitions by
    reading order, and recursively enter each partition with the
    respective others masked away in the foreground.
    
    Thus, there will be an alternation between horizontal and vertical
    cuts, as well as non-rectangular partitioning from h/v-lines and
    column separators.
    
    Each slice which cannot be cut/partitioned any further gets a new
    region label (in the order of the call chain, which is controlled
    by ``rl`` and ``bt``), covering all the line labels inside it.
    If ``rlabels`` is given, use this for initial regions.
    
    Afterwards, for each region label, simplify regions by using
    their convex hull polygon.
    
    Return a Numpy array of text region labels.
    """
    lbinary = binary * llabels
    # suppress separators (weak integration)
    if isinstance(sepmask, np.ndarray):
        lbinary *= sepmask == 0
        # prepare sepmask partitioning (see below):
        # where sepmask partitions would be empty,
        # add them to sepmask (to avoid adding fake partitions)
        sepmask = 1-morph.keep_marked(1-sepmask, lbinary>0)
        DSAVE('sepmask', [sepmask,binary])
    objects = [None] + morph.find_objects(llabels)
    # centers = measurements.center_of_mass(binary, llabels, np.unique(llabels))
    # def center(obj):
    #     if morph.sl.empty(obj):
    #         return [0,0]
    #     return morph.sl.center(obj)
    # centers = list(map(center, objects[1:]))
    if scale is None:
        scale = psegutils.estimate_scale(binary, zoom)
    bincounts = np.bincount(lbinary.flatten())
    
    LOG.debug('combining lines to regions')
    relabel = np.zeros(np.amax(llabels)+1, int)
    num_regions = 0
    def recursive_x_y_cut(box, mask=None, partition_type=None, debug=False):
        """Split lbinary at horizontal or vertical gaps recursively.
        
        - ``box`` current slice
        - ``mask`` (optional) binary mask for current box to focus
          line labels on (passed+sliced down recursively)
        - ``partition_type`` whether ``mask`` was created by partitioning
          immediately before (without any intermediate cuts), and thus
          must not be repeated in the current iteration
        
        Modifies ``relabel`` and ``num_regions``.
        """
        lbin = sl.cut(lbinary, box)
        if isinstance(mask, np.ndarray):
            lbin = np.where(mask, lbin, 0)
        def finalize():
            """Assign current line labels into new region, and re-order them inside."""
            nonlocal num_regions
            nonlocal relabel
            linelabels = np.setdiff1d(lbin, [0])
            if debug: LOG.debug('checking line labels %s for conflicts', str(linelabels))
            # when there is a conflict for a line label, assign (or keep) the more frequent region label
            linelabels = [label
                          for label in linelabels
                          if (not relabel[label] or
                              np.count_nonzero(lbin == label) > 0.5 * bincounts[label])]
            if not linelabels:
                return
            if rlabels is None:
                num_regions += 1
                if debug:
                    LOG.debug('new region {} for lines {}'.format(num_regions, linelabels))
                else:
                    LOG.debug('new region %d for %d lines', num_regions, len(linelabels))
                relabel[linelabels] = num_regions
            else:
                # (partial) initial segmentation exists - order existing groups against rest,
                # this must be done on full labels (bg+fg), so we first need to reconstruct
                # this slice's llab/rlab
                rlab = sl.cut(rlabels, box)
                if isinstance(mask, np.ndarray):
                    rlab = np.where(mask, rlab, 0)
                llab = sl.cut(llabels, box)
                if isinstance(mask, np.ndarray):
                    llab = np.where(mask, llab, 0)
                linelabels0 = np.zeros(llabels.max()+1, dtype=bool)
                linelabels0[linelabels] = True
                llab *= linelabels0[llab]
                newregion = rlab.max()+1
                rlab = np.where(llab, np.where(rlab, rlab, newregion), 0)
                order = np.argsort(morph.reading_order((lbin>0) * rlab, rl, bt))
                # get region label with highest share for each line,
                # then assign it to that region
                llab2rlab, llabcount = dict(), dict()
                for line, region, count in morph.correspondences(llab, rlab).T:
                    if line > 0 and region > 0 and count > llabcount.get(line, 0):
                        llabcount[line] = count
                        llab2rlab[line] = region
                rlab2llab = dict()
                for line, region in llab2rlab.items():
                    rlab2llab.setdefault(region, list()).append(line)
                for region in order:
                    if not region in rlab2llab:
                        continue
                    lines = rlab2llab[region]
                    num_regions += 1
                    if region == newregion:
                        if debug:
                            LOG.debug('new region {} for lines {}'.format(num_regions, lines))
                        else:
                            LOG.debug('new region %d for %d lines', num_regions, len(lines))
                    else:
                        if debug:
                            LOG.debug('new region {} for existing region {} lines {}'.format(num_regions, region, lines))
                        else:
                            LOG.debug('new region %d for existing region %d', num_regions, region)
                    relabel[lines] = num_regions
        
        _, lcounts = np.unique(lbin, return_counts=True)
        if (len(lcounts) <= 2 or
            sum(1 for count in lcounts if count > scale) <= 2):
            # only one label plus background left
            finalize()
            return
        
        # try split via annotated separators (strong integration)
        # i.e. does current slice of sepmask contain true partitions?
        # (at least 2 partitions which contain at least 1 line label each,
        #  where each line label in that partition in the current slice
        #  must cover a significant part of that line label in the full image)
        partitions, npartitions = None, 0
        if (isinstance(sepmask, np.ndarray) and
            np.count_nonzero(sepmask)):
            sepm = sl.cut(sepmask, box)
            if isinstance(mask, np.ndarray):
                sepm = np.where(mask, sepm, 1)
            # provide `partitions` for next step
            partitions, npartitions = 1-sepm, 1
            new_partition_type = None
            # try to find `partitions` in current step
            if partition_type != 'splitmask':
                # sepmask not applied yet, or already applied in higher X-Y branch:
                # try to apply in this cut like another separator
                partitions, npartitions = morph.label(1-sepm)
                if npartitions > 1:
                    # delete partitions that have no significant line labels,
                    # merge partitions that share any significant line labels
                    splitmap = np.zeros((len(objects), npartitions), dtype=bool)
                    for label in range(npartitions):
                        linecounts = np.bincount(lbin[partitions==label+1], minlength=len(objects))
                        linecounts[0] = 0 # without bg
                        # get significant line labels for this partition
                        # (but keep insignificant non-empty labels if complete)
                        mincounts = np.minimum(min_line * scale, np.maximum(1, bincounts))
                        linelabels = np.nonzero(linecounts >= mincounts)[0]
                        if linelabels.size:
                            splitmap[linelabels, label] = True
                            if debug: LOG.debug('  sepmask partition %d: %s', label+1, str(linelabels))
                        else:
                            partitions[partitions==label+1] = 0
                    if isinstance(rlabels, np.ndarray):
                        # keep existing regions in distinct partitions if possible
                        rlab = sl.cut(rlabels, box)
                        if isinstance(mask, np.ndarray):
                            rlab = np.where(mask, rlab, 0)
                        splitmap[np.unique(lbin[rlab>0])] = False
                    mergemap = np.arange(npartitions + 1)
                    for line in splitmap:
                        if not np.any(line):
                            continue
                        parts = np.flatnonzero(line)+1
                        mergemap[parts] = mergemap[parts[0]]
                    partitions = mergemap[partitions]
                    npartitions = len(np.setdiff1d(np.unique(mergemap), [0]))
                    new_partition_type = 'splitmask'
                    if debug: LOG.debug('  %d sepmask partitions after filtering and merging', npartitions)
            if partition_type != 'topological':
                # try to partition spanning lines against separator-split lines
                # get current slice's line labels
                def find_topological():
                    # run only if needed (no other partition/slicing possible)
                    nonlocal sepm, partitions, npartitions, new_partition_type
                    llab = sl.cut(llabels, box)
                    if isinstance(mask, np.ndarray):
                        llab = np.where(mask, llab, 0)
                    if isinstance(rlabels, np.ndarray):
                        # treat existing regions like separators
                        rlab = sl.cut(rlabels, box)
                        if isinstance(mask, np.ndarray):
                            rlab = np.where(mask, rlab, 0)
                        sepm = np.where(rlab, 1, sepm)
                    obj = [sl.intersect(o, box) for o in objects]
                    # get current slice's foreground
                    bin = sl.cut(binary, box)
                    if isinstance(mask, np.ndarray):
                        bin = np.where(mask, bin, 0)
                    # get current slice's separator labels
                    seplab, nseps = morph.label(sepm)
                    if nseps == 0:
                        return
                    # sepind = np.unique(seplab)
                    # (but keep only those with large fg i.e. ignore white-space seps)
                    seplabs, counts = np.unique(seplab * bin, return_counts=True)
                    kept = np.in1d(seplab.ravel(), seplabs[counts > scale * min_line])
                    seplab = seplab * kept.reshape(*seplab.shape)
                    #DSAVE('seplab', seplab)
                    sepobj = morph.find_objects(seplab)
                    if not len(sepobj):
                        return
                    # get current slice's line labels
                    # (but keep only those with foreground)
                    linelabels = np.setdiff1d(np.unique(lbin), [0])
                    nlines = linelabels.max() + 1
                    # find pairs of lines above each other with a separator next to them
                    leftseps = np.zeros((nlines, nseps), bool)
                    rghtseps = np.zeros((nlines, nseps), bool)
                    for line in linelabels:
                        for i, sep in enumerate(sepobj):
                            if sep is None:
                                continue
                            if sl.yoverlap(sep, obj[line]) / sl.height(obj[line]) <= 0.75:
                                continue
                            sepx = np.nonzero(seplab[obj[line][0]] == i + 1)[1]
                            binx = np.nonzero(lbin[obj[line][0]] == line)[1]
                            if not binx.size:
                                continue
                            # more robust to noise: 95% instead of max(), 5% instead of min()
                            if sepx.max() <= np.percentile(binx, 5):
                                leftseps[line, i] = True
                            if sepx.min() >= np.percentile(binx, 95):
                                rghtseps[line, i] = True
                    # true separators have some lines on either side
                    trueseps = leftseps.max(axis=0) & rghtseps.max(axis=0)
                    if not np.any(trueseps):
                        return
                    if debug: LOG.debug("trueseps: %s", str(trueseps))
                    neighbours = np.zeros((nlines, nlines), bool)
                    for i in linelabels:
                        for j in linelabels[i+1:]:
                            if sl.yoverlap_rel(obj[i], obj[j]) > 0.5:
                                continue
                            # pair must have common separator on one side,
                            # which must also have some line on the other side
                            if (np.any(leftseps[i] & leftseps[j] & trueseps) or
                                np.any(rghtseps[i] & rghtseps[j] & trueseps)):
                                if debug: LOG.debug("neighbours: %d/%d", i, j)
                                neighbours[i,j] = True
                    if not np.any(neighbours):
                        return
                    # group neighbours by adjacency (i.e. put any contiguous pairs
                    # of such line labels into the same group)
                    nlabels = llab.max() + 1
                    splitmap = np.zeros(nlabels, dtype=int)
                    for i, j in zip(*neighbours.nonzero()):
                        if splitmap[i] > 0:
                            splitmap[j] = splitmap[i]
                        elif splitmap[j] > 0:
                            splitmap[i] = splitmap[j]
                        else:
                            splitmap[i] = i
                            splitmap[j] = i
                    nsplits = splitmap.max()
                    # group non-neighbours by adjacency (i.e. put any other contiguous
                    # non-empty line labels into the same group)
                    nonneighbours = (splitmap==0)[llab] * (llab > 0) * (sepm == 0)
                    nonneighbours, _ = morph.label(nonneighbours)
                    for i, j in morph.correspondences(nonneighbours, llab, False).T:
                        if i > 0 and j > 0:
                            splitmap[j] = i + nsplits
                    if debug: LOG.debug('  groups of adjacent lines: %s', str(splitmap))
                    partitions = splitmap[llab]
                    DSAVE('partitions', partitions)
                    npartitions = len(np.setdiff1d(np.unique(splitmap), [0]))
                    if npartitions > 1:
                        if debug: LOG.debug("  found %d spanning partitions", npartitions)
                        # re-assign background to nearest partition
                        partitions = morph.spread_labels(np.where(llab, partitions, 0))
                        # re-assert mask, if any
                        if isinstance(mask, np.ndarray):
                            partitions = np.where(mask, partitions, 0)
                        new_partition_type = 'topological'
            else:
                def find_topological():
                    return
        
        # try cuts via h/v projection profiles
        y = np.mean(lbin>0, axis=1)
        x = np.mean(lbin>0, axis=0)
        # smoothed to avoid splitting valleys into gorges due to noise
        y = filters.gaussian_filter(y, scale/4)
        x = filters.gaussian_filter(x, scale/4)
        if debug:
            # show current cut/box inside full image
            llab = relabel[llabels]
            llab = np.where(llab, llab, llabels)
            if isinstance(mask, np.ndarray):
                llab[box] = np.where(mask, lbinary[box], 0)
            else:
                llab[box] = lbinary[box]
            # show projection at the sides
            for i in range(int(scale/2)):
                llab[box[0],box[1].start+i] = -10*np.log(y+1e-9)
                llab[box[0],box[1].stop-1-i] = -10*np.log(y+1e-9)
                llab[box[0].start+i,box[1]] = -10*np.log(x+1e-9)
                llab[box[0].stop-1-i,box[1]] = -10*np.log(x+1e-9)
            DSAVE('recursive_x_y_cut_' + (partition_type or 'sliced'), llab)
        gap_weights = list()
        for is_horizontal, profile in enumerate([y, x]):
            # find gaps in projection profiles
            # (measured as product of height and width,
            #  because we want robustness against noise)
            gaps, props = signal.find_peaks(
                # negative because we want minima
                -profile,
                # tolerate minimal noise
                height=-gap_height,
                # at least 2 average lines (or equivalently,
                # 1 large heading) in between: only best peaks
                distance=4*scale, # (but SciPy seems to have bug in peak discounting)
                # width should be derived from training on GT,
                # cf. Sylwester&Seth (1998): A trainable, single-pass algorithm for column segmentation
                width=gap_width*scale,
                # 'width' begins/ends at this share of height over base
                # (the smaller it becomes, the harder it is to meet the width threshold)
                rel_height=0.20)
            weights = props['widths']
            if gap_height:
                # when non-zero valleys are allowed, multiply width by penalty
                # e.g. log height (a height of 0.015 would become factor 0.20)
                #weights = weights * np.log(1e-9 - props['peak_heights'])/np.log(1e-9)
                # e.g. normalized linear (marginal gap_height would become 0.5)
                weights = weights * (1 + 0.5 * props['peak_heights']/gap_height)
            gap_weights.append((gaps, weights))
            if debug:
                LOG.debug('  {} gaps {} {} weights {}'.format(
                    'horizontal' if is_horizontal else 'vertical',
                    gaps, props, weights))
                if not gaps.shape[0]:
                    continue
                for start, stop, height in sorted(zip(
                        props['left_ips'].astype(int),
                        props['right_ips'].astype(int),
                        props['peak_heights']), key=lambda x: x[2]):
                    if is_horizontal:
                        llab[box[0].start+int(scale/2):box[0].stop-int(scale/2),box[1].start+start:box[1].start+stop] = -10*np.log(-height+1e-9)
                    else:
                        llab[box[0].start+start:box[0].start+stop,box[1].start+int(scale/2):box[1].stop-int(scale/2)] = -10*np.log(-height+1e-9)
                DSAVE('recursive_x_y_cut_gaps_' + ('h' if is_horizontal else 'v'), llab)
        # heuristic (not strict) decision on x or y cut,
        # factors to consider:
        # - number of minima [not used]
        # - width of minima
        # - height of minima
        # - (if sepmask is given:) number of partitions created
        # principles to uphold (when uncertain):
        # - for tables, prefer horizontal cuts - for "cell" like
        #   reading order (applied via ``prefer_vertical=False``
        #   when segmenting table regions)
        # - for text, prefer vertical cuts - for "paragraph" like
        #   reading order (applied via ``prefer_vertical=True``
        #   when segmenting full pages)
        # - generally, prefer most prominent direction, which
        #   will implicitly alternate between h and v cuts --
        #   when largest gap weight in less prominent direction
        #   becomes much larger than second-largest in prominent
        #   direction after cutting at largest gap there
        # - within each direction, only the largest and/or
        #   most partitioning gaps win (the others will have to
        #   wait re-appearing in a lower-level cut)
        # - cuts which would split line labels significantly
        #   are not allowed
        y_gaps, y_weights = gap_weights[0][0], gap_weights[0][1]
        x_gaps, x_weights = gap_weights[1][0], gap_weights[1][1]
        if debug: LOG.debug('   all y_gaps {} x_gaps {}'.format(y_gaps, x_gaps))
        # suppress cuts that significantly split any line labels
        y_allowed = [not(np.any(np.intersect1d(
            # significant line labels above
            np.nonzero(np.bincount(lbin[:gap,:].flatten(),
                                   minlength=len(objects))[1:] > min_line * scale)[0],
            # significant line labels below
            np.nonzero(np.bincount(lbin[gap:,:].flatten(),
                                   minlength=len(objects))[1:] > min_line * scale)[0],
            assume_unique=True)))
                        for gap in y_gaps]
        x_allowed = [not(np.any(np.intersect1d(
            # significant line labels left
            np.nonzero(np.bincount(lbin[:,:gap].flatten(),
                                   minlength=len(objects))[1:] > min_line * scale)[0],
            # significant line labels right
            np.nonzero(np.bincount(lbin[:,gap:].flatten(),
                                   minlength=len(objects))[1:] > min_line * scale)[0],
            assume_unique=True)))
                        for gap in x_gaps]
        y_gaps, y_weights = y_gaps[y_allowed], y_weights[y_allowed]
        x_gaps, x_weights = x_gaps[x_allowed], x_weights[x_allowed]
        if debug: LOG.debug('   allowed y_gaps {} x_gaps {}'.format(y_gaps, x_gaps))
        y_prominence = np.amax(y_weights, initial=0)
        x_prominence = np.amax(x_weights, initial=0)
        if debug: LOG.debug('   y_prominence {} x_prominence {}'.format(y_prominence, x_prominence))
        # suppress less prominent peaks (another heuristic...)
        # they must compete with the other direction next time
        # (when already new cuts or partitions will become visible)
        y_allowed = y_weights > 0.8 * y_prominence
        x_allowed = x_weights > 0.8 * x_prominence
        y_gaps, y_weights = y_gaps[y_allowed], y_weights[y_allowed]
        x_gaps, x_weights = x_gaps[x_allowed], x_weights[x_allowed]
        if debug: LOG.debug('   prominent y_gaps {} x_gaps {}'.format(y_gaps, x_gaps))
        if npartitions > 0:
            # TODO this can be avoided when backtracking below
            # suppress peaks creating fewer partitions than others --
            # how large in our preferred direction will the new partitions
            # of sepmask in both slices created by each cut candidate
            # add up?
            y_partitionscores = [sum(map(sl.height if prefer_vertical else sl.width,
                                         morph.find_objects(morph.label(
                                             partitions[:gap,:]>0)[0]) +
                                         morph.find_objects(morph.label(
                                             partitions[gap:,:]>0)[0])))
                                 for gap in y_gaps]
            x_partitionscores = [sum(map(sl.height if prefer_vertical else sl.width,
                                         morph.find_objects(morph.label(
                                             partitions[:,:gap]>0)[0]) +
                                         morph.find_objects(morph.label(
                                             partitions[:,gap:]>0)[0])))
                                 for gap in x_gaps]
            if debug: LOG.debug('   y_partitionscores {} x_partitionscores {}'.format(
                    y_partitionscores, x_partitionscores))
            # Now identify those gaps with the largest overall score
            y_allowed = y_partitionscores == np.max(y_partitionscores, initial=0)
            x_allowed = x_partitionscores == np.max(x_partitionscores, initial=0)
            y_gaps, y_weights = y_gaps[y_allowed], y_weights[y_allowed]
            x_gaps, x_weights = x_gaps[x_allowed], x_weights[x_allowed]
            if debug: LOG.debug('   most partitioning y_gaps {} x_gaps {}'.format(y_gaps, x_gaps))
        else:
            y_partitionscores = None
            x_partitionscores = None
        # suppress less prominent peaks again, this time stricter
        y_prominence = np.amax(y_weights, initial=0)
        x_prominence = np.amax(x_weights, initial=0)
        y_allowed = y_weights > 0.9 * y_prominence
        x_allowed = x_weights > 0.9 * x_prominence
        y_gaps, y_weights = y_gaps[y_allowed], y_weights[y_allowed]
        x_gaps, x_weights = x_gaps[x_allowed], x_weights[x_allowed]
        if debug: LOG.debug('   prominent y_gaps {} x_gaps {}'.format(y_gaps, x_gaps))
        
        # decide which direction, x or y
        # TODO: this most likely needs a backtracking mechanism
        # (not just h vs v, but also all cuts at once or just some)
        # But:
        # - How to avoid combinatorial explosion?
        # - How to measure quality of different results?
        #   (e.g. log-sum of all cuts to favour longer h or longer v cuts)
        if prefer_vertical is None:
            choose_vertical = y_prominence < x_prominence
        elif prefer_vertical:
            # for text, column gaps may be arbitrarily narrow;
            # choose horizontal cut iff vertical/y profile has
            # much higher/wider gaps (or other has none)
            choose_vertical = y_prominence < 5 * x_prominence
        else:
            # for tables, column gaps may be arbitrarily wide;
            # choose vertical cut iff horizontal/x profile has
            # much higher/wider gaps (or other has none)
            choose_vertical = y_prominence * 10 < x_prominence
        if choose_vertical:
            # do vertical cuts (multiple columns)
            gaps = x_gaps
            prominence = x_prominence
            partitionscores = x_partitionscores
            lim = len(x)
        else:
            # do horizontal cuts (multiple rows)
            gaps = y_gaps
            prominence = y_prominence
            partitionscores = y_partitionscores
            lim = len(y)

        if not np.any(gaps) and npartitions == 1:
            # no slices and no partitions, but separators exist
            # so try to fall back to more elaborate partitioning
            find_topological() # partitions, npartitions, new_partition_type

        # now that we have a decision on direction (x/y)
        # as well as scores for its gaps, decide whether
        # to prefer cuts at annotated separators (partitions) instead
        prominent = 2*gap_width*scale # another heuristic...
        if (npartitions > 1 and (
                # gaps are not prominent
                prominence < prominent or
                # fewer good gaps survived than partitions
                npartitions > len(gaps)+1 or
                # partitions without the cut still score better than after
                sum(map(sl.height if prefer_vertical else sl.width,
                        filter(None, morph.find_objects(partitions)))) > np.max(
                            partitionscores, initial=0))):
            # continue on each partition by suppressing the others, respectively
            order = morph.reading_order(partitions,rl,bt)
            partitions = order[partitions]
            LOG.debug('cutting by %d partitions on %s', npartitions, box)
            if debug:
                # show current cut/box inside full image
                llab2 = relabel[llabels]
                llab2 = np.where(llab2, llab2, llabels)
                if isinstance(mask, np.ndarray):
                    llab2[box] = np.where(mask, partitions, 0)
                else:
                    llab2[box] = partitions
                DSAVE('recursive_x_y_cut_partitions', llab2)
            for label in range(1, npartitions+1):
                LOG.debug('next partition %d on %s', label, box)
                recursive_x_y_cut(box, mask=partitions==label, partition_type=new_partition_type)
            return
        
        if not np.any(gaps):
            # no gaps left
            finalize()
            return
        # otherwise: cut on gaps
        LOG.debug('cutting %s on %s into %s', 'vertically'
                  if choose_vertical else 'horizontally',
                  box, gaps)
        cuts = list(zip(np.insert(gaps, 0, 0), np.append(gaps, lim)))
        if choose_vertical:
            if rl:
                cuts = reversed(cuts)
        else:
            if bt:
                cuts = reversed(cuts)
        for start, stop in cuts:
            #box[1*choose_vertical] ... dim to cut in
            #box[1-choose_vertical] ... dim to range over
            if choose_vertical: # "cut in vertical direction"
                sub = sl.box(0, len(y), start, stop)
            else: # "cut in horizontal direction"
                sub = sl.box(start, stop, 0, len(x))
            LOG.debug('next %s block on %s is %s', 'horizontal'
                      if choose_vertical else 'vertical',
                      box, sub)
            recursive_x_y_cut(sl.compose(box,sub),
                              mask=sl.cut(mask,sub) if isinstance(mask, np.ndarray)
                              else None)
    
    # start algorithm
    recursive_x_y_cut(sl.bounds(llabels))
    
    # apply re-assignments:
    rlabels = relabel[llabels]
    DSAVE('rlabels', rlabels)
    # FIXME: hulls can overlap, we just need simplification
    #        (but cv2.approxPolyDP is faulty and morphology costly)
    # LOG.debug('closing %d regions component-wise', np.amax(relabel))
    # # close regions (label by label)
    # for region in np.unique(relabel):
    #     if not region:
    #         continue # ignore bg
    #     # faster than morphological closing:
    #     region_hull = convex_hull_image(rlabels==region)
    #     rlabels[region_hull] = region
    # DSAVE('rlabels_closed', rlabels)
    return rlabels
