from __future__ import absolute_import

import warnings
import logging

import numpy as np
from scipy.ndimage import measurements, filters, interpolation, morphology
from scipy import stats
from PIL import Image

from . import ocrolib
from .ocrolib import morph, psegutils, sl
# for decorators (type-checks etc):
from .ocrolib.toplevel import *

from ocrd_utils import getLogger

LOG = getLogger('ocrolib') # to be refined by importer

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
    if h<60/zoom: return "image not tall enough for a region image %s"%(binary.shape,)
    if h>5000/zoom: return "image too tall for a region image %s"%(binary.shape,)
    if w<100/zoom: return "image too narrow for a region image %s"%(binary.shape,)
    if w>5000/zoom: return "image too wide for a region image %s"%(binary.shape,)
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
    if h>10000/zoom: return "image too tall for a page image %s"%(binary.shape,)
    if w<600/zoom: return "image too narrow for a page image %s"%(binary.shape,)
    if w>10000/zoom: return "image too wide for a page image %s"%(binary.shape,)
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
    # set uniformly bright / maximally differentiating colors
    cmap = cm.rainbow # default viridis is too dark on low end
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
        _,fname = mkstemp(suffix=title+".png")
        plt.imsave(fname,array,vmin=vmin,vmax=vmax,cmap=cmap)
        LOG.debug('DSAVE %s', fname)

@checks(ABINARY2,NUMBER)
def compute_images(binary, scale, maximages=5):
    """Finds (and removes) large connected foreground components.
    
    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``scale``, square root of average bbox area of characters
    - ``maximages``, maximum number of large components to keep
    (This could be drop-capitals, line drawings or photos.)
    
    Returns a same-size bool array as a mask image.
    """
    d0 = odd(max(2,scale/5))
    d1 = odd(max(2,scale/8))
    images = binary
    # 1- close a little to reconnect components that have been
    #   noisily binarized
    #images = morph.rb_closing(images, (d0,d1))
    #DSAVE('images1_closed', images+0.6*binary)
    # 1- filter largest connected components
    images = morph.select_regions(images,sl.area,min=(4*scale)**2,nbest=2*maximages)
    DSAVE('images1_large', images+0.6*binary)
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
    images = np.where(images, closed, 2)
    images = morph.spread_labels(images, maxdist=scale) % 2 | closed
    DSAVE('images4_reconstructed', images+0.6*binary)
    # 5- select nbest
    images = morph.select_regions(images,sl.area,min=(4*scale)**2,nbest=maximages)
    DSAVE('images5_selected', images+0.6*binary)
    # 6- dilate a little to get a smooth contour without gaps
    dilated = morph.r_dilation(images, (odd(scale),odd(scale)))
    images = morph.propagate_labels_majority(binary, dilated+1)
    images = morph.spread_labels(images, maxdist=scale)==2
    DSAVE('images6_dilated', images+0.6*binary)
    # we could repeat reconstruct-dilate here...
    return images > 0

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
    d1 = odd(max(1,scale/8))
    d0 = odd(max(1,scale/4))
    # TODO This does not cope well with slightly sloped or heavily fragmented lines
    horiz = binary
    # 1- close horizontally a little to make warped or
    #    noisily binarized horizontal lines survive:
    horiz = morph.rb_closing(horiz, (d0,d1))
    DSAVE('hlines1_h-closed', horiz+0.6*binary)
    # 2- open horizontally to remove everything
    #    that is horizontally non-contiguous:
    opened = morph.rb_opening(horiz, (1,hlminwidth*scale))
    DSAVE('hlines2_h-opened', opened+0.6*binary)
    # 3- reconstruct the losses up to a certain distance
    #    to avoid creeping into overlapping glyphs but still
    #    cover most of the line even if not perfectly horizontal
    # (it would be fantastic if we could calculate the
    #  distance transform with stronger horizontal weights)
    horiz = np.where(horiz, opened, 2)
    horiz = morph.spread_labels(horiz, maxdist=d1) % 2 | opened
    DSAVE('hlines3_reconstructed', horiz+0.6*binary)
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
    horiz = morph.r_dilation(horiz, (d0,d1))
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
    opened = morph.rb_opening(vert, (csminheight*scale,1))
    DSAVE('colseps2_v-opened', opened+0.6*binary)
    # 3- reconstruct the losses up to a certain distance
    #    to avoid creeping into overlapping glyphs but still
    #    cover most of the line even if not perfectly vertical
    # (it would be fantastic if we could calculate the
    #  distance transform with stronger vertical weights)
    vert = np.where(vert, opened, 2)
    vert = morph.spread_labels(vert, maxdist=d1) % 2 | opened
    DSAVE('colseps3_reconstructed', vert+0.6*binary)
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
    vert = morph.r_dilation(vert, (d0,d1))
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
    grad = (grad>0.5*np.amax(grad))
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
def hmerge_line_seeds(binary, seeds, scale, threshold=0.8):
    """Relabel line seeds such that regions of coherent vertical
    intervals get the same label, and join them morphologically."""
    # merge labels horizontally to avoid splitting lines at long whitespace
    # (to prevent corners from becoming the largest label when spreading
    #  into the background; and make contiguous contours possible), but
    # ignore conflicts which affect only small fractions of either line
    # (avoiding merges for small vertical overlap):
    relabel = np.unique(seeds)
    labels = relabel[relabel > 0] # without background
    objects = [(0,0)] + measurements.find_objects(seeds)
    centers = [(0,0)] + measurements.center_of_mass(binary, seeds, labels)
    for label in labels:
        seed = seeds == label
        DSAVE('hmerge1_seed', seed)
        # close to fill holes from underestimated scale
        seed = morph.rb_closing(seed, (scale, scale))
        DSAVE('hmerge2_closed', seed)
        # open horizontally to remove extruding ascenders/descenders
        seed = morph.rb_opening(seed, (1, 3*scale))
        DSAVE('hmerge3_h-opened', seed)
        # close horizontally to overlap with possible neighbors
        seed = morph.rb_closing(seed, (1, 2*seeds.shape[1]))
        DSAVE('hmerge4_h-closed', seed)
        # get overlaps
        neighbors, counts = np.unique(seeds * seed, return_counts=True)
        for candidate, count in zip(neighbors, counts):
            if candidate in [0, label]:
                continue
            total = np.count_nonzero(seeds == candidate)
            if count < threshold * total:
                LOG.debug('ignoring h-overlap between %d and %d (only %d of %d)', label, candidate, count, total)
                continue
            label_center = centers[label]
            label_box = objects[label]
            candidate_center = centers[candidate]
            candidate_box = objects[candidate]
            if not (candidate_box[0].start < label_center[0] < candidate_box[0].stop):
                LOG.debug('ignoring h-overlap between %d and %d (y center not within other)', label, candidate)
                continue
            if not (label_box[0].start < candidate_center[0] < label_box[0].stop):
                LOG.debug('ignoring h-overlap between %d and %d (does not contain other y center)', label, candidate)
                continue
            if (candidate_box[1].start < label_center[1] < candidate_box[1].stop):
                LOG.debug('ignoring h-overlap between %d and %d (x center within other)', label, candidate)
                continue
            if (label_box[1].start < candidate_center[1] < label_box[1].stop):
                LOG.debug('ignoring h-overlap between %d and %d (contains other x center)', label, candidate)
                continue
            LOG.debug('hmerging %d with %d', candidate, label)
            # the new label could have been relabelled already:
            new_label = relabel[label]
            # assign candidate to (new assignment for) label:
            relabel[candidate] = new_label
            # re-assign labels already relabelled to candidate:
            relabel[relabel == candidate] = new_label
            # fill the horizontal background between both regions:
            candidate_y, candidate_x = np.where(seeds == candidate)
            new_label_y, new_label_x = np.where(seeds == new_label)
            for y in np.intersect1d(candidate_y, new_label_y):
                can_x_min = candidate_x[candidate_y == y][0]
                can_x_max = candidate_x[candidate_y == y][-1]
                new_x_min = new_label_x[new_label_y == y][0]
                new_x_max = new_label_x[new_label_y == y][-1]
                if can_x_max < new_x_min:
                    seeds[y, can_x_max:new_x_min] = new_label
                if new_x_max < can_x_min:
                    seeds[y, new_x_max:can_x_min] = new_label
    # apply re-assignments:
    seeds = relabel[seeds]
    DSAVE("hmerge5_connected", seeds)
    return seeds
        
# from ocropus-gpageseg, but:
# - with fullpage switch
#   (opt-in for h/v-line and column detection),
# - with external separator mask
#   (opt-in for h/v-line pass-through)
# - with zoom parameter
#   (make fixed dimension params relative to pixel density,
#    instead of blind 300 DPI assumption)
# - with improved h/v-line and column detection
# - with v-line detection _before_ column detection
# - with h/v-line suppression _after_ large component filtering
# - with more robust line seed estimation,
# - with horizontal merge instead of blur,
# - with component majority for foreground
#   outside of seeds (instead of spread),
#   except for components with seed conflict
#   (which must be split anyway)
# - with tighter polygonal spread around foreground
# - with spread of line labels against separator labels
# - return bg line and sep labels intead of just fg line labels
@checks(ABINARY2)
def compute_segmentation(binary,
                         zoom=1.0,
                         fullpage=False,
                         seps=None,
                         maxcolseps=2,
                         maxseps=0,
                         maximages=0,
                         csminheight=4,
                         hlminwidth=10,
                         spread_dist=None,
                         check=True):
    """Find text line segmentation within a region or page.

    Given a binarized (and inverted) image as Numpy array ``image``, compute
    a complete segmentation of it into text lines as a label array.

    If ``fullpage`` is false, then avoid single-line horizontal splits.

    If ``fullpage`` is true, then
    - allow all horizontal splits, and search
    - for up to ``maxcolseps`` multi-line vertical whitespaces
      (as column separators, counted piece-wise) of at least
      ``csminheight`` multiples of ``scale``,
    - for up to ``maxseps`` vertical black lines
      (as column separators, counted piece-wise) of at least
      ``csminheight`` multiples of ``scale``, and
    - for any number of horizontal lines of at least
      ``hlminwidth`` multiples of ``scale``,
    - for anything in ``seps`` if given,
    then suppress these separator components and return them separately.
    
    Labels will be projected ("spread") from the foreground to the
    surrounding background within ``spread_dist`` distance (or half
    the estimated scale).

    Return a tuple of:
    - Numpy array of the textline background labels
      (not the foreground or the masked image;
       foreground may remain unlabelled for
       separators and other non-text like small
       noise, or large drop-capitals / images),
    - Numpy array of horizontal foreground lines mask,
    - Numpy array of vertical foreground lines mask,
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
        LOG.debug('computing images')
        images = compute_images(binary, scale, maximages=maximages)
        LOG.debug('computing horizontal/vertical line separators')
        hlines = compute_hlines(binary, scale, hlminwidth=hlminwidth, images=images)
        vlines = compute_separators_morph(binary, scale, csminheight=csminheight, maxseps=maxseps, images=images)
        binary = np.minimum(binary,1-hlines)
        binary = np.minimum(binary,1-vlines)
        binary = np.minimum(binary,1-images)
        if seps is not None:
            # suppress separators/images for line estimation
            binary = (1-seps) * binary
    else:
        hlines = np.zeros_like(binary, np.bool)
        vlines = np.zeros_like(binary, np.bool)
        images = np.zeros_like(binary, np.bool)

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
        sepmask = np.maximum(hlines, vlines)
        sepmask = np.maximum(sepmask, images)
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
        seeds = hmerge_line_seeds(binary, seeds, scale)
    
    LOG.debug('spreading seed labels')
    # spread labels from seeds to bg, but watch fg,
    # voting for majority on bg conflicts,
    # but splitting on seed conflicts
    llabels = morph.propagate_labels_majority(binary, seeds)
    llabels2 = morph.propagate_labels(binary, seeds, conflict=0)
    conflicts = llabels > llabels2
    llabels = np.where(conflicts, seeds, llabels)
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
    DSAVE('llabels', llabels + 0.6*binary)
    #segmentation = llabels*binary
    #return segmentation
    return llabels, hlines, vlines, images, colseps, scale

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
def lines2regions(binary, line_labels,
                  hlines=None, vlines=None,
                  colseps=None, scale=None,
                  zoom=1.0):
    """Aggregate text lines to text regions.

    Parameters:
    - ``binary``, a bool or int array of the page image, with 1=black
    - ``line_labels``, a segmentation of the page into adjacent textlines
    - (optionally) ``hlines``, a mask array of horizontal line fg seps
    - (optionally) ``vlines``, a mask array of vertical line fg seps
    - (optionally) ``colseps``, a mask array of vertical bg seps
    - (optionally) ``scale``, square root of the average bbox area of characters

    Combine (vertically) adjacent lines by morphologically
    closing them (vertically) based on average line height.

    Return a Numpy array of text region labels.
    """
    # FIXME Needs some top-down knowledge (X-Y cut?)
    if not scale:
        #scale = psegutils.estimate_scale(binary, zoom)
        objects = morph.find_objects(line_labels)
        heights = np.array(list(map(sl.height, filter(None, objects))))
        scale = int(np.median(heights)/4)
    LOG.debug('combining lines to regions')
    label_mask = np.array(line_labels > 0, dtype=np.bool)
    label_mask = np.pad(label_mask, scale) # protect edges
    label_mask = morph.rb_closing(label_mask, (scale, 1))
    label_mask = label_mask[scale:-scale, scale:-scale] # unprotect
    DSAVE('regions1_v-closed', label_mask)
    label_mask = morph.rb_opening(label_mask, (1, scale/2))
    DSAVE('regions2_h-opened', label_mask)
    # we need some form of component-wise closing here
    # extend margins (to ensure simplified hull polygon is outside children)
    label_mask = morph.r_dilation(label_mask, (3,3)) # 1px in each direction
    DSAVE('regions3_enlarged1px', label_mask)
    # split at boundaries/separators
    if not hlines is None:
        label_mask = label_mask * (hlines==0)
    if not vlines is None:
        label_mask = label_mask * (vlines==0)
    if not colseps is None:
        label_mask = label_mask * (colseps==0)
    DSAVE('regions4_split', label_mask)
    # identify
    region_labels, _ = morph.label(label_mask.astype(np.bool))
    DSAVE('regions5_labelled', region_labels)
    return region_labels
