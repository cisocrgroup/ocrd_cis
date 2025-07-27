################################################################
### various add-ons to the SciPy morphology package
################################################################

from numpy import *
#from scipy.ndimage import morphology,measurements,filters
from scipy.ndimage import measurements
from scipy.ndimage.interpolation import shift
import cv2
from .toplevel import *
from . import sl

@checks(ABINARY2)
def label(image,**kw):
    """Implement the scipy.ndimage.measurements.label function
    via much faster OpenCV.connectedComponents.
    
    Return a tuple:
    - same-size Numpy array with integer labels for fg components
    - number of components (eq. largest label)
    """
    # default connectivity in OpenCV: 8 (which is equivalent to...)
    # default connectivity in scikit-image: 2
    # connectivity=4 crashes (segfaults) OpenCV#21366
    n, labels = cv2.connectedComponents(image.astype(uint8))
    #n, labels = cv2.connectedComponentsWithAlgorithm(image.astype(uint8), connectivity=4, ltype=2, ccltype=cv2.CCL_DEFAULT)
    return labels, n-1
    # try: return measurements.label(image,**kw)
    # except: pass
    # types = ["int32","uint32","int64","uint64","int16","uint16"]
    # for t in types:
    #     try: return measurements.label(array(image,dtype=t),**kw)
    #     except: pass
    # # let it raise the same exception as before
    # return measurements.label(image,**kw)

@checks(SEGMENTATION)
def find_objects(image, **kw):
    """Redefine the scipy.ndimage.measurements.find_objects function to
    work with a wider range of data types.  The default function
    is inconsistent about the data types it accepts on different
    platforms.
    
    Return a list of slice tuples for each label (except 0/bg),
    or None for missing labels between 0 and max_label+1.
    """
    # This OpenCV based approach is MUCH slower:
    # objects = list()
    # for label in range(max_label+1 if max_label else amax(image)):
    #     mask = array(image==(label+1), uint8)
    #     if mask.any():
    #         x, y, w, h = cv2.boundingRect(mask)
    #         objects.append(sl.box(y,y+h,x,x+w))
    #     else:
    #         objects.append(None)
    # return objects
    try: return measurements.find_objects(image,**kw)
    except: pass
    types = ["int32","uint32","int64","uint64","int16","uint16"]
    for t in types:
        try: return measurements.find_objects(array(image,dtype=t),**kw)
        except: pass
    # let it raise the same exception as before
    return measurements.find_objects(image,**kw)

def check_binary(image):
    assert image.dtype=='B' or image.dtype=='i' or image.dtype==dtype('bool'),\
        "array should be binary, is %s %s"%(image.dtype,image.shape)
    assert amin(image)>=0 and amax(image)<=1,\
        "array should be binary, has values %g to %g"%(amin(image),amax(image))

@checks(uintpair)
def brick(size):
    return ones(size, uint8)

@checks(ABINARY2,uintpair)
def r_dilation(image,size,origin=0):
    """Dilation with rectangular structuring element using fast OpenCV.dilate."""
    return cv2.dilate(image.astype(uint8), brick(size))
    # return filters.maximum_filter(image,size,origin=(size[0]%2-1,size[1]%2-1))

@checks(ABINARY2,uintpair)
def r_erosion(image,size,origin=-1):
    """Erosion with rectangular structuring element using fast OpenCV.erode."""
    return cv2.erode(image.astype(uint8), brick(size))
    # return filters.minimum_filter(image,size,origin=0, mode='constant', cval=1)

@checks(ABINARY2,uintpair)
def r_opening(image,size,origin=0):
    """Opening with rectangular structuring element using fast OpenCV.morphologyEx."""
    return cv2.morphologyEx(image.astype(uint8), cv2.MORPH_OPEN, brick(size))
    # image = r_erosion(image,size,origin=0)
    # return r_dilation(image,size,origin=-1)

@checks(ABINARY2,uintpair)
def r_closing(image,size,origin=0):
    """Closing with rectangular structuring element using fast OpenCV.morphologyEx."""
    return cv2.morphologyEx(image.astype(uint8), cv2.MORPH_CLOSE, brick(size))
    # image = r_dilation(image,size,origin=0)
    # return r_erosion(image,size,origin=-1)

@checks(ABINARY2,uintpair)
def rb_dilation(image,size,origin=0):
    """Binary dilation using linear filters."""
    return cv2.dilate(image.astype(uint8), brick(size))
    # output = zeros(image.shape,'f')
    # filters.uniform_filter(image,size,output=output,origin=(size[0]%2-1,size[1]%2-1))
    # # 0 creates rounding artifacts
    # return array(output>1e-7,'i')

@checks(ABINARY2,uintpair)
def rb_erosion(image,size,origin=-1):
    """Binary erosion using linear filters."""
    return cv2.erode(image.astype(uint8), brick(size))
    # output = zeros(image.shape,'f')
    # filters.uniform_filter(image,size,output=output,origin=0, mode='constant', cval=1)
    # return array(output==1,'i')

@checks(ABINARY2,uintpair)
def rb_opening(image,size,origin=0):
    """Binary opening using linear filters."""
    return cv2.morphologyEx(image.astype(uint8), cv2.MORPH_OPEN, brick(size))
    # image = rb_erosion(image,size,origin=0)
    # return rb_dilation(image,size,origin=-1)

@checks(ABINARY2,uintpair)
def rb_closing(image,size,origin=0):
    """Binary closing using linear filters."""
    return cv2.morphologyEx(image.astype(uint8), cv2.MORPH_CLOSE, brick(size))
    # image = rb_dilation(image,size,origin=0)
    # return rb_erosion(image,size,origin=-1)

@checks(ABINARY2,ABINARY2)
def rb_reconstruction(image,mask,step=1,maxsteps=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2*step+1,2*step+1))
    dilated = image.astype(uint8)
    while maxsteps is None or maxsteps > 0:
        dilated = cv2.dilate(src=dilated, kernel=kernel)
        cv2.bitwise_and(src1=dilated, src2=mask.astype(uint8), dst=dilated)
        # did result change?
        if (image == dilated).all():
            return dilated
        if maxsteps:
            maxsteps -= step
    return dilated
    
@checks(GRAYSCALE,uintpair)
def rg_dilation(image,size,origin=0):
    """Grayscale dilation using fast OpenCV.dilate."""
    return cv2.dilate(image, brick(size))
    # return filters.maximum_filter(image,size,origin=origin)

@checks(GRAYSCALE,uintpair)
def rg_erosion(image,size,origin=0):
    """Grayscale erosion using fast OpenCV.erode."""
    return cv2.erode(image, brick(size))
    # return filters.minimum_filter(image,size,origin=origin, mode='constant', cval=1)

@checks(GRAYSCALE,uintpair)
def rg_opening(image,size,origin=0):
    """Grayscale opening using fast OpenCV.morphologyEx."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, brick(size))
    # image = r_erosion(image,size,origin=origin)
    # return r_dilation(image,size,origin=origin)

@checks(GRAYSCALE,uintpair)
def rg_closing(image,size,origin=0):
    """Grayscale closing using fast OpenCV.morphologyEx."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, brick(size))
    # image = r_dilation(image,size,origin=0)
    # return r_erosion(image,size,origin=-1)

@checks(GRAYSCALE,ABINARY2)
def rg_reconstruction(image,mask,step=1,maxsteps=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*step+1,2*step+1))
    dilated = image
    while maxsteps is None or maxsteps > 0:
        dilated = cv2.dilate(src=dilated, kernel=kernel)
        dilated = np.where(mask, dilated, image)
        # did result change?
        if (image == dilated).all():
            return dilated
        if maxsteps:
            maxsteps -= step
    return dilated

@checks(SEGMENTATION)
def showlabels(x,n=7):
    import matplotlib.pyplot as plt
    plt.imshow(where(x>0,x%n+1,0),cmap=plt.cm.gist_stern)

@checks(ABINARY2)
def find_contours(image):
    """Find hull polygons around connected fg components.
    
    Return a list of contours, each a tuple of:
    - coordinates (as a list of y,x tuples)
    - area
    """
    # skimage.measure.find_contours is not only slow but impractical:
    # - interrupts hull polygon when it intersects the margins (!)
    # - uses 0.5-based coordinates (i.e. center of pixel instead of top/left)
    contours, _ = cv2.findContours(image.astype(uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # convert to y,x tuples
    return [(contour[:,0,::-1], cv2.contourArea(contour))
            for contour in contours]

@checks(SEGMENTATION)
def find_label_contours(labels):
    contours = [[]]*(amax(labels)+1)
    for label in unique(labels):
        if not label:
            continue
        contours[label] = find_contours(labels==label)
    return contours

@checks(ALL(SEGMENTATION,ANONNEG))
def spread_labels(labels,maxdist=9999999):
    """Spread the given labels to the background."""
    #distances,features = morphology.distance_transform_edt(labels==0,return_distances=1,return_indices=1)
    #indexes = features[0]*labels.shape[1]+features[1]
    #spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    if not labels.any():
        return labels
    distances,indexes = cv2.distanceTransformWithLabels(array(labels==0,uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE,labelType=cv2.DIST_LABEL_PIXEL)
    spread = labels[where(labels>0)][indexes-1]
    if maxdist is None:
        return spread, distances
    spread *= (distances<maxdist)
    return spread

@checks(SEGMENTATION)
def dist_labels(labels):
    """Get the distance transformation of the segments."""
    if not labels.any():
        return labels
    return cv2.distanceTransform(labels,
                                 distanceType=cv2.DIST_L1,
                                 maskSize=3,
                                 dstType=cv2.CV_8U)

@checks(ABINARY2,ABINARY2)
def keep_marked(image,markers):
    """Given a marker image, keep only the connected components
    that overlap the markers."""
    labels,_ = label(image)
    marked = unique(labels*(markers!=0))
    kept = in1d(labels.ravel(),marked)
    return (image!=0)*kept.reshape(*labels.shape)

@checks(ABINARY2,ABINARY2)
def remove_marked(image,markers):
    """Given a marker image, remove all the connected components
    that overlap markers."""
    marked = keep_marked(image,markers)
    return image*(marked==0)

@checks(SEGMENTATION,SEGMENTATION)
def correspondences(labels1,labels2,return_counts=True):
    """Given two labeled images, compute an array giving the correspondences
    between labels in the two images (as tuples of label in `labels1`,
    label in `labels2`, and pixel count)."""
    q = 100000
    assert amin(labels1)>=0 and amin(labels2)>=0
    assert amax(labels2)<q
    combo = labels1*q+labels2
    result = unique(combo, return_counts=return_counts)
    if return_counts:
        result, counts = result
        result = array([result//q,result%q,counts])
    else:
        result = array([result//q,result%q])
    return result

@checks(ABINARY2,SEGMENTATION)
def propagate_labels_simple(regions,labels):
    """Given an image and a set of labels, apply the labels
    to all the connected components in the image that overlap a label."""
    rlabels,_ = label(regions)
    cors = correspondences(rlabels,labels,False)
    outputs = zeros(amax(rlabels)+1,'i')
    for o,i in cors.T: outputs[o] = i
    outputs[0] = 0
    return outputs[rlabels]

@checks(ABINARY2,SEGMENTATION)
def propagate_labels_majority(image,labels):
    """Given an image and a set of labels, apply the labels
    to all the connected components in the image that overlap a label.
    For each component that has a conflict, select the label
    with the largest overlap."""
    rlabels,_ = label(image)
    cors = correspondences(rlabels,labels)
    outputs = zeros(amax(rlabels)+1,'i')
    counts = zeros(amax(rlabels)+1,'i')
    for rlabel, label_, count in cors.T:
        if not rlabel or not label_:
            # ignore background correspondences
            continue
        if counts[rlabel] < count:
            outputs[rlabel] = label_
            counts[rlabel] = count
    outputs[0] = 0
    return outputs[rlabels]

@checks(ABINARY2,SEGMENTATION)
def propagate_labels(image,labels,conflict=0):
    """Given an image and a set of labels, apply the labels
    to all the connected components in the image that overlap a label.
    Assign the value `conflict` to any components that have a conflict."""
    rlabels,_ = label(image)
    cors = correspondences(rlabels,labels,False)
    outputs = zeros(amax(rlabels)+1,'i')
    oops = -(1<<30)
    for o,i in cors.T:
        if outputs[o]!=0: outputs[o] = oops
        else: outputs[o] = i
    outputs[outputs==oops] = conflict
    outputs[0] = 0
    return outputs[rlabels]

@checks(ANY(ABINARY2,SEGMENTATION),True)
def select_regions(binary,f,min=0,nbest=100000):
    """Given a scoring function f over slice tuples (as returned by
    find_objects), keeps at most nbest components whose scores is higher
    than min."""
    if binary.max() == 1:
        labels,_ = label(binary)
    else:
        labels = binary.astype(uint8)
    objects = find_objects(labels)
    scores = [f(o) for o in objects]
    best = argsort(scores)
    keep = zeros(len(objects)+1,'i')
    if nbest > 0:
        for i in best[-nbest:]:
            if scores[i]<=min: continue
            keep[i+1] = 1
    # print scores,best[-nbest:],keep
    # print sorted(list(set(labels.ravel())))
    # print sorted(list(set(keep[labels].ravel())))
    return keep[labels]

@checks(SEGMENTATION)
def all_neighbors(image, dist=1, bg=float('nan')):
    """Given an image with labels, find all pairs of labels
    that are directly (up to ``dist``) neighboring each other, ignoring the label ``bg``."""
    q = 100000
    assert amax(image)<q
    assert amin(image)>=0
    u = unique(q*image+shift(image,(dist,0),order=0,cval=bg))
    d = unique(q*image+shift(image,(-dist,0),order=0,cval=bg))
    l = unique(q*image+shift(image,(0,dist),order=0,cval=bg))
    r = unique(q*image+shift(image,(0,-dist),order=0,cval=bg))
    all = unique(r_[u,d,l,r])
    all = all[all!=bg]
    all = c_[all//q,all%q]
    all = unique(array([sorted(x) for x in all]), axis=0)
    return all

################################################################
### Iterate through the regions of a color image.
################################################################

@checks(SEGMENTATION)
def renumber_labels_ordered(a,correspondence=0):
    """Renumber the labels of the input array contiguously so
    that they range from 1...N"""
    assert amin(a)>=0
    assert amax(a)<=2**25
    labels = sorted(unique(ravel(a)))
    renum = zeros(amax(labels)+1,dtype='i')
    renum[labels] = arange(len(labels),dtype='i')
    if correspondence:
        return renum[a],labels
    else:
        return renum[a]

@checks(SEGMENTATION)
def renumber_labels(a):
    """Alias for renumber_labels_ordered"""
    return renumber_labels_ordered(a)

def pyargsort(seq,cmp=None,key=lambda x:x):
    """Like numpy's argsort, but using the builtin Python sorting
    function.  Takes an optional cmp."""
    return sorted(list(range(len(seq))),key=lambda x:key(seq.__getitem__(x)),cmp=None)

@checks(SEGMENTATION)
def renumber_by_xcenter(seg):
    """Given a segmentation (as a color image), change the labels
    assigned to each region such that when the labels are considered
    in ascending sequence, the x-centers of their bounding boxes
    are non-decreasing.  This is used for sorting the components
    of a segmented text line into left-to-right reading order."""
    objects = [(slice(0,0),slice(0,0))]+find_objects(seg)
    def xc(o):
        # if some labels of the segmentation are missing, we
        # return a very large xcenter, which will move them all
        # the way to the right (they don't show up in the final
        # segmentation anyway)
        if o is None: return 999999
        return mean((o[1].start,o[1].stop))
    xs = array([xc(o) for o in objects])
    order = argsort(xs)
    segmap = zeros(amax(seg)+1,'i')
    for i,j in enumerate(order): segmap[j] = i
    return segmap[seg]

@checks(SEGMENTATION)
def reading_order(seg,rl=False,bt=False):
    """Compute a new order for labeled objects based on y and x centers.
    
    ``seg`` may have discontinuous labels.
    ``rl`` whether to sort from right to left within a line
    ``bt`` whether to sort from bottom to top across lines
    
    First, sort by ycenter, then group by mutual ycenter_in.
    Second, sort groups by xcenter, then concatenate all groups.
    
    Return a map for the labels which will put them in reading order.
    """
    # TODO can be done better
    segmap = zeros(amax(seg)+1,'i')
    objects = [(slice(0,0),slice(0,0))]+find_objects(seg)
    if len(objects) <= 2:
        # nothing to do
        segmap[1:] = 1
        return segmap
    def pos(f,l):
        return array([f(x) if x else float('nan') for x in l])
    ys = pos(sl.ycenter,objects)
    yorder = argsort(ys)[::-1 if bt else 1]
    groups = [[yorder[0]]]
    for i,j in zip(yorder[:-1],yorder[1:]):
        oi = objects[i]
        oj = objects[j]
        if (oi and oj and
            sl.yoverlaps(oi,oj) and
            (sl.ycenter_in(oi,oj) or
             sl.ycenter_in(oj,oi)) and
            not any([sl.xoverlaps(oj,objects[k]) and sl.xoverlap_rel(oj,objects[k]) > 0.1
                     for k in groups[-1]])):
            groups[-1].append(j)
        else:
            groups.append([j])
    rorder = list()
    for group in groups:
        group = array(group)
        xs = pos(sl.xcenter,[objects[i] for i in group])
        xorder = argsort(xs)[::-1 if rl else 1]
        rorder.extend(group[xorder])
    for i,j in enumerate(rorder):
        segmap[j] = i
    return segmap

@checks(SEGMENTATION)
def ordered_by_xcenter(seg):
    """Verify that the labels of a segmentation are ordered
    spatially (as determined by the x-center of their bounding
    boxes) in left-to-right reading order."""
    objects = [(slice(0,0),slice(0,0))]+find_objects(seg)
    def xc(o): return mean((o[1].start,o[1].stop))
    xs = array([xc(o) for o in objects])
    for i in range(1,len(xs)):
        if xs[i-1]>xs[i]: return 0
    return 1
