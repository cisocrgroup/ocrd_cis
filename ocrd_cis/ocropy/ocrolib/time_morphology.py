from scipy.ndimage import morphology,filters,measurements
import numpy as np
from PIL import Image
from ocrd_cis.ocropy import common
import cv2

# compares performance of different implementations for image morphology:
# (in order of runtime, slowest first)
# - SciPy morphology/measurements
# - SciPy uniform filter
# - SciPy min/max filter
# - OpenCV

def cv_opening(bin, size):
    # note: calling with *even* size dimensions will cause 1-off errors after erosion!
    return cv2.morphologyEx(bin.astype(np.uint8), cv2.MORPH_OPEN, np.ones(size, np.uint8))

def cv_closing(bin, size):
    # note: calling with *even* size dimensions will cause 1-off errors after erosion!
    return cv2.morphologyEx(bin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones(size, np.uint8))

def cv_label(bin):
    n, labels = cv2.connectedComponents(bin.astype(np.uint8))
    return labels, n

def cv_contours(bin):
    contours, _ = cv2.findContours(bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # convert to y,x tuples
    return zip((contour[:,0,::-1], cv2.contourArea(contour)) for contour in contours)

def rb_opening(bin, size):
    return filters.uniform_filter(filters.uniform_filter(bin, size, float, mode='constant', cval=1) == 1, size, float, origin=-1) > 1e-7

def rb_closing(bin, size):
    return filters.uniform_filter(filters.uniform_filter(bin, size, float) > 1e-7, size, mode='constant', cval=1, origin=-1) == 1

def r_closing(bin, size):
    return filters.minimum_filter(filters.maximum_filter(bin, size), size, origin=-1)

def r_opening(bin, size):
    return filters.maximum_filter(filters.minimum_filter(bin, size), size, origin=-1)

def nd_closing(bin, size):
    return morphology.binary_closing(bin, np.ones(size))

def nd_opening(bin, size):
    return morphology.binary_opening(bin, np.ones(size))

def nd_label(bin):
    return measurements.label(bin)

#def sk_contours ...
# skimage.measure.find_contours is impractical:
# - interrupts hull polygon when it intersects the margins (!)
# - uses 0.5-based coordinates (i.e. center of pixel instead of top/left)

def load(path):
    img = Image.open(path)
    return common.pil2array(img)==0

def test_cv(bin):
    cv_opening(cv_closing(bin, (20,10)), (3,2))

def test2_cv(bin):
    cv_label(bin)

def test3_cv(bin):
    closed = cv_closing(bin, (int(bin.shape[0]//20),int(bin.shape[1]//20)))
    labels, n = cv_label(closed)
    return labels, n

def test4_cv(bin):
    labels, n = test3_cv(bin)
    contours = list([])
    for label in range(1, n+1):
        contours.append(cv_contours(labels==label))
    return contours

def test_rb(bin):
    rb_opening(rb_closing(bin, (20,10)), (3,2))

def test_r(bin):
    r_opening(r_closing(bin, (20,10)), (3,2))

def test_nd(bin):
    nd_opening(nd_closing(bin, (20,10)), (3,2))

def test2_nd(bin):
    nd_label(bin)

if __name__ == '__main__':
    import timeit
    import sys
    path = sys.argv[1]
    print('testing open/close based on opencv') # 0.14s
    print(timeit.timeit("test_cv(bin)", setup="from __main__ import test_cv, load; bin = load('%s')" % path, number=10))
    print('testing open/close based on scipy.nd maxi/minimum_filter') # 3.2s
    print(timeit.timeit("test_r(bin)", setup="from __main__ import test_r, load; bin = load('%s')" % path, number=10))
    print('testing open/close based on scipy.nd uniform_filter') # 3.8s
    print(timeit.timeit("test_rb(bin)", setup="from __main__ import test_rb, load; bin = load('%s')" % path, number=10))
    print('testing open/close based on scipy.nd morphology') # 17.3s
    print(timeit.timeit("test_nd(bin)", setup="from __main__ import test_nd, load; bin = load('%s')" % path, number=10))
    print('testing component analysis based on opencv') # 0.5s
    print(timeit.timeit("test2_cv(bin)", setup="from __main__ import test2_cv, load; bin = load('%s')" % path, number=100))
    print('testing component analysis based on scipy.nd measurements') # 3.4s
    print(timeit.timeit("test2_nd(bin)", setup="from __main__ import test2_nd, load; bin = load('%s')" % path, number=100))
    print('testing component analysis after close based on opencv') # 8.4s
    print(timeit.timeit("test3_cv(bin)", setup="from __main__ import test3_cv, load; bin = load('%s')" % path, number=100))
    print('testing contours after CA after close based on opencv') # 12.4s
    print(timeit.timeit("test4_cv(bin)", setup="from __main__ import test4_cv, load; bin = load('%s')" % path, number=100))
