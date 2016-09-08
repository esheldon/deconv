from __future__ import print_function
import numpy

class DeconvRangeError(Exception):
    """
    some range problem in the data
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

class DeconvMaxiter(Exception):
    """
    some range problem in the data
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


def get_canonical_kcenter(dims):
    """

    get the galsim canonical center for an image in k space.  See
    the drawKImage method

    """
    assert (dims[0] % 2)==0
    assert (dims[1] % 2)==0

    cen=[
        int( (dims[0]-1.0)/2.0 + 0.5),
        int( (dims[1]-1.0)/2.0 + 0.5),
    ]

    return cen

def make_rows_cols(dims, cen=None):
    """
    make row/col arrays

    parameters
    ----------
    dims: 2-element sequence
        [nrow, ncol]
    cen: optional 2-element sequence
        [rowcen, colcen]
        If not sent, the canonical center is used
    """

    if cen is None:
        cen=get_canonical_kcenter(dims)

    rows,cols=numpy.mgrid[
        0:dims[0],
        0:dims[1],
    ]

    rows=numpy.array(rows, dtype='f8')
    cols=numpy.array(cols, dtype='f8')

    rows -= cen[0]
    cols -= cen[1]

    return rows, cols

def symmetrize_image(im, cen=None, doplot=False, file=None, **kw):
    """
    create an azimuthally symmetric version of the image
    """

    rows, cols = make_rows_cols(im.shape, cen=cen)

    r2 = rows**2 + cols**2

    r2rav = r2.ravel()
    imrav = im.ravel()

    if doplot:
        import pyxtools
        r=numpy.sqrt(r2rav)

        tmp = imrav/imrav.max()
        plt=pyxtools.plot(
            r,
            tmp,
            ratio=1,
            **kw
        )
        plt=pyxtools.plot(
            r, tmp**2,
            plt=plt,
            sym='triangle',
            size=0.075,
            color='red',
            file=file,
            **kw)


def get_ratio_error(a, b, var_a, var_b, cov_ab):
    """
    get a/b and error on a/b
    """
    from math import sqrt

    var = get_ratio_var(a, b, var_a, var_b, cov_ab)

    error = sqrt(var)
    return error

def get_ratio_var(a, b, var_a, var_b, cov_ab):
    """
    get (a/b)**2 and variance in mean of (a/b)
    """
    if b == 0.0:
        raise ValueError("zero in denominator")

    rsq = (a/b)**2

    #var = rsq * (  var_a/a**2 + var_b/b**2 - 2*cov_ab/(a*b) )
    var = rsq * (  var_a/a**2 + var_b/b**2 )

    return var

DEF_MIN_REL_VAL=0.01

def find_min_kmax(mb_obs, **kw):
    """
    the observations must have deconvolvers set
    """

    kmax = 1.e9
    for obslist in mb_obs:
        for obs in obslist:
            kreal = obs.deconvolver.psf_kreal
            tkmax = util.find_kmax(kreal)

            if tkmax < kmax:
                kmax = tkmax

    return kmax

def find_kmax(gsim, **kw):
    """
    get the maximum radius in k space for which the value is larger
    than the indicated value relative to the maximum

    parameters
    ----------
    gsim: galsim image
        A Galsim image instance

    min_rel_val: float, optional
        The minimum value relative to the max to consider
    """

    min_rel_val = kw.get('min_rel_val',DEF_MIN_REL_VAL)

    dk=gsim.scale

    dims=gsim.array.shape
    cen = util.get_canonical_kcenter(dims)
    rows,cols=util.make_rows_cols(
        dims,
        cen=cen,
    )

    r2 = rows**2 + cols**2

    maxval = gsim.array.max()
    minval = maxval*min_rel_val

    w=numpy.where(gsim.array > minval)
    if w[0].size == 0:
        raise DeconvRangeError("no good psf values in k space")

    return math.sqrt(r2[w].max())

