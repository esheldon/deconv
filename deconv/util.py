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

