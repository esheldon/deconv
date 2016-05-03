from __future__ import print_function
import numpy
from .weight import KSigmaWeight
from .deconv import DeConvolver

from . import util


LOW_T=2**0

def calcmom_ksigma(gal_image, psf_image, sigma_weight, **kw):
    """
    Deconvolve the image and measure the moments using
    a sigmak type weight

    parameters
    ----------
    gal_image: Galsim image
        Galsim Image in real space
    psf_image: Galsim image
        Galsim Image in real space
    sigma_weight: float
        sigma for weight in real space, will be 1/sigma in k space.
        If scale if image is not unity, the sigma will be
        modified to be set to sigma*scale

    returns
    -------
    A Moments object.  Use get_result() to get the result dict
    """

    # deconvolve the psf
    deconv=DeConvolver(gal_image, psf_image)
    gs_kimage = deconv.get_kimage()

    # get the weight function
    dk=gs_kimage.scale
    kwt = KSigmaWeight(sigma_weight*dk, **kw)

    dims=gs_kimage.array.shape
    cen=util.get_canonical_kcenter(dims)
    kweight, rows, cols = kwt.get_weight(dims, cen)

    # calculate moments
    meas=Moments(
        gs_kimage,
        kweight,
    )
    meas.go()

    return meas

def calcmom_obs(obs):
    """
    Deconvolve the image and measure the moments for the ngmix observation
    """
    pass

class Moments(object):
    """
    Deconvolve the psf and image and measure moments
    """
    def __init__(self, gs_kimage, kweight, **kw):
        """
        parameters
        kimage: GS image
            Galsim Image in k space, deconvolved from the psf
        kweight: array
            A weight image in k space

        cen: 2-element sequence, optional
            Optional input center. Default is canonical galsim
            k space center

        rows: array, optional
            Optional input rows. If not sent, will be calculated from
            the image center.
        cols: array, optional
            Optional input cols. If not sent, will be calculated from
            the image center.
        """

        self._set_image(gs_kimage, **kw)
        self._set_weight(kweight, **kw)

    def get_result(self):
        """
        get the result dictionary
        """
        if not hasattr(self,'result'):
            raise RuntimeError("run go() first")
        return self.result

    def go(self):
        """
        measure weighted moments in k space
        """

        rows,cols=self.rows,self.cols
        kimage=self.kimage

        wim = self.kimage * self.kweight

        imsum = wim.sum()

        irrsum_k = (rows**2*wim).sum()
        ircsum_k = (rows*cols*wim).sum()
        iccsum_k = (cols**2*wim).sum()

        self._set_result(irrsum_k, ircsum_k, iccsum_k, imsum)

    def _set_result(self, irrsum_k, ircsum_k, iccsum_k, imsum):

        irr_k=irrsum_k/imsum
        irc_k=ircsum_k/imsum
        icc_k=iccsum_k/imsum

        dk=self.dk

        T_k = irr_k + icc_k

        flags=0
        e1=-9999.0
        e2=-9999.0
        T=-9999.0

        if T_k > 0:

            T=1.0/T_k
            T /= dk**2

            e1 = -(icc_k - irr_k)/T_k
            e2 = -2.0*irc_k/T_k
        else:
            flags=LOW_T

        self.result={
            'flags':flags,

            'e1':e1,
            'e2':e2,
            'T':T, # real space T

            'irr_k':irr_k,
            'irc_k':irc_k,
            'icc_k':icc_k,
            'T_k':T_k,

            'irrsum_k':irrsum_k,
            'ircsum_k':ircsum_k,
            'iccsum_k':iccsum_k,
            'imsum':imsum,

            'dk':dk,
        }


    def _set_image(self, gs_kimage, **kw):
        """
        set the image and center, do checks
        """
        self.gs_kimage=gs_kimage

        self.dk = self.gs_kimage.scale
        self.kimage = gs_kimage.array

        self._check_image(self.kimage)

        self.dims=self.kimage.shape

        if 'cen' in kw:
            self.cen=kw['cen']
        else:
            self.cen=util.get_canonical_kcenter(self.dims)

        self._set_rows_cols(**kw)

    def _set_rows_cols(self, **kw):

        if 'rows' in kw:
            self.rows=kw['rows']
            self.cols=kw['cols']
        else:
            dims=self.kimage.shape
            cen=self.cen

            rows,cols=numpy.mgrid[
                0:dims[0],
                0:dims[1],
            ]

            rows=numpy.array(rows, dtype='f8')
            cols=numpy.array(cols, dtype='f8')

            rows -= cen[0]
            cols -= cen[1]

            self.rows=rows
            self.cols=cols


    def _set_weight(self, kweight, **kw):
        self.kweight = kweight

    '''
    def _set_weight(self, **kw):
        """
        the weight in k space
        """
        kwt = KSigmaWeight(self.sigma_weight*self.dk, **kw)

        weight,rows,cols = kwt.get_weight(self.dims, self.cen)
        self.kweight=weight
        self.rows=rows
        self.cols=cols
    '''

    def _check_image(self, im):
        """
        so default center is right
        """
        assert (im.shape[0] % 2)==0
        assert (im.shape[1] % 2)==0

