from __future__ import print_function

from .weight import KSigmaWeight

LOW_T=2**0

class Moments(object):
    """
    Deconvolve the psf and image and measure moments
    """
    def __init__(self, gs_kimage, sigma_weight, **kw):
        """
        parameters
        kimage: image
            Galsim Image in k space, deconvolved from the psf
        sigma_weight: float
            sigma for weight in pixels, will be 1/sigma in k space pixels.
            Note the k space image will generally be in sky coordinates
            not pixel coordinates, so adjust accordingly
        **kw:
            keywords for KSigmaWeight
        """

        self.sigma_weight=sigma_weight
        self._set_image(gs_kimage)
        self._set_weight()

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

        rows=self.rows
        cols=self.cols
        kimage=self.kimage

        wim = self.kimage * self.weight

        rows,cols=self.rows,self.cols

        imsum = wim.sum()

        irr = (rows**2*wim).sum()/imsum
        irc = (rows*cols*wim).sum()/imsum
        icc = (cols**2*wim).sum()/imsum
        self._set_result(irr, irc, icc)

    def _set_result(self, irr, irc, icc):
        T = irr + icc

        flags=0
        e1=-9999.0
        e2=-9999.0
        Treal=-9999.0
        if T > 0:
            Treal=1.0/T
            Treal /= dk**2

            e1 = -(icc - irr)/T
            e2 = -2.0*irc/T
        else:
            flags=LOW_T

        self.result={
            'flags':flags,
            'dk':self.dk,
            'irr':irr,
            'irc':irc,
            'icc':icc,
            'e1':e1,
            'e2':e2,
            'T':T,
            'Treal':Treal,
        }


    def _set_weight(self):
        """
        the weight in k space
        """
        self.dk = self.gs_kimage.stepK()
        kw = KSigmaWeight(sigma_weight*self.dk, **kw)

        weight,rows,cols = kw.get_weight(self.dims, self.cen)
        self.weight=weight
        self.rows=rows
        self.cols=cols

    def _set_image(self, gs_kimage):
        """
        set the image and center, do checks
        """
        self.gs_kimage=gs_kimage
        self.kimage = gs_kimage.array

        self._check_image(self.kimage)

        self.dims=self.kimage.shape
        self.cen=[
            int( (self.dims[0]-1.0)/2.0 + 0.5),
            int( (self.dims[1]-1.0)/2.0 + 0.5),
        ]


    def _check_image(self, im):
        """
        so default center is right
        """
        assert (im.shape[0] % 2)==0
        assert (im.shape[1] % 2)==0

