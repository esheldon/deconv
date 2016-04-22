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
            sigma for weight in real space, will be 1/sigma in k space.
            Note the k space image may not be sampled with step 1,
            so we multiply by the scale of the gs_kimage
        **kw:
            keywords for KSigmaWeight
        """

        self.sigma_weight=sigma_weight
        self._set_image(gs_kimage)
        self._set_weight(**kw)

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


    def _set_weight(self, **kw):
        """
        the weight in k space
        """
        self.dk = self.gs_kimage.scale
        kwt = KSigmaWeight(self.sigma_weight*self.dk, **kw)

        weight,rows,cols = kwt.get_weight(self.dims, self.cen)
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
