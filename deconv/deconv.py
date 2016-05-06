from __future__ import print_function
import numpy

from .util import DeconvRangeError
try:
    import galsim
except:
    pass

class DeConvolver(object):
    """
    deconvolve the input PSF from the input image
    """
    def __init__(self, gal_image, psf_image, dk=None):
        """
        parameters
        ----------
        gal_image: galsim Image
            Galsim Image object
        psf_image: galsim Image
            Galsim Image object
        """

        self.dk=dk
        self._set_data(gal_image, psf_image)

    def get_kimage(self):
        """
        get the real part of the k-space image of the deconvolved galaxy

        returns
        -------
        kreal: Galsim Image
        """

        kreal,kimag = self.igal_nopsf.drawKImage(
            dtype=numpy.float64,
            scale=self.dk,
        )
        return kreal

    def get_gsobj(self):
        """
        get the galsim object
        """
        return self.igal_nopsf

    def _set_data(self, gal_image, psf_image_orig):

        psf_image = psf_image_orig.copy()

        imsum=psf_image.array.sum()
        if imsum == 0.0:
            raise DeconvRangeError("PSF image has zero flux")

        psf_image /= imsum

        self.igal = galsim.InterpolatedImage(gal_image)
        self.ipsf = galsim.InterpolatedImage(psf_image)

        self.ipsf_inv = galsim.Deconvolve(self.ipsf)

        self.igal_nopsf = galsim.Convolve(self.igal, self.ipsf_inv)


