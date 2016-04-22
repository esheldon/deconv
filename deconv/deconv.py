from __future__ import print_function

try:
    import galsim
except:
    pass

class DeConvolver(object):
    """
    deconvolve the input PSF from the input image
    """
    def __init__(self, gal_image, psf_image):
        """
        parameters
        ----------
        gal_image: galsim Image
            Galsim Image object
        psf_image: galsim Image
            Galsim Image object
        """

    def _set_data(self, gal_image, psf_image):
        self.igal = galsim.InterpolatedImage(gal_image)
        self.ipsf = galsim.InterpolatedImage(psf_image)

        self.ipsf_inv = galsim.Deconvolve(self.ipsf)

        self.igal_nopsf = galsim.Convolve(self.igal, self.ipsf_inv)

    def get_kimage(self):
        """
        get an interpolated image of the deconvolved galaxy
        """
        return self.igal_nopsf


