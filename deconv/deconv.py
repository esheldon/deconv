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

    def get_kimage(self, shear=None, dk=None, nx=None, ny=None):
        """
        get the real part of the k-space image of the deconvolved galaxy

        returns
        -------
        kreal: Galsim Image
        """

        if dk is None:
            dk=self.dk

        if shear is not None:
            return self.get_sheared_kimage(shear, dk=dk, nx=nx, ny=ny)

        kreal,kimag = self.igal_nopsf.drawKImage(
            dtype=numpy.float64,
            nx=nx,
            ny=ny,
            scale=dk,
        )
        return kreal, kimag

    def get_sheared_kimage(self, shear, dk=None, nx=None, ny=None):
        """
        get the real part of the k-space image of the deconvolved galaxy

        parameters
        ----------
        shear: shear object
            must have .g1 and .g2 attributes

        returns
        -------
        kreal: Galsim Image
        """

        if dk is None:
            dk=self.dk

        obj = self.get_sheared_gsobj(shear)

        kreal,kimag = obj.drawKImage(
            dtype=numpy.float64,
            nx=nx,
            ny=ny,
            scale=dk,
        )
        return kreal, kimag


    def get_sheared_gsobj(self, shear):
        """
        get sheared version of the interpolated, deconvolved object

        parameters
        ----------
        shear: shear object
            must have .g1 and .g2 attributes

        returns
        --------
        galsim InterpolatedImage
        """

        obj=self.get_gsobj()

        return obj.shear(
            g1=shear.g1,
            g2=shear.g2,
        )

    def get_gsobj(self):
        """
        get the galsim interpolated image for the deconvolved object
        """
        return self.igal_nopsf

    def _set_data(self, gal_image, psf_image_orig):

        psf_image = psf_image_orig.copy()

        imsum=psf_image.array.sum()
        if imsum == 0.0:
            raise DeconvRangeError("PSF image has zero flux")

        psf_image /= imsum

        self.igal = galsim.InterpolatedImage(gal_image, x_interpolant='lanczos15')
        self.ipsf = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos15')

        self.ipsf_inv = galsim.Deconvolve(self.ipsf)

        self.igal_nopsf = galsim.Convolve(self.igal, self.ipsf_inv)


