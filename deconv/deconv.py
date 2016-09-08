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

    def get_dk(self):
        return self.dk

    def get_kimage(self, shear=None, dk=None, nx=None, ny=None):
        """
        get the real part of the k-space image of the deconvolved galaxy

        returns
        -------
        kreal: Galsim Image
        """

        if shear is not None:
            return self.get_sheared_kimage(shear, dk=dk, nx=nx, ny=ny)
        else:
            return self.get_unsheared_kimage(dk=dk, nx=nx, ny=ny)

    def get_unsheared_kimage(self, dk=None, nx=None, ny=None):
        #print("sent dk:",dk)
        if dk is None:
            dk=self.dk

        #print("drawing unsheared k image with scale:",dk)
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

        #print("sent dk:",dk)
        if dk is None:
            dk=self.dk

        obj = self.get_sheared_gsobj(shear)

        #print("drawing sheared k image with scale:",dk)
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

        #self.igal = galsim.InterpolatedImage(gal_image, x_interpolant='lanczos15')
        #self.ipsf = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos15')
        self.igal = galsim.InterpolatedImage(gal_image)
        self.ipsf = galsim.InterpolatedImage(psf_image)

        self.ipsf_inv = galsim.Deconvolve(self.ipsf)

        self.igal_nopsf = galsim.Convolve(self.igal, self.ipsf_inv)


        # should we take this from igal or igal_nopsf?
        if self.dk is None:
            self.dk = self.igal.stepK()
            #self.dk = self.igal_nopsf.stepK()

class DeConvolverPrerender(DeConvolver):
    def get_sheared_kimage(self, shear, **kw):
        """
        don't shear the imaginary part
        """
        #print("getting prerender sheared")

        kreal_ii = self.get_sheared_gsobj(shear)

        nx=kw.get('nx',None)
        ny=kw.get('ny',None)

        kreal = kreal_ii.drawImage(
            dtype=numpy.float64,
            nx=nx,
            ny=ny,
            scale=self.dk,
        )
        return kreal, self.kimag

    def get_unsheared_kimage(self, **kw):
        #print("getting prerender unsheared")

        nx=kw.get('nx',None)
        if nx is not None:
            ny=kw.get('ny',None)

            kreal,kimag = self.igal_nopsf.drawKImage(
                dtype=numpy.float64,
                nx=nx,
                ny=ny,
                scale=self.dk,
            )
            return kreal, kimag
        else:
            return self.kreal.copy(), self.kimag.copy()

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


        return self.kreal_ii.shear(
            g1=shear.g1,
            g2=shear.g2,
        )


    def _set_data(self, *args, **kw):
        super(DeConvolverPrerender,self)._set_data(*args, **kw)

        self.kreal,self.kimag = self.igal_nopsf.drawKImage(
            dtype=numpy.float64,
            scale=self.dk,
        )
        self.kreal_ii=galsim.InterpolatedImage(
            self.kreal,
            scale=self.dk,
        )


DEFAULT_INTERP='quintic'

class DeConvolverPSFBase(DeConvolver):
    """
    get info about dk, etc. from the psf
    """

    def __init__(self, gal_image, psf_image, **kw):
        """
        parameters
        ----------
        gal_image: galsim Image
            Galsim Image object
        psf_image: galsim Image
            Galsim Image object
        """

        self.x_interpolant = kw.get('x_interpolant',DEFAULT_INTERP)
        self.k_interpolant = kw.get('k_interpolant',DEFAULT_INTERP)

        self._set_data(gal_image, psf_image)

    def get_dk(self):
        """
        we use the scale from the psf
        """
        return self.psf_kreal.scale

    def get_kimage(self, shear=None):
        """
        get the real part of the k-space image of the deconvolved galaxy

        returns
        -------
        kreal: Galsim Image
        """

        if shear is not None:
            return self.get_sheared_kimage(shear)
        else:
            return self.get_unsheared_kimage()


    def get_sheared_kimage(self, shear, **kw):
        """
        get the sheared k image

        don't shear the imaginary part
        """

        obj = self.get_sheared_gsobj(shear)

        kreal,kimag = obj.drawKImage(
            re=self.psf_kreal.copy(),
            im=self.psf_kimag.copy(),
        )
        return kreal, kimag

    def get_unsheared_kimage(self, **kw):
        """
        get the unsheared k image
        """

        kreal,kimag = self.igal_nopsf.drawKImage(
            re=self.psf_kreal.copy(),
            im=self.psf_kimag.copy(),
        )
        return kreal, kimag

    def _set_data(self, gal_image, psf_image_orig):

        psf_image = psf_image_orig.copy()

        imsum=psf_image.array.sum()
        if imsum == 0.0:
            raise DeconvRangeError("PSF image has zero flux")

        psf_image /= imsum

        print("using x interp:",self.x_interpolant)
        print("using k interp:",self.k_interpolant)
        self.igal = galsim.InterpolatedImage(
            gal_image,
            x_interpolant=self.x_interpolant,
            k_interpolant=self.k_interpolant,
        )
        self.ipsf = galsim.InterpolatedImage(
            psf_image,
            x_interpolant=self.x_interpolant,
            k_interpolant=self.k_interpolant,
        )

        self.ipsf_inv = galsim.Deconvolve(self.ipsf)

        self.igal_nopsf = galsim.Convolve(self.igal, self.ipsf_inv)

        self.psf_kreal,self.psf_kimag=self.ipsf.drawKImage()


