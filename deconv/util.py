from __future__ import print_function

from .deconv import DeConvolver
from .measure import Moments


def calcmom(gal_image, psf_image, sigma_weight):
    """
    Deconvolve the image and measure the moments

    parameters
    ----------
    gal_image: Galsim image
        Galsim Image in real space
    psf_image: Galsim image
        Galsim Image in real space
    sigma_weight: float
        sigma for weight in pixels, will be 1/sigma in k space pixels.
        Note the k space image will generally be in sky coordinates
        not pixel coordinates, so adjust accordingly

    returns
    -------
    A Moments object.  Use get_result() to get the result dict
    """
    deconv=DeConvolver(gal_image, psf_image)

    kimage = deconv.get_kimage()

    meas=Moments(kimage, sigma_weight)
    meas.go()

    return meas
