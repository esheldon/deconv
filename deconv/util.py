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
        sigma for weight in real space, will be 1/sigma in k space.

    returns
    -------
    A Moments object.  Use get_result() to get the result dict
    """
    deconv=DeConvolver(gal_image, psf_image)

    kimage = deconv.get_kimage()

    meas=Moments(kimage, sigma_weight)
    meas.go()

    return meas

def calcmom_obs(obs):
    """
    Deconvolve the image and measure the moments for the ngmix observation
    """
    pass
