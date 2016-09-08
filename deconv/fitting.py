from __future__ import print_function
import math
import numpy
from numpy import array, zeros
import galsim

from . import deconv

from . import util
from .util import DeconvMaxiter
from .util import DeconvRangeError


class FitK(object):
    def __init__(self, obs, **kw):
        import ngmix
        self.mb_obs = ngmix.observation.get_mb_obs(obs)

        self.nband=len(self.mb_obs)

        self.kw=kw

        self._set_deconvolvers()
        self._kmax = util.find_min_kmax(self.mb_obs)

        self._build_kobs()

    def _build_kobs(self):
        """
        build up the observations in k space

        1) divide weights by 1/kpsf**2
            - but we can't impose a directioin in k space, so this must
            be a circularized version, or some function we choose
        2) set the weights to zero outside of kmax

        to get circularized, we can try that used in Bernstein 2010

        < |T(k)|^{-2} >^{-2}

        where the averaging is azimuthal

        or we could choose, for now, a circularized version of the gaussian fit
        to the PSF, where we don't allow it to shrink
        
        we could also do moments, and then the weights could be related
        to the best fit gaussians of both gal and psf (from a forward model fit)

        """
        pass

    def _set_deconvolvers(self):
        """
        set a deconvolver on each observation
        """

        deconv_class = self._get_deconvolver_class()
        fix_noise=self.kw.get('fix_noise',False)

        for obslist in self.mb_obs:
            for obs in obslist:
                pobs=obs.psf

                jac=obs.jacobian
                psf_jac=pobs.jacobian

                wcs = jac.get_galsim_wcs()
                psf_wcs = psf_jac.get_galsim_wcs()

                gsim = galsim.Image(obs.image.copy(), wcs=wcs)
                psf_gsim = galsim.Image(pobs.image.copy(), wcs=psf_wcs)

                obs.deconvolver = deconv_class(
                    gsim,
                    psf_gsim,
                    **self.kw
                )

                if fix_noise:
                    rnoise = numpy.random.normal(
                        loc=0.0,
                        scale=1.0,
                        size=obs.image.shape,
                    )
                    nim = numpy.sqrt( 1.0/obs.weight )
                    rnoise *= nim

                    gs_nim = galsim.Image(nim, wcs=wcs)

                    # new copy of the psf image
                    psf_gsim = galsim.Image(pobs.image.copy(), wcs=psf_wcs)

                    obs.noise_deconvolver = deconv_class(
                        gs_nim,
                        psf_gsim,
                        **self.kw
                    )




