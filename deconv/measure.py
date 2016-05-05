from __future__ import print_function
import numpy
from numpy import array
import galsim

from .weight import KSigmaWeight
from .deconv import DeConvolver


from . import util

MAX_ALLOWED_E=0.9999999
DEFVAL=-9999.0

LOW_T=2**0
HIGH_E=2**1
LOW_FLUX=2**2
LOW_WSUM=2**3

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
    **kw:
        Keywords passed on to constuctors for KSigmaWeight and
        DeConvolver, such as dk

    returns
    -------
    A Moments object.  Use get_result() to get the result dict
    """

    # deconvolve the psf
    deconv=DeConvolver(gal_image, psf_image, **kw)
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

    # save some other stuff
    meas.gal_image=gal_image
    meas.psf_image=psf_image

    return meas

def calcmom_ksigma_obs(obs, sigma_weight, skip_flagged=True, **kw):
    """
    Deconvolve the image and measure the moments using
    a sigmak type weight

    parameters
    ----------
    obs: ngmix Observation
        An ngmix Observation, ObsList, or MultiBandObsList,
        with a psf Observations set
    sigma_weight: float
        sigma for weight in real space, will be 1/sigma in k space.
        If scale if image is not unity, the sigma will be
        modified to be set to sigma*scale
    skip_flagged: bool
        When averaging from multiple images, skip the ones with
        flags set.  Default True.

    **kw:
        Keywords passed on to constuctors for KSigmaWeight and
        DeConvolver, such as dk

    returns
    -------
    An ObsKSigmaMoments object.  Use get_result() to get the result dict
    """

    meas=ObsKSigmaMoments(obs, sigma_weight, **kw)
    meas.go()
    return meas

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
        if not hasattr(self,'_result'):
            raise RuntimeError("run go() first")
        return self._result

    def go(self):
        """
        measure weighted moments in k space
        """

        rows,cols=self.get_rows_cols()
        kimage=self.kimage

        wim = self.kimage * self.kweight

        wsum   = self.kweight.sum()
        wimsum = wim.sum()

        irrsum_k = (rows**2*wim).sum()
        ircsum_k = (rows*cols*wim).sum()
        iccsum_k = (cols**2*wim).sum()

        # put back on the natural scale
        dk=self.dk
        dk2 = dk**2

        irrsum_k *= dk2
        ircsum_k *= dk2
        iccsum_k *= dk2

        self._set_result(irrsum_k, ircsum_k, iccsum_k, wimsum, wsum)

    def _set_result(self, irrsum_k, ircsum_k, iccsum_k, wimsum, wsum):
        from math import sqrt

        flags=0

        wflux=DEFVAL
        irr_k=DEFVAL
        irc_k=DEFVAL
        icc_k=DEFVAL

        e1=DEFVAL
        e2=DEFVAL
        T=DEFVAL
        T_k=DEFVAL

        if wsum <= 0.0:
            flags |= LOW_WSUM
        else:

            wflux = wimsum/wsum

            if wimsum <= 0.0:
                flags |= LOW_FLUX
            else:
                irr_k=irrsum_k/wimsum
                irc_k=ircsum_k/wimsum
                icc_k=iccsum_k/wimsum

                T_k = irr_k + icc_k

                if T_k <= 0.0:
                    flags |= LOW_T
                else:

                    T=1.0/T_k

                    e1 = -(icc_k - irr_k)/T_k
                    e2 = -2.0*irc_k/T_k

                    etot = sqrt(e1**2 + e2**2)
                    if etot >= MAX_ALLOWED_E:
                        flags |= HIGH_E

        self._result={
            'flags':flags,

            'e1':e1,
            'e2':e2,
            'e':array([e1,e2]),
            'T':T, # real space T
            'wflux': wflux,

            'irr_k':irr_k,
            'irc_k':irc_k,
            'icc_k':icc_k,
            'T_k':T_k,

            'irrsum_k':irrsum_k,
            'ircsum_k':ircsum_k,
            'iccsum_k':iccsum_k,

            'wsum':wsum,
            'wimsum':wimsum,

            'dk':self.dk,
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

    def get_rows_cols(self, **kw):

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

        return rows, cols


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

class ObsKSigmaMoments(Moments):
    """
    measure on ngmix Observation, ObsList, or MultiBandObsList

    Use a fixed ksigma weight
    """

    def __init__(self, obs, sigma_weight, skip_flagged=True, **kw):
        """
        parameters
        ----------
        obs: ngmix observation type
            an Observation, ObsList, or MultiBandObsList
        """
        import ngmix
        self.mb_obs = ngmix.observation.get_mb_obs(obs)
        self.sigma_weight=sigma_weight
        self.skip_flagged=skip_flagged
        self.kw=kw

        # there is generally no single dk
        if 'dk' in kw:
            self.dk=kw['dk']
        else:
            self.dk=-9999.0

    def get_result_list(self):
        """
        get the list of result dictionaries from all images
        """
        if not hasattr(self,'_reslist'):
            raise RuntimeError("run go() first")
        return self._reslist

    def get_meas_list(self):
        """
        get the list of measureres
        """
        if not hasattr(self,'_measlist'):
            raise RuntimeError("run go() first")
        return self._measlist


    def go(self):
        """
        measure moments on all images and perform the sum and mean
        """
        reslist=[]
        measlist=[]

        for obslist in self.mb_obs:
            for obs in obslist:
                tmeas=_calcmom_ksigma_obs(
                    obs,
                    self.sigma_weight,
                    **self.kw
                )
                res=tmeas.get_result()

                measlist.append(tmeas)
                reslist.append(res)

        self._measlist=measlist
        self._reslist=reslist
        self._combine_results()

    def _combine_results(self):
        flags=0

        nimage_use   = 0
        nimage_total = 0
        irrsum_k     = 0.0
        ircsum_k     = 0.0
        iccsum_k     = 0.0

        wsum         = 0.0
        wimsum       = 0.0

        wfluxsum     = 0.0

        for tres in self._reslist:
            tflags=tres['flags']

            nimage_total += 1

            if tflags == 0 or not self.skip_flagged:
                nimage_use += 1

                irrsum_k += tres['irrsum_k']
                ircsum_k += tres['ircsum_k']
                iccsum_k += tres['iccsum_k']

                wsum     += tres['wsum']
                wimsum   += tres['wimsum']

                # for now straight average
                wfluxsum += tres['wflux']

                flags |= tflags

        # now we can use the result setter in the parent
        # class to finish the job

        self._set_result(irrsum_k, ircsum_k, iccsum_k, wimsum, wsum)
        res=self._result

        # for now straight average
        if nimage_use > 0:
            res['wflux'] = wfluxsum/nimage_use

        res['nimage_use']   = nimage_use
        res['nimage_total'] = nimage_total
        res['orflags']      = flags

def _calcmom_ksigma_obs(obs, sigma_weight, **kw):
    """
    Deconvolve the image and measure the moments using
    a sigmak type weight

    parameters
    ----------
    obs: ngmix Observation
        An ngmix Observation, with a psf Observation set
    sigma_weight: float
        sigma for weight in real space, will be 1/sigma in k space.
        If scale if image is not unity, the sigma will be
        modified to be set to sigma*scale

    returns
    -------
    A Moments object.  Use get_result() to get the result dict
    """

    image=obs.image
    pimage=obs.psf.image
    jac=obs.jacobian

    wcs = jac.get_galsim_wcs()

    gsim = galsim.Image(image.copy(), wcs=wcs)
    psf_gsim = galsim.Image(pimage.copy(), wcs=wcs)

    meas=calcmom_ksigma(gsim, psf_gsim, sigma_weight, **kw)

    return meas


