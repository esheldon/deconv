from __future__ import print_function
import numpy
from numpy import array, zeros
import galsim

from .weight import KSigmaWeight
from .deconv import DeConvolver


from . import util
from .util import DeconvMaxiter

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
    dk: float
        Step in k space, in units of 1/scale of the wcs
    **kw:
        Keywords passed on to constuctors for KSigmaWeight and
        DeConvolver

    returns
    -------
    A Moments object.  Use get_result() to get the result dict
    """

    # deconvolve the psf
    # note dk can be set in kw
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

'''
res=kwt.find_weight(gs_kimage.array, **kw)
for k in res:
    if k not in ['weight','rows','cols']:
        print("    ",k,res[k])

if res['flags'] != 0:
    print("    MAX IT REACHED, using standared")
    dims=gs_kimage.array.shape
    cen=util.get_canonical_kcenter(dims)
    kweight, rows, cols = kwt.get_weight(dims, cen)
else:
    kweight=res['weight']
'''

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

        rows,cols = util.make_rows_cols(dims, cen)
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

    parameters
    ----------
    obs: ngmix Observation
        An ngmix Observation, with a psf Observation set
    sigma_weight: float
        sigma for weight in real space, will be 1/sigma in k space.
    dk: float
        Step in k space, in units of 1/scale of the wcs
    **kw:
        other keywords for calcmom_ksigma
    """

    def __init__(self, obs, sigma_weight, skip_flagged=True, **kw):
        import ngmix
        self.mb_obs = ngmix.observation.get_mb_obs(obs)

        self.nband=len(self.mb_obs)

        self.sigma_weight=sigma_weight
        self.skip_flagged=skip_flagged
        self.kw=kw

        self._set_deconvolvers()

    def get_result_list(self):
        """
        get the list of result dictionaries from all images
        """
        if not hasattr(self,'_mb_reslist'):
            raise RuntimeError("run go() first")
        return self._mb_reslist

    def get_meas_list(self):
        """
        get the list of measureres
        """
        if not hasattr(self,'_mb_measlist'):
            raise RuntimeError("run go() first")
        return self._mb_measlist


    def go(self, shear=None):
        """
        measure moments on all images and perform the sum and mean
        """

        fix_noise=self.kw.get('fix_noise',False)

        mb_reslist=[]
        mb_measlist=[]

        for obslist in self.mb_obs:
            reslist=[]
            measlist=[]
            for obs in obslist:

                gs_kimage = obs.deconvolver.get_kimage(
                    shear=shear,
                )

                if fix_noise:
                    ny,nx=gs_kimage.array.shape
                    nshear=shear
                    if nshear is not None:
                        nshear = -nshear
                    """
                    print(ny,nx)
                    tmp = obs.noise_deconvolver.get_kimage(
                        shear=-shear,
                        nx=nx,
                        ny=ny,
                        dk=gs_kimage.scale,
                    )

                    print(gs_kimage.array.shape, tmp.array.shape)
                    stop
                    """

                    gs_kimage += obs.noise_deconvolver.get_kimage(
                        shear=nshear,
                        nx=nx,
                        ny=ny,
                        dk=gs_kimage.scale,
                    )


                dk=gs_kimage.scale
                kwt = KSigmaWeight(self.sigma_weight*dk, **self.kw)

                dims=gs_kimage.array.shape
                cen=util.get_canonical_kcenter(dims)
                kweight, rows, cols = kwt.get_weight(dims, cen)

                # calculate moments
                meas=Moments(
                    gs_kimage,
                    kweight,
                )
                meas.go()

                res=meas.get_result()

                measlist.append(meas)
                reslist.append(res)

            mb_reslist.append( reslist )
            mb_measlist.append( measlist )

        self._mb_measlist=mb_measlist
        self._mb_reslist=mb_reslist
        self._combine_results()

    def _combine_results(self):
        nband=self.nband

        orflags=0
        orflags_band    = zeros(nband,dtype='i4')

        nimage_band     = zeros(nband,dtype='i2')
        nimage_use_band = zeros(nband,dtype='i2')

        irrsum_k        = 0.0
        ircsum_k        = 0.0
        iccsum_k        = 0.0

        wsum            = 0.0
        wimsum          = 0.0

        wfluxsum_band   = zeros(nband)
        wfluxsum        = 0.0

        for band,reslist in enumerate(self._mb_reslist):

            for tres in reslist:

                nimage_band[band] += 1

                flags=tres['flags']

                orflags_band[band] |= flags
                orflags |= flags

                if flags == 0 or not self.skip_flagged:
                    nimage_use_band[band] += 1

                    irrsum_k += tres['irrsum_k']
                    ircsum_k += tres['ircsum_k']
                    iccsum_k += tres['iccsum_k']

                    wsum     += tres['wsum']
                    wimsum   += tres['wimsum']

                    # for now straight average
                    wfluxsum_band[band] += tres['wflux']
                    wfluxsum += tres['wflux']

        # the wflux set here will be over-written
        self._set_result(irrsum_k, ircsum_k, iccsum_k, wimsum, wsum)
        res=self._result

        nimage     = nimage_band.sum()
        nimage_use = nimage_use_band.sum()

        res['nimage_band']     = nimage_band
        res['nimage']          = nimage
        res['nimage_use_band'] = nimage_use_band
        res['nimage_use']      = nimage_use
        res['orflags_band']    = orflags_band
        res['orflags']         = orflags
        res['flags_band']      = orflags_band.copy()

        res['wflux_band']      = numpy.zeros(nband) + DEFVAL
        res['wflux']           = DEFVAL

        if nimage_use > 0:

            # for now straight average of fluxes
            w,=numpy.where(nimage_use_band > 0)
            if w.size > 0:
                res['wflux_band'][w] = wfluxsum_band[w]/nimage_use_band[w]
                res['flags_band'][w] = 0

            res['wflux'] = wfluxsum/nimage_use

    def _set_deconvolvers(self):

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


                obs.deconvolver = DeConvolver(
                    gsim,
                    psf_gsim,
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

                    obs.noise_deconvolver = DeConvolver(
                        gs_nim,
                        psf_gsim,
                    )



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
    dk: float
        Step in k space, in units of 1/scale of the wcs
    **kw:
        other keywords for calcmom_ksigma

    returns
    -------
    A Moments object.  Use get_result() to get the result dict
    """

    pobs=obs.psf

    image=obs.image
    pimage=pobs.image

    jac=obs.jacobian
    psf_jac=pobs.jacobian

    wcs = jac.get_galsim_wcs()
    psf_wcs = psf_jac.get_galsim_wcs()

    gsim = galsim.Image(image.copy(), wcs=wcs)
    psf_gsim = galsim.Image(pimage.copy(), wcs=psf_wcs)

    meas=calcmom_ksigma(gsim, psf_gsim, sigma_weight, **kw)

    return meas


