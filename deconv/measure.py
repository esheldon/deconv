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
    gs_kimage,gs_ikimage = deconv.get_kimage()

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

        self._trim = kw.get('trim',False)
        self._deweight = kw.get('deweight',False)

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


        if self._trim:

            cen=self.cen
            dims=self.kimage.shape
            mindist=min(cen[0], (dims[0]-1)-cen[0],
                        cen[1], (dims[1]-1)-cen[1])
            rsq=rows**2 + cols**2
            w=numpy.where(rsq <= mindist**2)

            wsum   = self.kweight[w].sum()
            wimsum = wim[w].sum()

            irrsum_k = (rows[w]**2*wim[w]).sum()
            ircsum_k = (rows[w]*cols[w]*wim[w]).sum()
            iccsum_k = (cols[w]**2*wim[w]).sum()


        else:
            wsum   = self.kweight.sum()
            wimsum = wim.sum()

            irrsum_k = (rows**2*wim).sum()
            ircsum_k = (rows*cols*wim).sum()
            iccsum_k = (cols**2*wim).sum()

        #wmax=wim.argmax()
        #cen=self.cen
        #print("maxval:",wim.ravel()[wmax],"val at center:",
        #      wim[cen[0],cen[1]])

        # put back on the natural scale
        dk=self.dk
        dk2 = dk**2

        irrsum_k *= dk2
        ircsum_k *= dk2
        iccsum_k *= dk2

        M1sum_k = iccsum_k - irrsum_k
        M2sum_k = 2.0*ircsum_k
        Tsum_k = iccsum_k + irrsum_k

        self._set_result(M1sum_k, M2sum_k, Tsum_k, wimsum, wsum)


    def _set_result(self, M1sum_k, M2sum_k, Tsum_k, wimsum, wsum):
        from math import sqrt

        flags=0


        wflux=DEFVAL
        M1_k=DEFVAL
        M2_k=DEFVAL
        T_k=DEFVAL

        e1=DEFVAL
        e2=DEFVAL
        T=DEFVAL

        if wsum <= 0.0:
            flags |= LOW_WSUM
        else:

            wflux = wimsum/wsum

            if wimsum <= 0.0:
                flags |= LOW_FLUX
            else:
                M1_k=M1sum_k/wimsum
                M2_k=M2sum_k/wimsum
                T_k=Tsum_k/wimsum

                if self._deweight:
                    raise NotImplementedError("fix deweight")
                    irr_k,irc_k,icc_k,flags = self._deweight_moments(irr_k, irc_k, icc_k)

                if flags==0:

                    if T_k <= 0.0:
                        flags |= LOW_T
                    else:

                        T=1.0/T_k
                        e1 = -M1_k/T_k
                        e2 = -M2_k/T_k

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

            'M1_k':M1_k,
            'M2_k':M2_k,
            'T_k':T_k,

            'M1sum_k':M1sum_k,
            'M2sum_k':M2sum_k,
            'Tsum_k':Tsum_k,

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

    def __init__(self, obs, sigma_weight, skip_flagged=True, weight_type='ksigma', **kw):
        import ngmix
        self.mb_obs = ngmix.observation.get_mb_obs(obs)

        self.nband=len(self.mb_obs)

        self.sigma_weight=sigma_weight
        self.weight_type=weight_type

        self.skip_flagged=skip_flagged
        self.kw=kw
        self._trim = kw.get('trim',False)
        self._deweight = kw.get('deweight',False)
        self._dk = kw.get('dk',None)

        #self._set_deconvolvers()
        self._set_deconvolvers_fullwcs()

    def get_result_list(self):
        """
        get the list of result dictionaries from all images
        """
        if not hasattr(self,'_mb_reslist'):
            raise RuntimeError("run go() first")
        return self._mb_reslist

    def _measure_moments(self, gs_kimage, *args):
        kwt = self._get_ksigma_weight(gs_kimage)

        dims=gs_kimage.array.shape
        cen=util.get_canonical_kcenter(dims)

        kweight, rows, cols = kwt.get_weight(dims, cen)

        # calculate moments
        meas=Moments(
            gs_kimage,
            kweight,
            trim=self._trim, # trim the k space image
        )
        meas.go()
        res=meas.get_result()

        return res

    def _get_ksigma_weight(self, gs_kimage):
        dk=gs_kimage.scale
        kwt = KSigmaWeight(self.sigma_weight*dk, **self.kw)
        return kwt


    def go(self, shear=None):
        """
        measure moments on all images and perform the sum and mean
        """

        fix_noise=self.kw.get('fix_noise',False)

        mb_reslist=[]

        for obslist in self.mb_obs:
            reslist=[]
            measlist=[]
            for obs in obslist:

                gs_kimage,gs_ikimage = obs.deconvolver.get_kimage(
                    shear=shear,
                )
                #print("dk:",gs_kimage.scale)

                #print("before:",gs_kimage(10,10))
                if fix_noise:
                    #print("fixing noise")
                    ny,nx=gs_kimage.array.shape
                    nshear=shear
                    if nshear is not None:
                        nshear = -nshear

                    rim,iim = obs.noise_deconvolver.get_kimage(
                        shear=nshear,
                        nx=nx,
                        ny=ny,
                        dk=gs_kimage.scale,
                    )
                    gs_kimage += rim

                    #print("after:",gs_kimage(10,10))

                res=self._measure_moments(gs_kimage, obs.weight)

                measlist.append(meas)
                reslist.append(res)

            mb_reslist.append( reslist )

        self._mb_reslist=mb_reslist
        self._combine_results()

    def _combine_results(self):
        nband=self.nband

        orflags=0
        orflags_band    = zeros(nband,dtype='i4')

        nimage_band     = zeros(nband,dtype='i2')
        nimage_use_band = zeros(nband,dtype='i2')

        M1sum_k  = 0.0
        M2sum_k  = 0.0
        Tsum_k   = 0.0

        wsum     = 0.0
        wimsum   = 0.0

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

                    M1sum_k += tres['M1sum_k']
                    M2sum_k += tres['M2sum_k']
                    Tsum_k += tres['Tsum_k']

                    wsum     += tres['wsum']
                    wimsum   += tres['wimsum']

                    # for now straight average
                    wfluxsum_band[band] += tres['wflux']
                    wfluxsum += tres['wflux']

        # the wflux set here will be over-written
        self._set_result(M1sum_k, M2sum_k, Tsum_k, wimsum, wsum)
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

    def _deweight_moments(self, irr_k, irc_k, icc_k):
        """
        we would subtract (1/sigma_k^2) I
        and sigmak is 1/sigma so just subtract sigma^2
        """
        #print("    deweighting")
        flags=0

        sigmasq=self.sigma_weight**2

        m=numpy.zeros( (2,2) )
        m[0,0] = irr_k
        m[0,1] = irc_k
        m[1,0] = irc_k
        m[1,1] = icc_k

        try:
            minv = numpy.linalg.inv(m)

            minv[0,0] -= sigmasq
            minv[1,1] -= sigmasq

            dm = numpy.linalg.inv(minv)
        except numpy.linalg.LinAlgError as err:
            print("error inverting matrix: '%s'" % str(err))
            flags=1

        irr=dm[0,0]
        irc=dm[0,1]
        icc=dm[1,1]

        return irr,irc,icc,flags

    def _set_deconvolvers(self):

        fix_noise=self.kw.get('fix_noise',False)

        for obslist in self.mb_obs:
            for obs in obslist:
                pobs=obs.psf

                jac=obs.jacobian
                psf_jac=pobs.jacobian

                gsim = galsim.Image(obs.image.copy(), scale=1.0)
                psf_gsim = galsim.Image(pobs.image.copy(), scale=1.0)


                obs.deconvolver = DeConvolver(
                    gsim,
                    psf_gsim,
                    dk=self._dk, # can be None
                )

                if fix_noise:
                    rnoise = numpy.random.normal(
                        loc=0.0,
                        scale=1.0,
                        size=obs.image.shape,
                    )
                    nim = numpy.sqrt( 1.0/obs.weight )
                    rnoise *= nim

                    gs_nim = galsim.Image(nim, scale=1.0)

                    # new copy of the psf image
                    psf_gsim = galsim.Image(pobs.image.copy(), scale=1.0)

                    obs.noise_deconvolver = DeConvolver(
                        gs_nim,
                        psf_gsim,
                        dk=self._dk, # can be None
                    )

    def _set_deconvolvers_fullwcs(self):

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
                    dk=self._dk, # can be None
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
                        dk=self._dk, # can be None
                    )


class ObsGaussMoments(ObsKSigmaMoments):
    def _measure_moments(self, gs_kimage, weight):
        import ngmix

        if self._deweight:
            raise NotImplementedError("no dweight yet for gauss moms")

        dk = gs_kimage.scale
        dksq = dk**2

        kwt, cen, dims = self._get_gauss_weight(gs_kimage)

        if self._trim:
            mindist=min(cen[0], (dims[0]-1)-cen[0],
                        cen[1], (dims[1]-1)-cen[1])
            rmax=mindist

        else:
            rmax=1.0e20

        jacob=ngmix.UnitJacobian(
            row=cen[0],
            col=cen[1],
        )
        kobs = ngmix.Observation(
            gs_kimage.array,
            weight=weight,
            jacobian=jacob,
        )
        res = kwt.get_weighted_moments(kobs, rmax=rmax)

        return res

    def _combine_results(self):
        nband=self.nband

        orflags=0
        orflags_band    = zeros(nband,dtype='i4')

        nimage_band     = zeros(nband,dtype='i2')
        nimage_use_band = zeros(nband,dtype='i2')

        parsum = zeros(6)
        pvarsum = zeros( (6,6) )

        wsum_band = zeros(nband)
        wsum      = 0.0

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

                    parsum += tres['pars']
                    pvarsum += tres['pars_cov']

                    wsum_band[band] += tres['wsum']
                    wsum            += tres['wsum']

                    # this is weight*flux
                    wfluxsum_band[band] += tres['pars'][5]
                    wfluxsum += tres['pars'][5]

        # the wflux set here will be over-written
        res=self._get_result(parsum, pvarsum, wsum)

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
                res['wflux_band'][w] = wfluxsum_band[w]/wsum_band[band]
                res['flags_band'][w] = 0

            res['wflux'] = wfluxsum/wsum


        wflux=DEFVAL
        irr_k=DEFVAL
        irc_k=DEFVAL
        icc_k=DEFVAL

        e1=DEFVAL
        e2=DEFVAL
        T=DEFVAL
        T_k=DEFVAL

        flags=res['flags']
        if flags==0:
            pars=res['pars']
            wfluxsum = pars[5]

            if wfluxsum <= 0.0:
                flags |= LOW_FLUX
            else:
                T_k = pars[4]/wfluxsum
                if T_k <= 0:
                    flags |= LOW_T
                else:

                    T_k *= dk**2
                    M1_k = pars[2]/wfluxsum*dk**2
                    M2_k = pars[3]/wfluxsum*dk**2

                    T=1.0/T_k
                    e1 = -M1_k/T
                    e2 = -M2_k/T

                    wfluxsum_err = sqrt(res['pars_cov'][5,5])
                    s2n_w = wfluxsum/wfluxsum_err

                    res['T_k'] = T_k
                    res['M1_k'] = M1_k
                    res['M2_k'] = M2_k
                    res['s2n_w'] = wflux/res['wsum']

        return res


    def _get_gaussian_weight(self, gs_kimage):
        import ngmix

        dk=gs_kimage.scale
        dims=gs_kimage.array.shape
        cen=util.get_canonical_kcenter(dims)

        sigma = self.sigma_weight
        sigmak = 1.0/sigma

        # the k space image does not have unit pixel size
        sigmak *= (1.0/dk)

        Tk = 2.0*sigmak**2
        pars = [cen[0], cen[1], 0.0, 0.0, Tk, 1.0]
        kwt = ngmix.GMixModel(pars, 'gauss')

        return kwt, cen, dims

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


