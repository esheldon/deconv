from __future__ import print_function
import math
import numpy
from numpy import array, zeros
import galsim

from .weight import KSigmaWeight, KSigmaWeightC
from . import deconv

from . import util
from .util import DeconvMaxiter
from .util import DeconvRangeError

MAX_ALLOWED_E=0.9999999
DEFVAL=-9999.0
PDEFVAL=9999.0

LOW_T=2**0
HIGH_E=2**1
LOW_FLUX=2**2
LOW_WSUM=2**3
LOW_S2N_DENOM_SUM=2**4

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
    deconv=deconv.DeConvolver(gal_image, psf_image, **kw)
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

        if 'trim' in kw:
            print("DEPRECATED: trim is ignored")
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
        self._deweight = kw.get('deweight',False)
        self._force_same = kw.get('force_same',False)

        self._type=kw.get('type','ksigma')

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
        )
        meas.go()
        res=meas.get_result()

        return res

    def _get_ksigma_weight(self, gs_kimage):
        dk=gs_kimage.scale
        kwt = KSigmaWeight(self.sigma_weight*dk, **self.kw)
        return kwt


    def go(self, shear=None, doplots=False):
        """
        measure moments on all images and perform the sum and mean
        """

        fix_noise=self.kw.get('fix_noise',False)

        mb_reslist=[]

        dk = self._dk

        if not hasattr(self,'nx'):
            self.nx,self.ny=None,None
            #self.nx,self.ny=66,66

        for il,obslist in enumerate(self.mb_obs):
            reslist=[]
            for obs in obslist:

                ivar=obs.weight

                gs_kimage,gs_ikimage = obs.deconvolver.get_kimage(
                    shear=shear,
                    dk=dk,
                    nx=self.nx,
                    ny=self.ny,
                )

                self.ny,self.nx=gs_kimage.array.shape

                if fix_noise:

                    nshear=shear
                    if nshear is not None:
                        nshear = -nshear

                    ndk = gs_kimage.scale
                    rim,iim = obs.noise_deconvolver.get_kimage(
                        shear=nshear,
                        dk=ndk,
                        nx=self.nx,
                        ny=self.ny,
                    )
                    print(obs.deconvolver.dk,obs.noise_deconvolver.dk)
                    print(gs_kimage.array.shape, rim.array.shape)
                    gs_kimage += rim

                    # adding equal noise doubles the variance
                    ivar = ivar * (1.0/2.0)

                print("shape:",gs_kimage.array.shape,"dk:",gs_kimage.scale,"shear:",shear)
                if doplots:
                    import images
                    dims=gs_kimage.array.shape
                    kwt = KSigmaWeight(self.sigma_weight*gs_kimage.scale)
                    cen=util.get_canonical_kcenter(dims)
                    kweight, rows, cols = kwt.get_weight(dims, cen)

                    pim=gs_kimage.array*kweight
                    #pim=gs_kimage.array

                    #off=int(12*(0.1/obs.deconvolver.dk))
                    off=int(18*(0.1/obs.deconvolver.dk))
                    pim = pim[cen[0]-off:cen[0]+off+1,
                              cen[1]-off:cen[1]+off+1]
                    #images.multiview(obs.image,title='image')
                    #images.multiview(pim,title=str(shear))#,
                    images.multiview(obs.deconvolver.kreal,title=str(shear))#,
                                     #file='/astro/u/esheldon/www/tmp/plots/tmp.png')

                res=self._measure_moments(gs_kimage, ivar)

                reslist.append(res)

            mb_reslist.append( reslist )

        if self._force_same:
            self._kdims=ny,nx
            self._dk = dk

        self._mb_reslist=mb_reslist
        self._combine_results()

    def go_old(self, shear=None):
        """
        measure moments on all images and perform the sum and mean
        """

        fix_noise=self.kw.get('fix_noise',False)

        mb_reslist=[]

        # always force the same dk as first image
        dk = self._dk # could be None the first time through

        if dk is None:
            ny,nx=None,None
        else:
            ny,nx=self._kdims

        for il,obslist in enumerate(self.mb_obs):
            reslist=[]
            for obs in obslist:

                ivar=obs.weight

                #print("    using dk:",dk)
                gs_kimage,gs_ikimage = obs.deconvolver.get_kimage(
                    shear=shear,
                    dk=dk,
                    nx=nx,
                    ny=ny,
                )

                if self._force_same and il == 0:
                    # this will force them all to be the same from here on
                    dk = gs_kimage.scale
                    ny,nx=gs_kimage.array.shape

                #print("dk:",dk,"nx,ny:",nx,ny)

                if fix_noise:
                    #if self._force_same:
                    #    ndk = gs_kimage.scale
                    #else:
                    #    ndk = None

                    ndk = gs_kimage.scale
                    nny,nnx=gs_kimage.array.shape
                    nshear=shear
                    if nshear is not None:
                        nshear = -nshear

                    rim,iim = obs.noise_deconvolver.get_kimage(
                        shear=nshear,
                        nx=nnx,
                        ny=nny,
                        dk=ndk,
                    )
                    gs_kimage += rim

                    # adding equal noise doubles the variance
                    ivar = ivar * (1.0/2.0)

                res=self._measure_moments(gs_kimage, ivar)

                reslist.append(res)

            mb_reslist.append( reslist )

        if self._force_same:
            self._kdims=ny,nx
            self._dk = dk

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


                obs.deconvolver = deconv.DeConvolver(
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

                    obs.noise_deconvolver = deconv.DeConvolver(
                        gs_nim,
                        psf_gsim,
                        dk=self._dk, # can be None
                    )

    def _set_deconvolvers_fullwcs(self):

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

                    obs.noise_deconvolver = deconv_class(
                        gs_nim,
                        psf_gsim,
                        dk=obs.deconvolver.get_dk(),
                    )

    def _get_deconvolver_class(self):
        import deconv
        if self._type=='prerender':
            return deconv.DeConvolverPrerender
        elif self._type=='psfbase':
            return deconv.DeConvolverPSFBase
        elif self._type=='ksigma':
            return deconv.DeConvolver
        else:
            raise NotImplementedError("bad type: '%s'" % self._type)

class ObsGaussMoments(ObsKSigmaMoments):
    def __init__(self, *args, **kw):
        super(ObsGaussMoments,self).__init__(*args, **kw)

        self.noise_dims=array([32.0, 36.0, 48.0])
        self.noise_factors=array([18.34, 20.69, 27.75])

        #self.noise_factors={
        #    32:18.34,
        #    36:20.69,
        #    48:27.75,
        #}


    def _get_noise_factor(self, dim):
        #return self.noise_factors[dim]
        return numpy.interp(dim, self.noise_dims, self.noise_factors)

    def _measure_moments(self, gs_kimage, weight):
        import ngmix

        if self._deweight:
            raise NotImplementedError("no dweight yet for gauss moms")

        dk = gs_kimage.scale
        dk2 = dk**2
        dk4 = dk**4

        kwt, cen, dims = self._get_weight_object(gs_kimage)

        jacob=ngmix.UnitJacobian(
            row=cen[0],
            col=cen[1],
        )

        medwt = numpy.median(weight)
        dim=weight.shape[0]
        #noise_factor=self._get_noise_factor(dim)
        noise_factor = dim
        weight_factor = 1.0/noise_factor**2

        kivar = zeros( gs_kimage.array.shape ) + medwt*weight_factor


        kobs = ngmix.Observation(
            gs_kimage.array,
            weight=kivar,
            jacobian=jacob,
        )


        res = kwt.get_weighted_moments(kobs)

        # put onto a common scale
        if res['flags']==0:
            res['pars'][0:0+2] *= dk
            res['pars'][2:2+3] *= dk2

            res['pars_cov'][0:0+2, 0:0+2] *= dk2
            res['pars_cov'][2:2+2, 2:2+2] *= dk4

        return res

    def _combine_results(self):
        nband=self.nband

        orflags=0
        orflags_band    = zeros(nband,dtype='i4')

        nimage_band     = zeros(nband,dtype='i2')
        nimage_use_band = zeros(nband,dtype='i2')

        parsum        = zeros(6)
        pvarsum       = zeros( (6,6) )
        wsum_band     = zeros(nband)
        wsum          = 0.0
        wfluxsum_band = zeros(nband)
        wfluxsum      = 0.0
        wflux_band    = zeros(nband) + DEFVAL

        s2n_numer_sum = 0.0
        s2n_denom_sum = 0.0

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

                    s2n_numer_sum += tres['s2n_numer_sum']
                    s2n_denom_sum += tres['s2n_denom_sum']

        w,=numpy.where(wsum_band > 0.0)
        if w.size > 0:
            wflux_band[w] = wfluxsum_band[w]/wsum_band[w]

        # the wflux set here will be over-written
        res=self._get_result(parsum, pvarsum, wsum, s2n_numer_sum, s2n_denom_sum)

        nimage     = nimage_band.sum()
        nimage_use = nimage_use_band.sum()

        res['nimage_band']     = nimage_band
        res['nimage']          = nimage
        res['nimage_use_band'] = nimage_use_band
        res['nimage_use']      = nimage_use
        res['orflags_band']    = orflags_band
        res['orflags']         = orflags
        res['flags_band']      = orflags_band.copy()

        res['wflux_band']      = wflux_band

        self._result=res

    def _get_result(self, parsum, pvarsum, wsum, s2n_numer_sum, s2n_denom_sum):
        """
        averages:
            flux = sum(w*F)/sum(w)
            T = sum(w*T)/sum(w)
            M1 = sum(w*(x^2-y^2))/sum(w)
            M2 = sum(w*(2*x*y))/sum(w)
            e1 = -M1/T # minus due to k space
            e2 = -M2/T
        """
        from math import sqrt

        flags=0

        wflux=DEFVAL
        M1_k=DEFVAL
        M2_k=DEFVAL
        T_k=DEFVAL

        e1=DEFVAL
        e2=DEFVAL
        T=DEFVAL

        T_err  = PDEFVAL
        e1_err = PDEFVAL
        e2_err = PDEFVAL
        e_cov = zeros( (2,2) )+PDEFVAL

        flux_s2n=DEFVAL

        s2n_w=DEFVAL

        if wsum <= 0.0:
            print("    LOW_WSUM")
            flags = LOW_WSUM
        else:

            wflux    = parsum[5]/wsum

            if parsum[5] <= 0.0:
                print("    LOW_FLUX",wflux)
                flags = LOW_FLUX
            elif pvarsum[5,5] <= 0.0:
                print("    LOW FLUX VARSUM")
                flags = LOW_FLUX
            elif s2n_denom_sum <= 0.0:
                print("    LOW S2N_DENOM_SUM")
                flags = LOW_S2N_DENOM_SUM
            else:

                flux_s2n = parsum[5]/sqrt(pvarsum[5,5])

                s2n_w = s2n_numer_sum/sqrt(s2n_denom_sum)

                if self._deweight:
                    #irr_k,irc_k,icc_k,flags = self._deweight_moments(irr_k, irc_k, icc_k)
                    raise NotImplementedError("fix deweight")

                if flags==0:

                    if parsum[4] <= 0.0:
                        print("    LOW_T")
                        flags = LOW_T
                    else:

                        # T is 1/T_k so parsum[5]/parsum[4]

                        # e real space is - e k space
                        Tk =  parsum[4]/parsum[5]
                        sigmaksq = Tk/2.0
                        sigmasq = 1.0/sigmaksq
                        T = 2.0*sigmasq

                        denom_index=4

                        e1 = -parsum[2]/parsum[denom_index]
                        e2 = -parsum[3]/parsum[denom_index]

                        T_err = util.get_ratio_error(
                            parsum[5],
                            parsum[4],
                            pvarsum[5,5],
                            pvarsum[4,4],
                            pvarsum[5,4],
                        )
                        e1_err = util.get_ratio_error(
                            parsum[2],
                            parsum[denom_index],
                            pvarsum[2,2],
                            pvarsum[denom_index,denom_index],
                            pvarsum[2,denom_index],
                        )
                        e2_err = util.get_ratio_error(
                            parsum[3],
                            parsum[denom_index],
                            pvarsum[3,3],
                            pvarsum[denom_index,denom_index],
                            pvarsum[3,denom_index],
                        )

                        e_cov[0,1] = 0.0
                        e_cov[1,0] = 0.0
                        e_cov[0,0] = e1_err**2
                        e_cov[1,1] = e2_err**2

                        etot = sqrt(e1**2 + e2**2)
                        if etot >= MAX_ALLOWED_E:
                            flags = HIGH_E

        res={
            'flags':flags,

            'parsum':parsum,
            'pvarsum':pvarsum,

            'T':T, # real space T
            'T_err': T_err,

            'e1':e1,
            'e2':e2,
            'e':array([e1,e2]),

            'e1_err':e1_err,
            'e2_err':e2_err,
            'e_cov':e_cov,

            'wflux': wflux,
            'flux_s2n':flux_s2n,

            's2n_w':s2n_w,

            'wsum':wsum,

        }

        return res


    def _get_weight_object(self, gs_kimage):
        import ngmix

        dk=gs_kimage.scale
        dims=gs_kimage.array.shape
        cen=util.get_canonical_kcenter(dims)

        sigma = self.sigma_weight
        sigmak = 1.0/sigma

        # the k space image does not have unit pixel size
        sigmak *= (1.0/dk)

        Tk = 2.0*sigmak**2
        pars = [0.0, 0.0, 0.0, 0.0, Tk, 1.0]
        kwt = ngmix.GMixModel(pars, 'gauss')

        return kwt, cen, dims



class ObsKSigmaMomentsC(ObsGaussMoments):
    def _get_weight_object(self, gs_kimage):

        dk=gs_kimage.scale
        dims=gs_kimage.array.shape
        cen=util.get_canonical_kcenter(dims)

        kwt = KSigmaWeightC(self.sigma_weight*dk)

        return kwt, cen, dims

class KSigmaMomentsPSFBase(ObsKSigmaMomentsC):

    def __init__(self, obs, **kw):
        import ngmix

        self.mb_obs = ngmix.observation.get_mb_obs(obs)

        self.nband=len(self.mb_obs)

        self.kw=kw
        self._deweight = kw.get('deweight',False)
        self._type=kw.get('type',False)
        self._set_deconvolvers()
        self._set_sigma_weight()

    def go(self, shear=None, doplots=False):
        """
        measure moments on all images and perform the sum and mean
        """

        raise RuntimeError("doesn't account for mis-centering")

        fix_noise=self.kw.get('fix_noise',False)

        mb_reslist=[]


        for il,obslist in enumerate(self.mb_obs):
            reslist=[]
            for obs in obslist:

                # half for real and half for imag
                ivar = 0.5*obs.weight

                kr,ki = obs.deconvolver.get_kimage(
                    shear=shear,
                )

                if fix_noise:

                    nshear=shear
                    if nshear is not None:
                        nshear = -nshear

                    rim,iim = obs.noise_deconvolver.get_kimage(
                        shear=nshear,
                    )
                    kr += rim

                    # adding equal noise doubles the variance
                    ivar = ivar * (1.0/2.0)

                res=self._measure_moments(kr, ivar)

                reslist.append(res)

            mb_reslist.append( reslist )

        self._mb_reslist=mb_reslist
        self._combine_results()


    def _set_sigma_weight(self):
        """
        find the largest
        """

        N=4.0
        
        sigma_weight=0.0
        dk = 0.0
        for obslist in self.mb_obs:
            for obs in obslist:

                kreal = obs.deconvolver.psf_kreal
                kmax = util.find_kmax(kreal)

                if False:
                    import images
                    images.multiview(
                        kreal.array,
                        title='psf k real',
                    )

                print("    kmax:",kmax)

                tsigma = math.sqrt(2*N)/kmax
                if tsigma > sigma_weight:
                    sigma_weight = tsigma
                    dk = kreal.scale

        self.sigma_weight = sigma_weight/dk
        print("    sigma weight:",self.sigma_weight)

    def _set_deconvolvers(self):

        deconv_class = self._get_deconvolver_class()
        fix_noise=self.kw.get('fix_noise',False)

        interp=self.kw.get('interp',None)

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

class KSigmaMomentsPSFBasePS(KSigmaMomentsPSFBasePS):
    def __init__(self, *args, **kw):
        raise RuntimeError("not finished")

        super(KSigmaMomentsPSFBasePS,self).__init__(*args, **kw)
        self.ps=None
        self.scratch=None

    def _get_ps_scratch(self, kr, ki):
        if self.scratch is None:
            self.ps = kr.copy()
            self.scratch = ki.copy()
        else:
            self.ps.array[:,:] = kr.array[:,:]
            self.scratch.array[:,:] = ki.array[:,:]

        return self.ps, self.scratch

    def _make_ps(self, kr, ki):

        # ps and scratch will be copies of kr,ki
        ps, scratch = self._get_ps_scratch(kr, ki)

        ps *= kr
        scratch *= ki

        ps += scratch

        return ps
   
    def go(self, shear=None, doplots=False):
        """
        measure moments on all images and perform the sum and mean
        """

        fix_noise=self.kw.get('fix_noise',False)

        mb_reslist=[]

        first=True
        for il,obslist in enumerate(self.mb_obs):
            reslist=[]
            for obs in obslist:

                ivar=obs.weight

                kr,ki = obs.deconvolver.get_kimage(
                    shear=shear,
                )

                ps = self._make_ps(kr, ki)

                if fix_noise:

                    nshear=shear
                    if nshear is not None:
                        nshear = -nshear

                    noise_kr,noise_ki= obs.noise_deconvolver.get_kimage(
                        shear=nshear,
                    )

                    noise_ps = self._make_ps(noise_kr, noise_ki)

                    ps -= noise_ps

                    # adding equal noise doubles the variance
                    ivar = ivar * (1.0/2.0)

                res=self._measure_moments(kr, ivar)

                reslist.append(res)

            mb_reslist.append( reslist )

        self._mb_reslist=mb_reslist
        self._combine_results()


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


