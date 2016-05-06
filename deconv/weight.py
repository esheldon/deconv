from __future__ import print_function
import numpy
from numpy import array

from . import util

class KSigmaWeight(object):
    """
    A simple weight as  used in Bernstein et al. 2015

        (1 - k^2 sigma^2/2 N)^N  k < sqrt(2N)/sigma
                  0              k > sqrt(2N)/sigma
    """
    def __init__(self, sigma, N=4, **kw):
        self.sigma=sigma
        self.sigma2=sigma**2
        self.N=N

        self.kmax=numpy.sqrt(2*N)/sigma
        self.kmax2=self.kmax**2

    def find_weight_admom(self, imin, **kw):
        """
        iterate to get the weighted center and return
        the weight
        """
        import admom

        im=imin.copy()

        dims=im.shape

        guess=1.0/self.sigma2
        ccen=util.get_canonical_kcenter(dims)
        rows,cols=util.make_rows_cols(dims, cen=ccen)

        r2=rows**2 + cols**2
        w=numpy.where(r2 > self.kmax2)
        im[w] = 0.0

        ares=admom.admom(im, ccen[0], ccen[1], guess=guess,
                         maxiter=200)

        res={
            'flags':ares['whyflag'],
            'numiter':ares['numiter'],
            'Tguess':2*guess,
        }

        if res['flags']==0:
            cen=[ares['wrow'], ares['wcol']]
            weight, wrows, wcols=self.get_weight(dims, cen)

            res.update( {
                'cen':cen,
                'ccen':ccen,
                'T':ares['Irr']+ares['Icc'],
                'weight':weight,
                'rows':wrows,
                'cols':wcols,
            })
        return res


    def find_weight(self, im, **kw):
        """
        iterate to get the weighted center and return
        the weight
        """

        dims=im.shape
        rows,cols=util.make_rows_cols(dims, cen=[0.,0.])

        maxiter=kw.get('maxiter',100)
        tol=kw.get('tol',1.0e-3)

        cen=array(util.get_canonical_kcenter(dims),dtype='f8')
        cen += numpy.random.uniform(
            low=-0.1,
            high=0.1,
            size=2
        )

        flags=1
        for i in xrange(maxiter):
            weight,wrows,wcols=self.get_weight(dims, cen)

            wim=im*weight
            wimsum=wim.sum()

            row=(rows*wim).sum()/wimsum
            col=(cols*wim).sum()/wimsum

            drow=abs(row-cen[0])
            dcol=abs(col-cen[1])

            print("    %d: %g %g" % (i,row,col))
            if abs(drow) < tol and abs(dcol) < tol:
                flags=0
                break

            cen=[row,col]

        numiter=i+1

        res={
            'weight':weight,
            'rows':wrows,
            'cols':wcols,
            'cen':cen,
            'flags':flags,
            'numiter':numiter,
            'drow':drow,
            'dcol':dcol,
        }
        return res


    def get_weight(self, dims, cen):
        """
        get the weight function and the rows,cols for the grid
        """
        wt=numpy.zeros(dims)

        rows,cols=util.make_rows_cols(dims, cen=cen)

        k2 = rows**2 + cols**2

        w=numpy.where(k2 < self.kmax2)
        #print("npix > 0: %d/%d" % (w[0].size, dims[0]*dims[1]))
        if w[0].size > 0:
            wt[w] = (1.0 - k2[w]*self.sigma2/2/self.N)**self.N

        return wt, rows, cols


