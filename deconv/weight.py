from __future__ import print_function
import numpy

class KSigmaWeight(object):
    """
    A simple weight as  used in Bernstein et al. 2015

        (1 - k^2 sigma^2/2 N)^N  k < sqrt(2N)/sigma
                  0              k > sqrt(2N)/sigma
    """
    def __init__(self, sigma, N=4):
        self.sigma=sigma
        self.sigma2=sigma**2
        self.N=N

        self.kmax=numpy.sqrt(2*N)/sigma
        self.kmax2=self.kmax**2

    def get_weight(self, dims, cen):
        """
        get the weight function and the rows,cols for the grid
        """
        wt=numpy.zeros(dims)
        rows,cols=numpy.mgrid[
            0:dims[0],
            0:dims[1],
        ]

        rows=numpy.array(rows, dtype='f8')
        cols=numpy.array(cols, dtype='f8')

        rows -= cen[0]
        cols -= cen[1]

        k2 = rows**2 + cols**2

        w=numpy.where(k2 < self.kmax2)
        #print("npix > 0: %d/%d" % (w[0].size, dims[0]*dims[1]))
        if w[0].size > 0:
            wt[w] = (1.0 - k2[w]*self.sigma2/2/self.N)**self.N

        return wt, rows, cols


