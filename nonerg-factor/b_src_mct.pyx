# -*- coding: utf-8 -*-

#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

def cycle(int kmax, double dk, double rho,
          np.ndarray[np.double_t, ndim=1] c,
          np.ndarray[np.double_t, ndim=1] S,
          np.ndarray[np.double_t, ndim=1] f):

   outM = open('outM','w',1)
   cdef np.ndarray[np.double_t, ndim=1] M = np.zeros( kmax )
   cdef np.ndarray[np.double_t, ndim=1] K = np.arange( .5*dk, (2*kmax+.5)*dk, dk )
   cdef np.ndarray[np.double_t, ndim=1] K2 = K**2
   cdef np.ndarray[np.double_t, ndim=1] Km5 = K**(-5)
   cdef int q, t, k, p
   cdef double sumk, sums, store
   cdef double coeff = rho*(dk**2)/( 32*(np.pi)**2 )
   # loop over the M(q,t) arguments
   for q in range(kmax):
      # integral over dummy variable k
      sumk = 0.0
      for k in range(kmax):
         sump = 0.0
         for p in range(abs(q-k),min(q+k,kmax)):
            sump += K[p]*f[p]*S[p]*( (K2[q]+K2[k]-K2[p])*c[k] + 
                                     (K2[q]+K2[p]-K2[k])*c[p] )**2
         sumk += sump*f[k]*S[k]*K[k]
      # integral done; scale for factors and print
      M[q] = sumk*coeff*S[q]*Km5[q]
      outM.write( ('{0:.6e}\t{1:.10e}\n').format(K[q],M[q]) )
   outM.close()
   # job done; compute the new F now!
   f = M / (1.+M)
   return f
