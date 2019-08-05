# -*- coding: utf-8 -*-

#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

#CAN STILL BE OPTIMIZED IF NECESSARY


def M_calc(int kmax, int tmax, double ds, double dk, double dt, double rho,
           np.ndarray[np.double_t, ndim=1] c,
           np.ndarray[np.double_t, ndim=1] S,
           np.ndarray[np.double_t, ndim=2] F):

    outM = open('OutM','w',1)
    cdef np.ndarray[np.double_t, ndim=2] Memory = np.zeros( [kmax, tmax] )#, dtype=np.double_t )
    cdef int q, t, k, s, u, smax = int(1./ds)
    cdef double sumk, sums, carnot, cos
    cdef double coeff = rho*(dk**3)*ds/( 8*(np.pi)**2 )
    # loop over the M(q,t) arguments
    for t in range(tmax):
        for q in range(1,kmax):
            # integral over dummy variable k
            sumk = 0.0
            for k in range(kmax):
                sums = 0.0
                for s in range(-smax,smax+1):
                  # carnot = |q-k| from cosine theorem
                    cos = s*ds
                    carnot = q**2+k**2-2.*k*q*cos
                    if carnot<.25:
                        u = 0
                    else:
                        u = int(np.sqrt(carnot)+0.5)
                    if u >= kmax:
                        u = kmax-1
                    # remember to better construct the weights!
                    if abs(s)==smax:
                        sums += .5*F[u,t]*(k*cos*c[k]+(q-k*cos)*c[u])**2
                    else:
                        sums +=    F[u,t]*(k*cos*c[k]+(q-k*cos)*c[u])**2
                    
                sumk += sums*F[k,t]*(k**2)
            #integral done; scale for factors
            Memory[q,t] = sumk*coeff*S[q]/(q**2)
            outM.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\n').format(
                          t*dt,q*dk,Memory[q,t] ))
    # job done
    return Memory
