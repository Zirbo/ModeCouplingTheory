# -*- coding: utf-8 -*-

#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

def cycle(int kmax, int tmax, double dk, double dt,
          double rho, double kT, double mass,
          np.ndarray[np.double_t, ndim=1] c,
          np.ndarray[np.double_t, ndim=1] S,
          np.ndarray[np.double_t, ndim=2] F):

   outM = open('OutM','w',1)
   cacca = open('cacca','w',1)
   cdef np.ndarray[np.double_t, ndim=2] M = np.zeros( [kmax, tmax] )
   cdef np.ndarray[np.double_t, ndim=1] K = np.arange( .5*dk, (2*kmax+.5)*dk, dk )
   cdef np.ndarray[np.double_t, ndim=1] K2 = K**2
   cdef np.ndarray[np.double_t, ndim=1] Km3 = K**(-3)
   cdef int q, t, k, p
   cdef double sumk, sums, store
   cdef double coeff = kT*rho*(dk**2)/( mass*32*(np.pi)**2 )
   # loop over the M(q,t) arguments
   for t in range(tmax):
      for q in range(kmax):
         # integral over dummy variable k
         sumk = 0.0
         for k in range(kmax):
            sump = 0.0
            for p in range(abs(q-k),min(q+k,kmax)):
               sump += K[p]*F[p,t]*( (K2[q]+K2[k]-K2[p])*c[k] + 
                                     (K2[q]+K2[p]-K2[k])*c[p] )**2
            sumk += sump*F[k,t]*K[k]
#            cacca.write('{0:4}\t{1:4}\t{2:4}\t{3:.10e}\n'.format(
#                         t,q,k,sump*F[k,t]*K[k]*coeff*S[q]/(K[q]**5)) )
         # integral done; scale for factors and print
         M[q,t] = sumk*coeff*Km3[q]
         outM.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\n').format(t*dt,K[q],M[q,t]) )
   outM.close()
   # job done; compute the new F now!
   cdef double integral
   cdef np.ndarray[np.double_t, ndim=2] Fn = np.zeros( [kmax, tmax] )
   cdef np.ndarray[np.double_t, ndim=1] omega = kT*K2[:kmax]/(mass*S)
   outF = open('OutF','w',1)
   for q in range(kmax):
      for t in range(tmax):
         #integrals = 0.0
         #for m in range(t):
         #   for n in range(m):
         #      integrals += F[q,n]*(omega[q]+M[q,m-n]) - M[q,n]*S[q]
         #Fn[q,t] = S[q] - integrals*(dt**2)
         integral = 0.0
         for i in range(t):
            integral += F[q,i]*(omega[q]+M[q,t-i]) - M[q,i]*S[q]
         Fn[q,t] = S[q] - integral*dt
         # integral done; print
         outF.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\n').format(t*dt,K[q],Fn[q,t]) )
   outF.close()
   return Fn
