#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import pyximport; pyximport.install()
import scipy as sp
import numpy as np
from scipy import interpolate as itp
from scipy import fft, ifft
from b_src_mct import M_calc

# ------MODE COUPLING THEORY-----------
# what do I do? I take as input S(k),
# and from that I should magically produce
# F(k,t) after some (==a lot of) iterations...
# Et voila'!

# inputs
rho = 0.4             # density
kT = 0.3              # reduced temperature
mass = 3.             # IPC mass
Sk_in = "MC_Sk_4_3"   # static structure factor
dk = 0.5              # integration k-grid step
dt = .5               # integration t-grid step
ds = 0.1              # dcosQ [-1,+1] integration step
#tmax = 64; Ntmax = int(tmax/dt)
                      # integration t-grid end
Ntmax = 128           # choose one!
smoothing = .003      # S(k) is spline int/extrapolated
Sk_out = "Fit_S"      # to check the quality of the splines
Fkt_fitout = "Fit_F"  # check the intial guess
Fkt_out = "OutF"      # final output Interm Scattering F
Mkt_out = "OutM"      # final output memory function
Nft = 2**10           # points for fourier-laplace transform
#


S_MC = []
k_MC = []
S_input = open(Sk_in,'r')
for linea in S_input:
    k_MC.append(float(linea.split()[0]))
    S_MC.append(float(linea.split()[1]))

# kill the first element of both (0, Nparticles)
k_MC.pop(0)
S_MC.pop(0)
k_new_end = k_MC[-1]
Nkmax = int(k_new_end/dk)

print("QT Grid: {}k x {}t = {}".format(Nkmax,Ntmax,Nkmax*Ntmax))
print("QW Grid: {}k x {}t = {}".format(Nkmax,Nft,Nkmax*Nft))

# interpolate g
kg = sp.arange(0., k_new_end, dk)
merdadelculo = itp.splrep(k_MC,S_MC,s=smoothing)
S = itp.splev(kg,merdadelculo,der=0)
S_output = open(Sk_out,'w')
c = (1.-1./S)/rho
for i in zip(kg,S,c):
    S_output.write( ('{0:.6e}\t{1:.10e}\t{2:.10e}\n').format(i[0],i[1],i[2]) )
S_output.flush()
S_output.close()

# ask if continue, otherwise exit.
if input('Type y to proceed: ') != 'y':
    exit()

# construct a guess for F, which has to be a matrix
F = sp.empty( (Nkmax, Ntmax) )
outF = open(Fkt_fitout,'w')
for k in range(Nkmax):
    for t in range(Ntmax):
        F[k,t] = S[k]*sp.exp(-(k*dk*0.01)*t*dt)
        outF.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\n').format(
                                               t*dt,k*dk,F[k,t]) )
# this guess made resembles good enough the solution
# as long as there is no two step decay, except for low k.
# Can be improved by fine tuning the exponent.
outF.flush()
outF.close()

# now, to the iteration cycle:
iteration = 1
while iteration > 0:
   Mtemp = M_calc(Nkmax,Ntmax,ds,dk,dt,rho,c,S,F)
   M = np.zeros((Nkmax,Nft))
   M[:,0:Ntmax] = Mtemp   # All k, on [0,tmax][-tmax,0] (mostly zeros)
   del Mtemp
   # Memory kernel has been computer from F
   Mhat = fft(M)*dt
   Fhat = np.zeros((Nkmax,Nft),dtype=np.complex)
   dw = 2*np.pi/(dt*Nft)
   w = np.arange(0.,dw*Nft,dw)
   # up to here seems reasonable
   outFc = open('Hat_F','w')
   for k in range(Nkmax):
      Omega = (kT*(k*dk)**2)/(mass*S[k])
      Fhat[k,:] =  S[k]*( sp.sqrt(-1)*w[:] + Omega*Mhat[k,:] ) / (
            -w[:]**2 + Omega*(1.+sp.sqrt(-1)*w[:]*Mhat[k,:]) )
      for wi in range(Nft):
         outFc.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\t{3:.10e}\n').format(
                           w[wi],k*dk,Fhat[k,wi].real,Fhat[k,wi].imag) )
   outFc.flush();   outFc.close()
   # is this coefficient right?
   Ftemp = ifft(Fhat)*np.pi/dt
   outF = open(Fkt_out,'w')
#   for k in range(Nkmax):
#      for t in range(Ntmax):
#         outF.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\t{3:.10e}\n').format(
#                           t*dt,k*dk,Ftemp[k,t].real,Ftemp[k,t].imag) )
   F[1:,:] = Ftemp[1:,:Ntmax].real
   F[0,:] = np.zeros(Ntmax)
   for k in range(Nkmax):
      for t in range(Ntmax):
         outF.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\n').format(
                           t*dt,k*dk,F[k,t].real) )
   outF.flush();   outF.close()
   if input("It "+str(iteration)+" done. Continue?\n") != 'y':
      break
   iteration += 1
# and that should be it...
