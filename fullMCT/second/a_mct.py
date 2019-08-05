#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import pyximport; pyximport.install()
import scipy as sp
import numpy as np
from scipy import interpolate as itp
from scipy.optimize import curve_fit
from b_src_mct import cycle
from time import time

# ------MODE COUPLING THEORY-----------
# what do I do? I take as input S(k),
# and from that I should magically produce
# F(k,t) after some (==a lot of) iterations...
# Et voila'!

# inputs
rho =.718687          # density
kT = .92              # reduced temperature
mass = 3.             # IPC mass
#Sk_in = "MC_Sk_4_1"   # static structure factor
#Sk_in = "MC_Sk_trassa"# static structure factor
#Sk_in = "Sk_KF_6"     # static structure factor
Sk_in = "SEva"        # static structure factor
dk = .25              # integration k-grid step
RealKmax = 100.       # integration k-grid limit
inputKmax = 19.       # integration k-grid limit
dt = 1.               # integration t-grid step
# integration t-grid end; fix it how like
tmax = 128; Ntmax = int(tmax/dt)
#Ntmax = 128
#smoothing = .003      # S(k) is spline int/extrapolated
smoothing = 5.00     # S(k) is spline int/extrapolated
Sk_out = "Fit_S"      # to check the quality of the splines
Fkt_fitout = "Fit_F"  # check the intial guess
Fkt_out = "OutF"      # final output Interm Scattering F
Mkt_out = "OutM"      # final output memory function
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
#k_new_end = k_MC[-1]
Nkmax = int(RealKmax/dk)
inputNkmax = int(inputKmax/dk)

print("Total grid points: {}k x {}t = {}".format(Nkmax,Ntmax,Nkmax*Ntmax))

# interpolate S from the inputfile
kgrid = sp.arange(.5*dk, Nkmax*dk, dk)
S = np.zeros(Nkmax)
merdadelculo = itp.splrep(k_MC,S_MC,s=smoothing)
S[:inputNkmax] = itp.splev(kgrid[:inputNkmax],merdadelculo,der=0)
# extrapolate S to the extended grid via a prototype function
if (inputKmax < RealKmax):
   def extrap(x,A,B,a,w,p):
      return A+B*np.exp(-a*x)*np.sin(w*x+p)/x
   p0 = (1.,4.,.01,-1.,3.)
   p1,pcov = curve_fit(extrap,kgrid[int(5./dk):inputNkmax],S[int(5./dk):inputNkmax],p0)
   print(p1)
   p1[0]=1.
   S[inputNkmax:] = extrap(kgrid[inputNkmax:],p1[0],p1[1],p1[2],p1[3],p1[4])
# fertig! Druck so dass der Benutzer seine Meinung sagen kann!
S_output = open(Sk_out,'w')
c = (S-1.)/(S*rho)
for i in zip(kgrid,S,c):
   S_output.write( ('{0:.6e}\t{1:.10e}\t{2:.10e}\n').format(i[0],i[1],i[2]) )
S_output.flush()
S_output.close()

# ask if continue, otherwise exit.
if input('Check S(k) and type y to proceed: ') != 'y':
   exit()

# construct a guess for F, which has to be a matrix
F = sp.empty( (Nkmax, Ntmax) )
outF = open(Fkt_fitout,'w')
for k in range(Nkmax):
   for t in range(Ntmax):
      F[k,t] = max(S[k]*(sp.exp(-(0.005*kgrid[k]*t*dt)**2)) , S[k]-((kT/mass)*(kgrid[k]*t*dt)**2) )
      #F[k,t] = S[k]
      outF.write( ('{0:.6e}\t{1:.6e}\t{2:.10e}\n').format(
                                               t*dt,(k+.5)*dk,F[k,t]) )
# this guess made resembles good enough the solution
# as long as there is no two step decay, except for low k.
# Can be improved by fine tuning the exponent.
outF.flush()
outF.close()

# now, to the iteration cycle:
iteration = 1
while iteration > 0:
   tempo = time()
   F = cycle(Nkmax,Ntmax,dk,dt,rho,kT,mass,c,S,F)
   tempo = time() - tempo
   if input("It "+str(iteration)+" done in "+str(tempo)+" s. Continue?\n") != 'y':
      break
   iteration += 1
# and that should be it...
