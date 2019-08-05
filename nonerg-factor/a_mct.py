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
# Produces the nonergodicity factor f(q) = lim t->+oo F(q,t)/S(q)
# no time dependencies

# inputs
rho = .718687         # density
kT = .92              # reduced temperature
mass = 3.             # IPC mass
#Sk_in = "MC_Sk_4_1"   # static structure factor
#Sk_in = "fluid30n4015"# static structure factor
#Sk_in = "S30n40_10"   # static structure factor
Sk_in = "SEva"        # static structure factor
dk = .25              # integration k-grid step
RealKmax = 200.       # integration k-grid limit
inputKmax = 19.       # integration k-grid limit
#smoothing = .003      # S(k) is spline int/extrapolated
smoothing = 5.00      # S(k) is spline int/extrapolated
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


# interpolate S from the inputfile
kgrid = sp.arange(.5*dk, Nkmax*dk, dk)
S = np.zeros(Nkmax)
splineinterpolator = itp.splrep(k_MC,S_MC,s=smoothing)
S[:inputNkmax] = itp.splev(kgrid[:inputNkmax],splineinterpolator,der=0)
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

f = np.empty(Nkmax)
f[:] = np.maximum(S[:]-1.,.25)
#f[:] = .4

# now, to the iteration cycle:
iteration = 1
while iteration > 0:
   tempo = time()
   outF = open('outF','w',10)
   for q in range(Nkmax):                                                        
      outF.write( ('{0:.6e}\t{1:.10e}\t{2:.10e}\n').format(kgrid[q],f[q],f[q]*S[q]) )
   outF.close()
   f = cycle(Nkmax,dk,rho,c,S,f)
   tempo = time() - tempo
   print("It "+str(iteration)+" done in "+str(tempo)+" s.\n")
#   if input("It "+str(iteration)+" done in "+str(tempo)+" s. Continue?\n") != 'y':
#      break
   iteration += 1
# and that should be it...
