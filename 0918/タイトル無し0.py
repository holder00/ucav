# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:05:27 2021

@author: Takumi
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def state(state0,t,T,alp,phi):
    x,y,z,V,psi,gam = state0[0],state0[1],state0[2],state0[3],state0[4],state0[5]
    D = (Cd2*alp*alp+Cd0)*V*V
    L = (Cl1*alp+Cl0)*V*V
    alp = gam
    v = [0]*6
    v[0] = V*np.cos(gam)*np.sin(psi)
    v[1] = V*np.cos(gam)*np.cos(psi)
    v[2] = V*np.sin(gam)
    v[3] = T/mass*np.cos(alp) -D/mass -g*np.sin(gam)
    v[4] = L/(mass*V*np.cos(gam))*np.sin(phi)
    v[5] = (T*np.sin(alp) + L)/(mass*V)*np.cos(phi)-g/V*np.cos(gam)
    return v

mass = 10000 #kg
g = 9.80665 #m/ss

Cd2 = 0.3
Cd0 = 0.1
Cl1 = 0.1
Cl0 = 2


x = 0
y = 0
z = 1000

sim_dt = 0.1

#state
V = 250 #m/s
gam = np.deg2rad(0)
psi = np.deg2rad(0)

T = 5000
alp = np.deg2rad(0)
phi = np.deg2rad(30)

res = np.zeros([10000,6])

state0 = [x,y,z, V, psi, gam]
t = np.arange(0, 1000, sim_dt)

for i in range(10000):
    v = state(state0, t, T, alp, phi)
    
    state0 = state0 + v*np.array([sim_dt])
    res[i] = state0
    
V = 250 #m/s
gam = np.deg2rad(0)
psi = np.deg2rad(0)

T = 5000
alp = np.deg2rad(0)
phi = np.deg2rad(30)
state0 = [x,y,z, V, psi, gam]

v = odeint(state, state0, t, args=(T, alp, phi))
fig = plt.figure(1)
ax = fig.gca(projection='3d')
X = v[:, 0]
Y = v[:, 1]
Z = v[:, 2]
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.plot(X, Y, Z)
# ax.set_aspect('equal')



# X = res[:, 0]
# Y = res[:, 1]
# Z = res[:, 2]+2500
# max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

# mid_x = (X.max()+X.min()) * 0.5
# mid_y = (Y.max()+Y.min()) * 0.5
# mid_z = (Z.max()+Z.min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)

# ax.plot(X, Y, Z)



