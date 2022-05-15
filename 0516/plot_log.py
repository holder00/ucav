# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:28:09 2021

@author: OokiT1
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

res_name = "test"
f = open(res_name+"episode_vf_loss"+".pkl", mode="rb")
a = pickle.load(f)
# a = np.vstack((a,a))
f.close()
plt.ion()
f = open(res_name+"policy"+".pkl", mode="rb")
c = pickle.load(f)
fig = plt.figure(2)
f.close()
plt.clf()
cmap = plt.get_cmap("tab10")

ax = fig.add_subplot(311)
i = 0
res_max = a[:,-3]
res_mean = a[:,-2]
res_min = a[:,-1]
plt.plot(res_mean, linestyle = "dashed", label = res_name,color=cmap(i))
plt.plot(res_max,label = res_name+"max",color=cmap(i))
plt.plot(res_min, linestyle = ":",label = res_name+"min",color=cmap(i))
ax.legend()
plt.grid("on")
ax = fig.add_subplot(312)

for i in range(len(c)):
    res_max_b1 = c[i][:,0]
    res_mean_b1 = c[i][:,1]
    res_min_b1 = c[i][:,2]
    plt.plot(res_mean_b1, linestyle = "dashed", label = res_name+"Blue_"+str(i),color=cmap(i))
    plt.plot(res_max_b1,label = res_name+"Blue_"+str(i)+" max",color=cmap(i))
    plt.plot(res_min_b1, linestyle = ":",label = res_name+"Blue_"+str(i)+" min",color=cmap(i))
ax.legend()
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
plt.grid("on")

ax = fig.add_subplot(313)
for i in range(len(c)):
    pol = a[:,i]
    plt.plot(pol,label = res_name+"Blue_"+str(i)+" vf_loss",color=cmap(i))
ax.legend()
plt.grid("on")
plt.savefig("reward_res.png")