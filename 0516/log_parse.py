# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:08:55 2021

@author: OokiT1
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
f = open('Train.ipynb', 'r',encoding='utf-8')
test_case = "役割選択"
datalist = f.readlines()
# print (datalist[0])
# print (datalist[1])
# print (datalist[2])

def parse_log(keyword):
    l_in = [s for s in datalist if keyword in s]
    res = np.zeros([len(l_in),2])
    res_2 = np.zeros([len(l_in),1])
    for i in range(len(l_in)):
        res[i,0] = i
    
    
        a = l_in[i].split(":") 
        a = a[1].split(" ")
        a = a[1].split('",')
        a = a[0].split('\\n')
        # a = a[0].strip()
        a[1] = float(a[0])
        # b = .split(" ")
        # a = b[0]
        res_2[i] = a[1]
    return res_2

cmap = plt.get_cmap("tab20")
fig = plt.figure()
ax = fig.add_subplot(111)
# l_in2 = l_in.split(':')
res_mean = parse_log("vf_loss")
res_max = parse_log("episode_reward_max")
res_min = parse_log("episode_reward_min")
f.close()
i = 0
plt.plot(res_mean, linestyle = "dashed", label = "tb100_sgds50_sgdi_1",color=cmap(i))
plt.plot(res_max,label=test_case,color=cmap(i))
f = open("results_file.pkl",'wb')
res = [res_mean,res_max,res_min]
pickle.dump(res,f)
# pickle.dump(res_max,f)
# pickle.dump(res_min,f)
f.close()

# plt.plot(res_min,label=test_case)
# plt.legend([test_case], bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18,prop={"family":"MS Gothic"})

# i += 1
# f = open("results_file_tb50_sgds1_sgdi_1.pkl", mode="rb")
# tp = pickle.load(f)
# [res_mean,res_max,res_min] = tp
# plt.plot(res_mean[:201], linestyle = "dashed", label = "tb100_sgds50_sgdi_1",color=cmap(i))
# plt.plot(res_max[:201],label=test_case,color=cmap(i))
# f.close()
# ax.legend()

# i += 1
# f = open("results_file_tb100_sgds1_sgdi_10.pkl", mode="rb")
# tp = pickle.load(f)
# [res_mean,res_max,res_min] = tp
# plt.plot(res_mean, linestyle = "dashed", label = "tb100_sgds1_sgdi_10",color=cmap(i))
# plt.plot(res_max,label=test_case,color=cmap(i))
# f.close()
# ax.legend()

# i += 1
# f = open("results_file_tb100_sgds1_sgdi_1.pkl", mode="rb")
# tp = pickle.load(f)
# [res_mean,res_max,res_min] = tp
# plt.plot(res_mean, linestyle = "dashed", label = "tb100_sgds1_sgdi_1",color=cmap(i))
# plt.plot(res_max,label=test_case,color=cmap(i))
# f.close()
# ax.legend()

# i += 1
# f = open("results_file_tb100_sgds100_sgdi_1.pkl", mode="rb")
# tp = pickle.load(f)
# [res_mean,res_max,res_min] = tp
# plt.plot(res_mean, linestyle = "dashed", label = "tb100_sgds100_sgdi_1",color=cmap(i))
# plt.plot(res_max,label=test_case,color=cmap(i))
# f.close()
# ax.legend()

# i += 1
# f = open("results_file_tb100_sgds20_sgdi_20.pkl", mode="rb")
# tp = pickle.load(f)
# [res_mean,res_max,res_min] = tp
# plt.plot(res_mean, linestyle = "dashed", label = "tb100_sgds20_sgdi_20",color=cmap(i))
# plt.plot(res_max,label=test_case,color=cmap(i))
# f.close()
# ax.legend()

plt.grid("on")
plt.savefig("reward_res.png")