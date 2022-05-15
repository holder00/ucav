# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 22:21:54 2022

@author: Takumi
"""
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utility.result_env import render_env
# from result_env import render_env
import warnings
import matplotlib
import time
import ctypes
import threading

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        steps = 61
        f = open(str(steps)+"_"+"blue"+"_hist"+".pkl",mode="rb")
        self.blue_hist = pickle.load(f)
        f.close()
        f = open(str(steps)+"_"+"red"+"_hist"+".pkl",mode="rb")
        self.red_hist = pickle.load(f)
        f.close()
        f = open(str(steps)+"_"+"mrm"+"_hist"+".pkl",mode="rb")
        self.mrm_hist = pickle.load(f)
        f.close()
        f = open("info"+".pkl",mode="rb")
        self.info = pickle.load(f)
        f.close()
        self.time_max = self.mrm_hist.shape[0]

        self.radar_plot = True
        self.master = master
        self.master.title('Simulation Result')
        self.pack()
        self.create_widgets()
        self.start_up()
        self.draw_plot()
        

    def create_widgets(self):
        self.canvas_frame = tk.Frame(self.master)
        self.canvas_frame.pack(side=tk.TOP)
        self.control_frame = tk.Frame(self.master)
        self.control_frame.pack(side=tk.BOTTOM)
        self.radar_text = tk.StringVar()
        self.radar_text.set("Radar_switch")
        
        # self.button_frame = tk.Frame(self.master)
        self.button = tk.Button(self.control_frame, text="toggle_view",command=self.play_plot)
        self.button.pack(side=tk.RIGHT, padx=5,pady=5)
        self.topview_button = tk.Button(self.control_frame, text="top_view",command=self.topview)
        self.topview_button.pack(side=tk.RIGHT, padx=5,pady=5)
        self.radar_button = tk.Button(self.control_frame, text=self.radar_text.get(),command=self.radar_switch)
        self.radar_button.pack(side=tk.RIGHT, padx=5,pady=5)

        self.canvas = FigureCanvasTkAgg(fig1, self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        # self.toolbar.update()
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.x_v = tk.DoubleVar()
        self.x_scale = tk.Scale(self.control_frame,
            variable=self.x_v,
            length = 500,           # 全体の長さ
            from_=0,
            to=self.time_max,
            resolution=1,
            orient=tk.HORIZONTAL,
            command=self.draw_plot,
            width = 20,             # 全体の太さ
            sliderlength = 20,      # スライダー（つまみ）の幅
            tickinterval=self.time_max         # 目盛りの分解能(初期値0で表示なし)
            )
        
        self.x_scale.pack(anchor=tk.NW)

        self.y_v = tk.DoubleVar()
        self.y_scale = tk.Scale(self.control_frame,
            variable=self.y_v,
            length = 500,           # 全体の長さ
            from_=1,
            to=len(self.blue_hist),
            resolution=10,
            orient=tk.HORIZONTAL,
            command=self.draw_plot,
            width = 20,             # 全体の太さ
            sliderlength = 20,      # スライダー（つまみ）の幅
            tickinterval=self.time_max         # 目盛りの分解能(初期値0で表示なし)
            )
        self.y_scale.pack(anchor=tk.NW)

    def start_up(self):
        self.x_v.set(1.0)
        self.y_v.set(50.0)
        self.topview()
        # self.elev = 90
        # self.azim = -90
        # self.xlim = (0,self.info["WINDOW_SIZE_lat"]/1000)
        # self.ylim = (0,self.info["WINDOW_SIZE_lon"]/1000)
        # self.zlim = (0,self.info["WINDOW_SIZE_alt"]/1000)
        # self.xticks = np.linspace(0,self.info["WINDOW_SIZE_lat"]/1000,9)
        # self.yticks = np.linspace(0,self.info["WINDOW_SIZE_lon"]/1000,9)
        # self.zticks = np.linspace(0,self.info["WINDOW_SIZE_alt"]/1000,9)

    def topview(self):
        self.elev = 90
        self.azim = -90
        self.xlim = (0,self.info["WINDOW_SIZE_lat"]/1000)
        self.ylim = (0,self.info["WINDOW_SIZE_lon"]/1000)
        self.zlim = (0,self.info["WINDOW_SIZE_alt"]/1000)
        self.xticks = np.linspace(0,self.info["WINDOW_SIZE_lat"]/1000,9)
        self.yticks = np.linspace(0,self.info["WINDOW_SIZE_lon"]/1000,9)
        self.zticks = np.linspace(0,self.info["WINDOW_SIZE_alt"]/1000,9)
        self.draw_plot()
        
    def plotter(self, t, hist_t, status_num, faction, trajec,radar_plot):
        global ax, plt,axx

        if faction == "mrm":
            color_f = "k"    
        elif faction == "blue":
            color_f = "b"
        elif faction == "red":
            color_f = "r"
        
        if t == 1:
            num = int((np.shape(hist_t)[0])/status_num)
            for i in range(num):
                hist = hist_t[1]
                ax = render_env.rend_3d_mod3(t,hist[i*status_num:(i+1)*status_num],color_f,1,self.info,i,trajec,radar_plot)
                # ax.view_init(elev=self.elev, azim=self.azim)
                # ax.set_xlim(self.xlim)
                # ax.set_ylim(self.ylim)
                # ax.set_zlim(self.zlim)
        else:
            
            num = int((np.shape(hist_t)[1])/status_num)
            for i in range(num):
                hist = hist_t[0:int(t),:]
                
                ax = render_env.rend_3d_mod3(t,hist[:,i*status_num:(i+1)*status_num],color_f,1,self.info,i,trajec,radar_plot)

                self.canvas.draw()   

            ax.view_init(elev=self.elev, azim=self.azim)
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_zlim(self.zlim)
            # ax.set_xticks(self.xticks)
            # ax.set_yticks(self.yticks)
            # ax.set_zticks(self.zticks)
        plt.subplots_adjust(left=-0.1,right=1.1,bottom=-0.1,top=1.1)

        
    def play_plot(self, event=None):
        # global ax, plt

        self.xlim, self.ylim, self.zlim, self.elev, self.azim = fig1.get_axes()[0]._get_view()
        # print(self.xlim, self.ylim, self.zlim, self.elev, self.azim)


           
            
    def getkey(self,key):
        return(bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000))     

    def radar_switch(self, event=None):
        # global ax, plt
        self.xlim, self.ylim, self.zlim, self.elev, self.azim = fig1.get_axes()[0]._get_view()
        if self.radar_plot:
            self.radar_plot = False
        else:
            self.radar_plot = True
        self.draw_plot()

    def draw_plot(self, event=None):
        # global ax, plt
        plt.clf()
        
        status_num = 13
        t = self.x_v.get()
        w = self.y_v.get()
        t = int(t)
        if t < 1:
            t = 1
        
        self.plotter(t, self.mrm_hist, status_num, "mrm", w,self.radar_plot)
        self.plotter(t, self.red_hist, status_num, "red", w,self.radar_plot)
        self.plotter(t, self.blue_hist, status_num, "blue", w,self.radar_plot)
        
        
 

f = open("info"+".pkl",mode="rb")
info = pickle.load(f)
f.close()
plt.ioff()
ESC = 0x1B          # ESCキーの仮想キーコード

# plt.ion()           # 対話モードオン
fig1 = plt.figure(1,figsize=(8.0, 6.0))

# fig1 = Figure(figsize=(5, 5), dpi=100)
# ax = fig.add_subplot(111)
ax = fig1.gca(projection='3d')



# ax, = ax.plot([],[], 'green')

root = tk.Tk()
app = Application(master=root)
app.mainloop()