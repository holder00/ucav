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
        steps = 7#56#61
        hist_dir = './' + "UCAV" + '/hist/'
        f_name_b = hist_dir+str(steps)+"_"+"blue_"+"hist"+".pkl"
        f_name_r = hist_dir+str(steps)+"_"+"red_"+"hist"+".pkl"
        f_name_m = hist_dir+str(steps)+"_"+"mrm_"+"hist"+".pkl"
        
        test_num = 2
        test_num = str(test_num)+"_"
        f_name_b = hist_dir+str(steps)+"_"+"blue_"+test_num+"hist"+".pkl"
        f_name_r = hist_dir+str(steps)+"_"+"red_"+test_num+"hist"+".pkl"
        f_name_m = hist_dir+str(steps)+"_"+"mrm_"+test_num+"hist"+".pkl"
        
        f = open(f_name_b,mode="rb")
        self.blue_hist = pickle.load(f)
        f.close()
        
        f = open(f_name_r,mode="rb")
        self.red_hist = pickle.load(f)
        f.close()
        f = open(f_name_m,mode="rb")
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
        self.control_frame.pack(side=tk.LEFT)
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(side=tk.LEFT)
        self.radar_text = tk.StringVar()
        self.radar_text.set("Radar_switch")
        
        # self.button_frame = tk.Frame(self.master)
        self.button = tk.Button(self.button_frame, text="toggle_view",command=self.play_plot)
        self.button.pack(anchor=tk.SE, padx=5,pady=5)
        self.topview_button = tk.Button(self.button_frame, text="top_view",command=self.topview)
        self.topview_button.pack(anchor=tk.SE, padx=5,pady=5)
        self.zoom_button = tk.Button(self.button_frame, text="zoom_toggle",command=self.zoom_toggle)
        self.zoom_button.pack(anchor=tk.SE, padx=5,pady=5)
        self.radar_button = tk.Button(self.button_frame, text=self.radar_text.get(),command=self.radar_switch)
        self.radar_button.pack(anchor=tk.SE, padx=5,pady=5)

        self.canvas = FigureCanvasTkAgg(fig1, self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        # self.toolbar.update()
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.x_v = tk.DoubleVar()
        self.x_scale = tk.Scale(self.control_frame,
            variable=self.x_v,
            length = 500,           # ???????????????
            from_=0,
            to=self.time_max,
            resolution=1,
            orient=tk.HORIZONTAL,
            command=self.draw_plot,
            width = 20,             # ???????????????
            sliderlength = 20,      # ????????????????????????????????????
            tickinterval=self.time_max         # ?????????????????????(?????????0???????????????)
            )
        
        self.x_scale.pack(anchor=tk.NW)

        self.y_v = tk.DoubleVar()
        self.y_scale = tk.Scale(self.control_frame,
            variable=self.y_v,
            length = 500,           # ???????????????
            from_=1,
            to=len(self.blue_hist),
            resolution=10,
            orient=tk.HORIZONTAL,
            command=self.draw_plot,
            width = 20,             # ???????????????
            sliderlength = 20,      # ????????????????????????????????????
            tickinterval=self.time_max         # ?????????????????????(?????????0???????????????)
            )
        self.y_scale.pack(anchor=tk.NW)

    def start_up(self):
        self.x_v.set(1.0)
        self.y_v.set(50.0)
        self.zoom = True
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
        
    def zoom_toggle(self):

        if self.zoom:
            self.zoom = False
        else:
            self.zoom = True

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
                if self.zoom and faction !="mrm":
                    self.xlim,self.ylim = self.limit_calc([self.xlim,self.ylim], hist[-1,1+i*status_num:3+i*status_num],hist[-1,12+i*status_num])
                    # self.zlim = self.limit_calc(self.zlim, hist[-1,3])
                 

            ax.view_init(elev=self.elev, azim=self.azim)
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_zlim(self.zlim)
            self.canvas.draw()  
            # ax.set_xticks(self.xticks)
            # ax.set_yticks(self.yticks)
            # ax.set_zticks(self.zticks)
        plt.subplots_adjust(left=-0.1,right=1.1,bottom=-0.1,top=1.1)
        
    def limit_calc(self,lim,val,hitpoint):

        if hitpoint > 0:
            val = val/1000
            if self.first:
                self.first = False
                self.temp = [[val[0],val[0]],[val[1],val[1]]]
            for i in range(2):
                if 0 <= val[i] and val[i] <= self.temp[i][0]:
                    self.temp[i][0] = val[i]-10
                    
                if self.temp[i][1] <= val[i] and val[i] <= self.info["WINDOW_SIZE_lat"]/1000:
                    self.temp[i][1] = val[i]+10
                    
            if self.temp[0][1]-self.temp[0][0] > self.temp[1][1]-self.temp[1][0]:
                
                lim_scale = self.temp[0][1]-self.temp[0][0]
                self.temp[1][0] = (self.temp[1][1]+self.temp[1][0]-lim_scale)/2
                self.temp[1][1] = (self.temp[1][1]+self.temp[1][0]+lim_scale)/2
                
            else:
                lim_scale = self.temp[1][1]-self.temp[1][0]
                self.temp[0][0] = (self.temp[0][1]+self.temp[0][0]-lim_scale)/2
                self.temp[0][1] = (self.temp[0][1]+self.temp[0][0]+lim_scale)/2

        return self.temp[0],self.temp[1]
        
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
        self.first = True
        
 

f = open("info"+".pkl",mode="rb")
info = pickle.load(f)
f.close()
plt.ioff()
ESC = 0x1B          # ESC??????????????????????????????

# plt.ion()           # ?????????????????????
fig1 = plt.figure(1,figsize=(8.0, 6.0))

# fig1 = Figure(figsize=(5, 5), dpi=100)
# ax = fig.add_subplot(111)
ax = fig1.gca(projection='3d')



# ax, = ax.plot([],[], 'green')

root = tk.Tk()
app = Application(master=root)
app.mainloop()