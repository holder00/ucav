# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:19:25 2021

@author: ookit1
"""
import pyvista as pv
from plot_craft import plot_craft


filename = 'UCAV.stl'
mesh = pv.read(filename)

# for i in range(100):
# print(i)
pos = [10000,0,0]
att = [0,30,0]
mesh1 = plot_craft.plot_craft(mesh,pos,att)

pos = [10000,0,0]
att = [0,30,0]
mesh2 = plot_craft.plot_craft(mesh,pos,att)
plotter = pv.Plotter()
plotter.show(auto_close=False)

plotter.add_mesh(mesh1)
plotter.add_mesh(mesh2)
plotter.show()
for i in range(100):
    pos = [10000+1000*i,0,0]
    att = [0,30,0]
    mesh1 = plot_craft.plot_craft(mesh,pos,att)
    plotter.update(1, force_redraw=True)
    