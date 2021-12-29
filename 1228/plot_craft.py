# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:14:50 2021

@author: ookit1
"""
import pyvista as pv

class plot_craft:  
    def plot_craft(mesh, pos,att):
        
        # rot = mesh.copy()
        rot = mesh
        rot.rotate_x(att[0], point=mesh.center)
        rot.rotate_y(att[1], point=mesh.center)
        rot.rotate_z(att[2], point=mesh.center)
        mesh.translate((pos[0],pos[1],pos[2]))
        return rot
