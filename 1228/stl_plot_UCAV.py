# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:43:40 2021

@author: ookit1
"""

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

def mesh_scale(my_mesh, scale_x, scale_y, scale_z):
    my_mesh.x = my_mesh.x * scale_x
    my_mesh.y = my_mesh.y * scale_y
    my_mesh.z = my_mesh.z * scale_z 
    return my_mesh

# 描画領域を新規作成
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# STLファイルを読み込み、メッシュデータからプロットデータに変換
your_mesh = mesh.Mesh.from_file('UCAV.stl')
mesh_scale(your_mesh,0.01,0.01,0.01)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# 大きさを自動調整
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# 表示
pyplot.show()