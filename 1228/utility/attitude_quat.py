#!/usr/bin/env python
# coding: utf-8

# # 姿勢演算　クォータニオン操作

# ### 注意<br>
# 準備としてnumpy-quaternionをインストールしておく<br>
# !pip install numpy-quaternion

# 参考サイト：https://qiita.com/momomo_rimoto/items/3a245736c5fd90fe8270

# In[1]:


import sys
sys.path.append('../')


# In[10]:


import numpy as np
import quaternion


# In[11]:


class attitude_conv_rev01:
    #準備としてnumpy-quaternionをインストールしておくこと。
    def __init__(self):
        self.frd = np.array([0,0,0])
        self.ned = np.array([0,0,0])
        self.local = np.array([0,0,0])
        self.focal = np.array([0,0,0])
        
    def ned2frd(self, north,east,below,roll,pitch,yaw):
        #入力：　北、東、南 ：単位m
        #入力：ロール、ピッチ、ヨ― ：単位°
        #出力： 前、右、下 単位m
        #オイラーシーケンス：ヨー→ピッチ→ロール
        
        self.ned = np.array([north,east,below])
        q1=quaternion.from_rotation_vector([0,0,np.radians(yaw)])
        q2=quaternion.from_rotation_vector([0,np.radians(pitch),0])
        q3=quaternion.from_rotation_vector([np.radians(roll),0,0])

        quat=q3.inverse()*q2.inverse()*q1.inverse()
        rot = quaternion.as_rotation_matrix(quat)
        self.frd = np.matmul(rot, self.ned)
        
        self.fwd,self.rh,self.dwn = self.frd
        
        return self.fwd,self.rh,self.dwn
    
    
    def frd2ned(self, fwd,rh,dwn,roll,pitch,yaw):
        #入力：　前、右、下 単位m
        #入力：ロール、ピッチ、ヨ― 単位°
        #出力： 北、東、南 ：単位m
        #オイラーシーケンス：ヨー→ピッチ→ロール
        
        self.frd = np.array([fwd,rh,dwn])
               
        q1=quaternion.from_rotation_vector([0,0,np.radians(yaw)])
        q2=quaternion.from_rotation_vector([0,np.radians(pitch),0])
        q3=quaternion.from_rotation_vector([np.radians(roll),0,0])
        
        quat=q1*q2*q3

        rot = quaternion.as_rotation_matrix(quat)
        self.ned = np.matmul(rot, self.frd)
        
        self.north,self.east,self.below = self.ned
        
        return self.north,self.east,self.below
    
    def local2ned(self, x_loc, y_loc, z_loc, ido, keido):
        #入力：　局所X、局所Y、局所Z 単位m
        #入力：緯度、経度　単位°
        #出力： 北、東、南 ：単位m
        #オイラーシーケンス：経度→緯度
        
        self.xyz=np.array([x_loc,y_loc,z_loc])
        
        q1=quaternion.from_rotation_vector([0,np.radians(-90),0])
        q2=quaternion.from_rotation_vector([np.radians(keido),0,0])
        q3=quaternion.from_rotation_vector([0,np.radians(-ido),0])
        
        quat = q3.inverse()*q2.inverse()*q1.inverse()
        
        rot = quaternion.as_rotation_matrix(quat)
        
        self.ned = np.matmul(rot, self.xyz)
        self.north,self.east,self.below = self.ned
        
        return self.north,self.east,self.below
    
    def ned2local(self, north, east, below, ido, keido):
        #入力：北、東、南 ：単位m　
        #入力：緯度、経度　単位°
        #出力： 局所X、局所Y、局所Z 単位m
        #オイラーシーケンス：経度→緯度
        
        self.ned = np.array([north,east,below])

        q1=quaternion.from_rotation_vector([0,np.radians(-90),0])
        q2=quaternion.from_rotation_vector([np.radians(keido),0,0])
        q3=quaternion.from_rotation_vector([0,np.radians(-ido),0])
        
        quat=q1*q2*q3
        
        rot = quaternion.as_rotation_matrix(quat)
        
        self.local = np.matmul(rot, self.ned)
        self.x_loc, self.y_loc, self.z_loc = self.local
        
        return self.x_loc, self.y_loc, self.z_loc
    
    def frd2focal(self, fwd, rh, dwn, spin, el, az):
        #入力：前、右、下 ：単位m　
        #入力：アジマス、エレベーション、スピン　：単位°
        #出力： 水平右、垂直下、奥行前 ：単位m
        #オイラーシーケンス：アジマス→エレベーション→スピン
        self.frd = np.array([fwd,rh,dwn])
        
        q1=quaternion.from_rotation_vector([0,0,np.radians(az)])
        q2=quaternion.from_rotation_vector([0,np.radians(el),0])
        q3=quaternion.from_rotation_vector([np.radians(spin),0,0])
        q4=quaternion.from_rotation_vector([0,0,np.radians(90)])
        q5=quaternion.from_rotation_vector([np.radians(90),0,0])

        quat=q5.inverse()*q4.inverse()*q3.inverse()*q2.inverse()*q1.inverse()
        rot = quaternion.as_rotation_matrix(quat)
        self.focal = np.matmul(rot, self.frd)
        
        self.focal_hor,self.focal_ver,self.focal_dep = self.focal
        
        return self.focal_hor,self.focal_ver,self.focal_dep
    
    def focal2frd(self, focal_hor, focal_ver, focal_dep, spin, el, az):
        #入力： 水平右、垂直下、奥行前 ：単位m
        #入力：アジマス、エレベーション、スピン　：単位°
        #出力：前、右、下 ：単位m　
        #オイラーシーケンス：アジマス→エレベーション→スピン
        
        self.focal = np.array([focal_hor,focal_ver,focal_dep])
        
        q1=quaternion.from_rotation_vector([0,0,np.radians(az)])
        q2=quaternion.from_rotation_vector([0,np.radians(el),0])
        q3=quaternion.from_rotation_vector([np.radians(spin),0,0])
        q4=quaternion.from_rotation_vector([0,0,np.radians(90)])
        q5=quaternion.from_rotation_vector([np.radians(90),0,0])

        quat=q1*q2*q3*q4*q5
        rot = quaternion.as_rotation_matrix(quat)
        self.frd = np.matmul(rot, self.focal)
        
        self.fwd,self.rh,self.dwn = self.frd
        
        return self.fwd,self.rh,self.dwn
    

