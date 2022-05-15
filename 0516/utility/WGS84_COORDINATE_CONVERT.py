#!/usr/bin/env python
# coding: utf-8

# # WGS84座標変換

# 作成：電シス 浅井2021.8.30

# In[1]:


import numpy as np
import quaternion


# In[3]:


import sys
sys.path.append('../')


# In[4]:


from utility.COORDINATION_SYSTEMS import ECEF, LLA, NED, FRD, AC_ATTITUDE, LOCAL, SENSOR_ATTITUDE, FOCAL_POSITION, FOCAL_ANGLE


# ## クラス定義

# In[5]:


class WGS84_COORDINATION_CONVERT:
    #WGS84での座標変換
    def __init__(self):
        self.a = 6378137.0 #赤道面平均半径　単位m
        self.f = 1/298.257223563 #扁平率
        self.e = np.sqrt(2.0*self.f - self.f**2)
    
    def lla2ecef(self, lla = LLA()):
        #原理：https://www.enri.go.jp/~fks442/K_MUSEN/1st/1st060428rev2.pdf
        self.lla = lla
        ecef = ECEF()
        N = self.a / np.sqrt( 1.0- (self.e**2) *np.sin(np.radians(self.lla.lat))**2)
        
        ecef.x = (N + self.lla.alt)*np.cos(np.radians(self.lla.lat))*np.cos(np.radians(self.lla.lon))
        ecef.y = (N + self.lla.alt)*np.cos(np.radians(self.lla.lat))*np.sin(np.radians(self.lla.lon))
        ecef.z = (N*(1.0 - self.e**2) + self.lla.alt)*np.sin(np.radians(self.lla.lat))
        
        #print(ecef.x, ecef.y, ecef.z)
        
        return ecef

    def ecef2lla(self, ecef = ECEF()):
        #原理：https://www.enri.go.jp/~fks442/K_MUSEN/1st/1st060428rev2.pdf
        self.ecef = ecef
        
        lla = LLA()
        
        e_sq = 2.0*self.f - self.f**2
        e_dot_sq = e_sq/((1.0-self.f)**2)
        b=(1.0-self.f)*self.a
        
        p= np.sqrt(self.ecef.x**2 + self.ecef.y**2)
        theta_rad = np.arctan((self.ecef.z/p)/(1.0 - self.f))
        
        #メモ逆正弦の地域が[-pi,pi]となるようにarctan2を使っている。
        lla.lat = np.degrees(np.arctan2(self.ecef.z + e_dot_sq*b*np.sin(theta_rad)**3, p - e_sq*self.a*np.cos(theta_rad)**3))
        lla.lon = np.degrees(np.arctan2(self.ecef.y, self.ecef.x))
        
        N = self.a / np.sqrt( 1.0- (self.e**2) *(np.sin(np.radians(lla.lat))**2))
        
        lla.alt = (p/np.cos(np.radians(lla.lat))) - N
        print(lla.lat,lla.lon,lla.alt)
        return lla
    
    def get_distance(self, lla1 = LLA(), lla2 = LLA()):
        #入力 WGS84 LLA座標
        #出力　レンジ 単位m
        self.lla1 = lla1 #WGS84緯度経度高度座標 単位°とm
        self.lla2 = lla2 #WGS84緯度経度高度座標 単位°とm
        
        ecef1 =  self.lla2ecef(self.lla1)
        ecef2 =  self.lla2ecef(self.lla2)
                
        distance = np.sqrt((ecef1.x - ecef2.x)**2 + (ecef1.y - ecef2.y)**2 +(ecef1.z - ecef2.z)**2)
        
        return distance  

