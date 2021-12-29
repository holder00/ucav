#!/usr/bin/env python
# coding: utf-8

# # WGS84でXYZ＠ECEFとLLA(Latitude,Logitude,Altitude)を変換する

# 作成：電シス　浅井(2021.8.18)

# 参考サイト：https://www.enri.go.jp/~fks442/K_MUSEN/1st/1st060428rev2.pdf

# In[1]:


import numpy as np


# ### クラス定義

# In[2]:


class convert_wgs84_rev01:
    def __init__(self):

        self.a = 6378137.0 #赤道面平均半径　単位m
        self.f = 1/298.257223563 #扁平率
        self.e = np.sqrt(2.0*self.f - self.f**2)

    def lla2ecef(self, lat, lon, alt):
        self.lat = lat #緯度　単位°
        self.lon = lon #軽度　単位°
        self.alt = alt #高度　単位m
        
        
        N = self.a / np.sqrt( 1.0- (self.e**2) *np.sin(np.radians(self.lat))**2)
        
        self.x = (N + self.alt)*np.cos(np.radians(self.lat))*np.cos(np.radians(self.lon))
        self.y = (N + self.alt)*np.cos(np.radians(self.lat))*np.sin(np.radians(self.lon))
        self.z = (N*(1.0 - self.e**2) + self.alt)*np.sin(np.radians(self.lat))
        
        #print("self.x=",self.x)
        #print("self.y=",self.y)
        #print("self.z=",self.z)          

        return self.x, self.y, self.z
    
    def ecef2lla(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        e_sq = 2.0*self.f - self.f**2
        e_dot_sq = e_sq/((1.0-self.f)**2)
        b=(1.0-self.f)*self.a
        p= np.sqrt(self.x**2 + self.y**2)
        theta_rad = np.arctan((self.z/p)/(1.0 - self.f))
        
        self.lat = np.degrees(np.arctan((z + e_dot_sq*b*np.sin(theta_rad)**3)/(p - e_sq*self.a*np.cos(theta_rad)**3)))
        self.lon = np.degrees(np.arctan(self.y/self.x))
        
        N = self.a / np.sqrt( 1.0- (self.e**2) *(np.sin(np.radians(self.lat))**2))
        
        self.alt = (p/np.cos(np.radians(self.lat))) - N
        
        #print("e_sq=",e_sq)
        #print("e_dot_seq",e_dot_sq)
        #print("b=",b)
        #print("self.lat=",self.lat)
        #print("self.lon=",self.lon)
        #print("self.alt=",self.alt)

        return self.lat, self.lon, self.alt
    
    def direct_range(self,lat1,lon1,alt1,lat2,lon2,alt2):
        self.lat1 = lat1 #緯度　単位°
        self.lon1 = lon1 #軽度　単位°
        self.alt1 = alt1 #高度　単位m
        self.lat2 = lat2 #緯度　単位°
        self.lon2 = lon2 #軽度　単位°
        self.alt2 = alt2 #高度　単位m
        
        self.x1, self.y1, self.z1 = convert_wgs84_rev01().lla2ecef(self.lat1,self.lon1 ,self.alt1)
        self.x2, self.y2, self.z2 = convert_wgs84_rev01().lla2ecef(self.lat2,self.lon2 ,self.alt2)
        
        self.range_m = np.sqrt((self.x1 - self.x2)**2 + (self.y1 - self.y2)**2 +(self.z1 - self.z2)**2)
        
        return self.range_m    
