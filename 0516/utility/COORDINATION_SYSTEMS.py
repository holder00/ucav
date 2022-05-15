#!/usr/bin/env python
# coding: utf-8

# # 座標系クラス

# 作成：電シス 浅井2021.8.30

# In[1]:


class ECEF:
    #地球中心地球固定座標
    def __init__(self, x=0, y=0, z=0):
        self.x = x #地球中心原点、子午線方向　単位m
        self.y = y #地球中心原点、x及びzと直交　右ネジ方向　単位m
        self.z = z #地球中心原点、自転軸方向　単位m


# In[2]:


class LLA:
    #緯度経度高度座標
    def __init__(self, lat =0, lon =0, alt =0):
        self.lat = lat #緯度　単位°
        self.lon = lon #経度　単位°
        self.alt = alt #高度　単位m
        #オイラーシーケンス：経度→緯度


# In[3]:


class NED:
    #機体局所座標
    def __init__(self, north=0, east=0, down=0):
        self.north = north #機体原点、真北方向　単位m
        self.east = east #機体原点、真東方向　単位m
        self.down = down #機体原点、垂直真下　単位m


# In[4]:


class FRD:
    #機体固定座標
    def __init__(self, fwd =0, rh =0, dwn=0):
        self.fwd = fwd #機体原点、前方　単位m
        self.rh = rh #機体原点、右方　単位m
        self.dwn = dwn #機体原点、下方　単位m


# In[5]:


class LOCAL:
    #機体原点、機体局所座標系
    def __init__(self, local_x =0, local_y =0, local_z =0):
        self.local_x = local_x
        self.local_y = local_y
        self.local_z = local_z


# In[6]:


class AC_ATTITUDE:
    #機体姿勢
    def __init__(self, roll=0, pitch =0, yaw =0):
        self.roll = roll #ロール 単位°
        self.pitch = pitch #ピッチ 単位°
        self.yaw = yaw #ヨー　単位°
        #オイラーシーケンス：ヨー→ピッチ→ロール


# In[7]:


class SENSOR_ATTITUDE:
    #センサ角度
    def __init__(self, spin =0, el =0, az =0):
        self.spin = spin #スピン 単位°
        self.el = el #エレベーション 単位°
        self.az = az #アジマス 単位°
        #オイラーシーケンス：アジマス→エレベーション→スピン


# In[8]:


class FOCAL_POSITION:
    #センサ固定座標
    def __init__(self, horizontal=0, vertical=0, depth=0):
        self.horizontal = horizontal #水平右　単位m
        self.vertical = vertical #垂直下　単位m
        self.depth = depth #奥行前　単位m


# In[9]:


class FOCAL_ANGLE:
    #センサ取得角度
    def __init__(self, focal_el=0, focal_az =0):
        self.focal_el = focal_el #センサ取得角度水平右方向　単位°
        self.focal_az = focal_az #センサ取得角度仰角　単位°


# In[ ]:




