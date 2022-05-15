#!/usr/bin/env python
# coding: utf-8

# # 相手機の距離(m)と方位(AZ,EL)を計算する

# 必要なパッケージ

# In[5]:


import sys
sys.path.append('../')

import numpy as np
import quaternion


# インポートする自作クラス

# In[6]:


from attitude_quat import attitude_conv_rev01
from wgs84 import convert_wgs84_rev01


# クラス定義

# In[3]:


class focal_location_rev01:
    def __init__(self):
        self.own_lat =0; self.own_lon=0; self.own_alt=0
        self.own_roll =0; self.own_pitch=0; self.own_yaw=0
        self.sensor_spin =0; self.sensor_el=0; self.sensor_az=0
        
        self.opponent_lat =0; self.opponent_lon =0; self.opponent_alt =0
        self.opponent_roll =0; self.opponent_pitch =0; self.opponent_yaw =0
        
    def get_focal_distance(self,
                  own_lat, own_lon, own_alt,
                  opponent_lat, opponent_lon, opponent_alt):
        #用途：相手までの直線距離を算出。単位はm
        #入力：自機位置 緯度経度高度　単位：°
        #入力：相手機位置 緯度経度高度 単位：°
        #出力：直線距離 単位：m
        
        self.own_lat = own_lat; self.own_lon=own_lon; self.own_alt=own_alt        
        self.opponent_lat = opponent_lat; self.opponent_lon = opponent_lon; self.opponent_alt = opponent_alt
        
        sensed_range = convert_wgs84_rev01().direct_range(self.own_lat,self.own_lon,self.own_alt, self.opponent_lat,self.opponent_lon,self.opponent_alt)
        
        return sensed_range
    
    def get_focal_angle(self,
                              own_lat, own_lon, own_alt,
                              own_roll, own_pitch, own_yaw,
                              sensor_spin, sensor_el, sensor_az,
                              opponent_lat, opponent_lon, opponent_alt):
        #用途：センサ投影角度を算出
        #入力：自機位置 緯度経度高度　単位：°
        #入力：自機姿勢 ロール、ピッチ、ヨー 単位：°
        #入力：センサ視軸 スピン、EL、AZ 単位：°
        #入力：相手機位置 緯度経度高度 単位：°
        #出力：センサ投影角度 投影AZ、投影EL 単位：°（オイラーシーケンス：投影AZ→投影EL）
        
        self.own_lat = own_lat; self.own_lon=own_lon; self.own_alt=own_alt
        self.own_roll = own_roll; self.own_pitch = own_pitch; self.own_yaw = own_yaw
        self.sensor_spin =sensor_spin; self.sensor_el = sensor_el; self.sensor_az = sensor_az
        self.opponent_lat = opponent_lat; self.opponent_lon = opponent_lon; self.opponent_alt = opponent_alt
        
        #緯度経度高度をECEF座標に変換
        own_x, own_y, own_z = convert_wgs84_rev01().lla2ecef(own_lat, own_lon, own_alt)
        opponent_x,opponent_y,opponent_z = convert_wgs84_rev01().lla2ecef(opponent_lat, opponent_lon, opponent_alt)
        
        #自機を原点においた局所座標
        local_x = opponent_x - own_x
        local_y = opponent_y - own_y
        local_z = opponent_z - own_z
        
        #局所座標からNED座標（真北、東、下）に変換
        north, east, below = attitude_conv_rev01().local2ned(local_x,local_y,local_z,own_lat, own_lon)
        
        #NED座標を機体固定座標（前、右、下）に変換
        fwd, rh, dwn = attitude_conv_rev01().ned2frd(north,east,below,own_roll, own_pitch, own_yaw)
        
        #機体固定座標をセンサ固定座標（水平右、垂直下、奥行前）に変換
        focal_hor, focal_ver, focal_dpt = attitude_conv_rev01().frd2focal(fwd, rh, dwn, sensor_spin, sensor_el, sensor_az)
        
        #センサ固定座標からセンサ投影角度を算出
        focal_az = np.degrees(np.arctan(focal_hor / focal_dpt))
        focal_el = -np.degrees(np.arctan(focal_ver / np.sqrt(focal_hor**2 +focal_dpt**2)))
        return focal_az, focal_el
        

        
