#!/usr/bin/env python
# coding: utf-8

# # センサの取得情報を計算する

# In[5]:


import numpy as np
import quaternion


# In[6]:


import sys
sys.path.append('../')
sys.path.append('../utility')


# In[7]:


#座標系クラスのインポート
from utility.COORDINATION_SYSTEMS import ECEF, LLA, NED, FRD, AC_ATTITUDE, LOCAL, SENSOR_ATTITUDE, FOCAL_POSITION, FOCAL_ANGLE


# In[8]:


#WGS84座標変換クラスのインポート
from utility.WGS84_COORDINATE_CONVERT import WGS84_COORDINATION_CONVERT


# In[9]:


#座標変換クラスのインポート
from utility.ATTITUDE_CONVERT import ATTITUDE_CONVERT


# ## クラス定義

# In[10]:


class SENSOR_FOCAL(WGS84_COORDINATION_CONVERT):
    
    def get_focal_distance(self, own_lla = LLA(), opp_lla = LLA()):
        #入力 WGS84 LLA座標
        #出力　レンジ 単位m
        self.own_lla= own_lla #WGS84緯度経度高度座標 単位°とm
        self.opp_lla = opp_lla #WGS84緯度経度高度座標 単位°とm
        
        #誤差
        gosa = 0
        
        distance = self.get_distance(self.own_lla, self.opp_lla) + gosa
        
        return distance  
    
    def get_focal_angle(self, own_lla = LLA(), own_att = AC_ATTITUDE(), sens_att = SENSOR_ATTITUDE(), opp_lla = LLA()):
        #用途：センサ投影角度を算出
        #入力：自機位置 緯度経度高度　単位：°
        #入力：自機姿勢 ロール、ピッチ、ヨー 単位：°
        #入力：センサ視軸 スピン、EL、AZ 単位：°
        #入力：相手機位置 緯度経度高度 単位：°
        #出力：センサ投影角度 投影AZ、投影EL 単位：°（オイラーシーケンス：投影AZ→投影EL）
        
        self.own_lla = own_lla
        self.own_att = own_att
        self.sens_att = sens_att
        self.opp_lla = opp_lla
        
        #緯度経度高度をECEF座標に変換
        own_ecef = self.lla2ecef(own_lla)
        opp_ecef = self.lla2ecef(opp_lla)
        
        #自機を原点においた局所座標
        local = LOCAL()
        local.local_x = opp_ecef.x - own_ecef.x
        local.local_y = opp_ecef.y - own_ecef.y
        local.local_z = opp_ecef.z - own_ecef.z
        
        #局所座標からNED座標（真北、東、下）に変換
        ned = ATTITUDE_CONVERT().local2ned(local, self.own_lla)
        
        #NED座標を機体固定座標（前、右、下）に変換
        frd = ATTITUDE_CONVERT().ned2frd(ned,self.own_att)
        
        #機体固定座標をセンサ固定座標（水平右、垂直下、奥行前）に変換
        focal_position = ATTITUDE_CONVERT().frd2focal(frd, self.sens_att)
        
        #センサ固定座標からセンサ投影角度を算出
        focal_angle = FOCAL_ANGLE()
        focal_angle.focal_az = np.degrees(np.arctan2(focal_position.horizontal , focal_position.depth))
        focal_angle.focal_el = -np.degrees(np.arctan2(focal_position.vertical , np.sqrt(focal_position.horizontal**2 +focal_position.depth**2)))
        return focal_angle
    



