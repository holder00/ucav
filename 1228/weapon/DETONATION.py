#!/usr/bin/env python
# coding: utf-8

# # 起爆（信管）モデル

# In[5]:


import sys
sys.path.append('../')
sys.path.append('../utility')
sys.path.append('../sensor')


# In[6]:


import numpy as np
import quaternion
#座標系クラスのインポート
from COORDINATION_SYSTEMS import ECEF, LLA, NED, FRD, AC_ATTITUDE, LOCAL, SENSOR_ATTITUDE, FOCAL_POSITION, FOCAL_ANGLE
#WGS84座標変換クラスのインポート
from WGS84_COORDINATE_CONVERT import WGS84_COORDINATION_CONVERT
#座標変換クラスのインポート
from ATTITUDE_CONVERT import ATTITUDE_CONVERT
#センサ取得情報クラスのインポート
from SENSOR_FOCAL import SENSOR_FOCAL
#センサ性能クラスのインポート
from SENSOR_PERFORMANCE import FOV


# ## クラス定義

# In[7]:


class DETONATION():
    def __init__ (self):
        self.detonated =0
        
    def get_detonated(self, own_lla =LLA(), own_att =AC_ATTITUDE(), sens_att = SENSOR_ATTITUDE(), opp_lla=LLA(), fov = FOV(),fuse_range=0):
        #用途：センサの認知判定を行う。
        #入力：自機位置 緯度経度高度　単位：°　（オイラーシーケンス：経度→緯度）
        #入力：自機姿勢 ロール、ピッチ、ヨー 単位：°（オイラーシーケンス：ヨー→ピッチ→ロール）
        #入力：センサ視軸 スピン、EL、AZ 単位：° （オイラーシーケンス：投影AZ→投影EL）
        #入力：相手機位置 緯度経度高度 単位：°（オイラーシーケンス：経度→緯度）
        #入力：センサ覆域（画像なら視野角） 覆域AZ、覆域EL 単位：°（オイラーシーケンス：覆域AZ→覆域EL）
        #入力：信管作動レンジ(fuse_range)：単位m
        #出力：作動状況 単位：0=非作動 1=作動
        
        #self.own_lla = own_lla
        #self.own_att = own_att
        #self.sens_att = sens_att
        #self.opp_lla = opp_lla
        
        
        #距離判定
        distance = SENSOR_FOCAL().get_focal_distance(own_lla, opp_lla)
        
        
        #視野判定
        focal = SENSOR_FOCAL().get_focal_angle(own_lla, own_att, sens_att, opp_lla)
        
        #recog_level = self.RECOGNITION_LEVEL(detected = 3, oriented = 3, recognized = 3, identified = 3)

        self.detonated = 0
        
        if(distance < fuse_range):
            if(focal.focal_az < fov.az):
                if(focal.focal_el < fov.el):      
                    self.detonated = 1
        
        return  self.detonated




