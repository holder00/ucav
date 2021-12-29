#!/usr/bin/env python
# coding: utf-8

# # レーダー　トイモデル（反射異方性考慮）

# In[5]:


import numpy as np
import quaternion

import sys
sys.path.append('../')
sys.path.append('../utility')

#座標系クラスのインポート
from utility.COORDINATION_SYSTEMS import ECEF, LLA, NED, FRD, AC_ATTITUDE, LOCAL, SENSOR_ATTITUDE, FOCAL_POSITION, FOCAL_ANGLE
#WGS84座標変換クラスのインポート
from utility.WGS84_COORDINATE_CONVERT import WGS84_COORDINATION_CONVERT
#座標変換クラスのインポート
from utility.ATTITUDE_CONVERT import ATTITUDE_CONVERT
#センサ取得情報クラスのインポート
from SENSOR.SENSOR_FOCAL import SENSOR_FOCAL
#センサ性能クラスのインポート
from SENSOR.SENSOR_PERFORMANCE import FOV
#センサモデルクラスのインポート
from SENSOR.SENSOR_MODEL import SENSOR_RECOGNITION


# In[19]:


class RADAR_TOY_MODEL(SENSOR_RECOGNITION):
    def get_arrival_angle_on_opponent(self,own_lla = LLA(), opp_lla =LLA(), opp_att =AC_ATTITUDE()):
        #相手機から見た、RF到来角（AZ、EL）単位°を算出する。

        #緯度経度高度をECEF座標に変換
        own_ecef = WGS84_COORDINATION_CONVERT().lla2ecef(own_lla)
        opp_ecef = WGS84_COORDINATION_CONVERT().lla2ecef(opp_lla)
        
        #相手機を原点においた局所座標
        opp_local = LOCAL()
        opp_local.local_x = own_ecef.x - opp_ecef.x
        opp_local.local_y = own_ecef.y - opp_ecef.y
        opp_local.local_z = own_ecef.z - opp_ecef.z
        
        #相手機原点の機体局所座標から相手機原点のNED座標（真北、東、下）に変換
        opp_ned = ATTITUDE_CONVERT().local2ned(opp_local, opp_lla)
        
        #相手機原点のNED座標を相手機原点の機体固定座標（前、右、下）に変換
        opp_frd = ATTITUDE_CONVERT().ned2frd(opp_ned,opp_att)
        
        #相手機原点の機体固定座標（前、右、下）で、
        arrival_angle = FOCAL_ANGLE()
        arrival_angle.focal_az = np.degrees(np.arctan2(opp_frd.rh , opp_frd.fwd))
        arrival_angle.focal_el = -np.degrees(np.arctan2(opp_frd.dwn , np.sqrt(opp_frd.rh**2 +opp_frd.fwd**2)))
        
        return arrival_angle
    
    def get_reflection_effect_nonphysical_unit(self,arraival_angle = FOCAL_ANGLE()):
        #非物理単位での反射効果を表す。単位はなし。
        #センサレンジを伸縮することで、反射の影響を表現する。
        #側方を基準1
        #前後をeffect_az=0.5、上下をeffect_el=2,それ以外effect_elとeffect_azともに1
        #全体効果はeffect = effect_az * effect_el
        
        effect_on_az =1
        effect_on_el = 1
        
        #AZ方向
        if(arraival_angle.focal_az > 45):
            if(arraival_angle.focal_az < 135):
                effect_on_az =0.5
            
        if(arraival_angle.focal_az > 225):
            if(arraival_angle.focal_az < 315):
                effect_on_az =0.5
        
        #EL方向
        if(arraival_angle.focal_el < 45):
            effect_on_el =2
            
        if(arraival_angle.focal_el > 135):
            if(arraival_angle.focal_el < 225):
                effect_on_el =2
        
        if(arraival_angle.focal_el > 315):
            effect_on_el =2
            
        effect = effect_on_az * effect_on_el
        
        return effect
    
    def get_rf_detection_level(self,
                               own_lla = LLA(0,0,0), 
                               own_att = AC_ATTITUDE(0,0,0), 
                               sens_att =SENSOR_ATTITUDE(0,0,0),
                               opp_lla =LLA(0,0,0),
                               opp_att =AC_ATTITUDE(0,0,0),
                               sensor_range =0,
                               fov=FOV(0,0,0)):
        arrival_angle = self.get_arrival_angle_on_opponent(own_lla, opp_lla, opp_att)
        effect = self.get_reflection_effect_nonphysical_unit(arrival_angle)
        # masu変更　0930
        # クラス変数にし、uavで参照用
        self.sensor_range_effected = effect* sensor_range
        self.detected, _, _,_ =self.get_recognition_level(own_lla, own_att,sens_att,opp_lla,fov, self.sensor_range_effected)


# In[20]:


#radar = RADAR_TOY_MODEL()

#own_lla = LLA(36.0,135.1,10000) #自機位置
#own_att = AC_ATTITUDE(0,0,0) #自機姿勢
#sens_att =SENSOR_ATTITUDE(0,80,0) #センサ視軸
#opp_lla = LLA(36,135,12000) #相手機位置
#opp_att = AC_ATTITUDE(0,0,0) #自機姿勢
#sensor_fov=FOCAL_ANGLE(0,0) #センサ覆域
#fov=FOV(0,0,0)
#sensor_range = 10000 #センサレンジ　単位m

#radar.get_arrival_angle_on_opponent(own_lla,opp_lla,opp_att)
#radar.get_rf_detection_level(own_lla, own_att,sens_att, opp_lla, opp_att,sensor_range,fov)
#print("detected",radar.detected)

