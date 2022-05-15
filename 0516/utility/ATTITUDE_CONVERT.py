#!/usr/bin/env python
# coding: utf-8

# # ATTITUDE_CONVERT

# 作成：電シス 浅井2021.8.30

# 参考サイト：https://qiita.com/momomo_rimoto/items/3a245736c5fd90fe8270

# In[3]:


import numpy as np
import quaternion

import sys
# sys.path.append('../')
# sys.path.append('../sensor')


# In[4]:


#座標系クラスのインポート
from utility.COORDINATION_SYSTEMS import ECEF, LLA, NED, FRD, AC_ATTITUDE, LOCAL, SENSOR_ATTITUDE, FOCAL_POSITION, FOCAL_ANGLE


# In[5]:


#WGS84座標変換クラスのインポート
from utility.WGS84_COORDINATE_CONVERT import WGS84_COORDINATION_CONVERT


# ## クラス定義

# In[6]:


class ATTITUDE_CONVERT:
    def ned2frd(self, ned = NED(), ac_attitude = AC_ATTITUDE()):
        #入力：　北、東、南 ：単位m
        #入力：ロール、ピッチ、ヨ― ：単位°
        #出力： 前、右、下 単位m
        #オイラーシーケンス：ヨー→ピッチ→ロール
        self.ned = ned
        
        self.ac_attitude = ac_attitude        
        q1=quaternion.from_rotation_vector([0,0,np.radians(self.ac_attitude.yaw)])
        q2=quaternion.from_rotation_vector([0,np.radians(self.ac_attitude.pitch),0])
        q3=quaternion.from_rotation_vector([np.radians(self.ac_attitude.roll),0,0])

        quat=q3.inverse()*q2.inverse()*q1.inverse()
        rot = quaternion.as_rotation_matrix(quat)
        
        self.frd = FRD()
        self.frd.fwd, self.frd.rh, self.frd.dwn = np.matmul(rot, np.array([self.ned.north, self.ned.east, self.ned.down]))
              
        return self.frd
    
    def frd2ned(self, frd = FRD(), ac_attitude = AC_ATTITUDE()):
        #入力：　前、右、下 単位m
        #入力：ロール、ピッチ、ヨ― 単位°
        #出力： 北、東、南 ：単位m
        #オイラーシーケンス：ヨー→ピッチ→ロール
        
        self.frd = frd
               
        self.ac_attitude = ac_attitude
        q1=quaternion.from_rotation_vector([0,0,np.radians(self.ac_attitude.yaw)])
        q2=quaternion.from_rotation_vector([0,np.radians(self.ac_attitude.pitch),0])
        q3=quaternion.from_rotation_vector([np.radians(self.ac_attitude.roll),0,0])
        
        quat=q1*q2*q3
        rot = quaternion.as_rotation_matrix(quat)
        
        self.ned = NED()
        self.ned.north, self.ned.east, self.ned.down = np.matmul(rot, np.array([self.frd.fwd, self.frd.rh, self.frd.dwn]))
        
        return self.ned
    
    def local2ned(self, local= LOCAL(), lla = LLA() ):
        #入力：　局所X、局所Y、局所Z 単位m
        #入力：緯度、経度　単位°
        #出力： 北、東、南 ：単位m
        #オイラーシーケンス：経度→緯度
        
        self.local = local
        self.lla = lla
        
        q1=quaternion.from_rotation_vector([0,np.radians(-90),0])
        q2=quaternion.from_rotation_vector([np.radians(self.lla.lon),0,0])
        q3=quaternion.from_rotation_vector([0,np.radians(-self.lla.lat),0])
        
        quat = q3.inverse()*q2.inverse()*q1.inverse()
        
        rot = quaternion.as_rotation_matrix(quat)
        
        self.ned = NED()
        self.ned.north, self.ned.east, self.ned.down = np.matmul(rot, np.array([self.local.local_x, self.local.local_y, self.local.local_z]))
        
        return self.ned
    
    def ned2local(self, ned = NED(), lla = LLA()):
        #入力：北、東、南 ：単位m　
        #入力：緯度、経度　単位°
        #出力： 局所X、局所Y、局所Z 単位m
        #オイラーシーケンス：経度→緯度
        
        self.ned = ned
        self.lla = lla

        q1=quaternion.from_rotation_vector([0,np.radians(-90),0])
        q2=quaternion.from_rotation_vector([np.radians(self.lla.lon),0,0])
        q3=quaternion.from_rotation_vector([0,np.radians(-self.lla.lat),0])
        
        quat=q1*q2*q3
        
        rot = quaternion.as_rotation_matrix(quat)
        
        self.local = LOCAL()
        self.local.local_x, self.local.local_y, self.local.local_z = np.matmul(rot, np.array([self.ned.north, self.ned.east, self.ned.down]))
        
        return self.local
    
    def frd2focal(self, frd = FRD(), sensor_attitude =SENSOR_ATTITUDE() ):
        #入力：前、右、下 ：単位m　
        #入力：アジマス、エレベーション、スピン　：単位°
        #出力： 水平右、垂直下、奥行前 ：単位m
        #オイラーシーケンス：アジマス→エレベーション→スピン
        self.frd = frd
        self.sensor_attitude = sensor_attitude
        
        q1=quaternion.from_rotation_vector([0,0,np.radians(self.sensor_attitude.az)])
        q2=quaternion.from_rotation_vector([0,np.radians(self.sensor_attitude.el),0])
        q3=quaternion.from_rotation_vector([np.radians(self.sensor_attitude.spin),0,0])
        q4=quaternion.from_rotation_vector([0,0,np.radians(90)])
        q5=quaternion.from_rotation_vector([np.radians(90),0,0])

        quat=q5.inverse()*q4.inverse()*q3.inverse()*q2.inverse()*q1.inverse()
        rot = quaternion.as_rotation_matrix(quat)
        
        self.focal_position = FOCAL_POSITION()
        self.focal_position.horizontal, self.focal_position.vertical, self.focal_position.depth = np.matmul(rot, np.array([self.frd.fwd, self.frd.rh, self.frd.dwn]))
        
        return self.focal_position

    def focal2frd(self, focal_position = FOCAL_POSITION(), sensor_attitude =SENSOR_ATTITUDE()):
        #入力： 水平右、垂直下、奥行前 ：単位m
        #入力：アジマス、エレベーション、スピン　：単位°
        #出力：前、右、下 ：単位m　
        #オイラーシーケンス：アジマス→エレベーション→スピン
        self.focal_position = focal_position
        self.sensor_attitude = sensor_attitude
        
        q1=quaternion.from_rotation_vector([0,0,np.radians(self.sensor_attitude.az)])
        q2=quaternion.from_rotation_vector([0,np.radians(self.sensor_attitude.el),0])
        q3=quaternion.from_rotation_vector([np.radians(self.sensor_attitude.spin),0,0])
        q4=quaternion.from_rotation_vector([0,0,np.radians(90)])
        q5=quaternion.from_rotation_vector([np.radians(90),0,0])

        quat=q1*q2*q3*q4*q5
        rot = quaternion.as_rotation_matrix(quat)
        
        self.frd = FRD()
        self.frd.fwd, self.frd.rh, self.frd.dwn = np.matmul(rot, np.array([self.focal_position.horizontal, self.focal_position.vertical, self.focal_position.depth]))
                
        return self.frd
    

