#!/usr/bin/env python
# coding: utf-8

# # センサクラス
# #### masu 2021.9

# In[2]:


import sys

from UAV.uav_3d import uav_3d
import copy

#座標系クラスのインポート
from utility.COORDINATION_SYSTEMS import ECEF, LLA, NED, FRD, AC_ATTITUDE, LOCAL, SENSOR_ATTITUDE, FOCAL_POSITION, FOCAL_ANGLE
#WGS84座標変換クラスのインポート
# from utility.WGS84_COORDINATE_CONVERT import WGS84_COORDINATION_CONVERT
#座標変換クラスのインポート
# from utility.ATTITUDE_CONVERT import ATTITUDE_CONVERT
#センサ取得情報クラスのインポート
from SENSOR.SENSOR_FOCAL import SENSOR_FOCAL
#センサ性能クラスのインポート
from SENSOR.SENSOR_PERFORMANCE import FOV
#センサモデルクラスのインポート
from SENSOR.SENSOR_MODEL import SENSOR_RECOGNITION

from SENSOR.RADAR_TOY_MODEL import RADAR_TOY_MODEL        #レーダー　トイモデル


# In[4]:


class sensor:
    ### コンストラクタ
    ###　入力：blue
    ###　入力：red
    def __init__(self,blue,red):
        nBNum = len(blue)
        nRNum = len(red)
        nSize = len(blue) * len(red)
        
        #blue側のオブジェクト生成
        self.b_radar = [RADAR_TOY_MODEL()] * nSize
        for i in range(nSize):
            self.b_radar[i] = RADAR_TOY_MODEL()
            self.b_radar[i].detected = 0
            #print("detected",self.b_radar[i].detected)
        
        #red側のオブジェクト生成
        self.r_radar = [RADAR_TOY_MODEL()] * nSize
        for i in range(nSize):
            self.r_radar[i] =RADAR_TOY_MODEL()
            self.r_radar[i].detected = 0

        #blue,red情報の実体コピー
        self.s_blue = copy.deepcopy(blue)
        self.s_red = copy.deepcopy(red)

        #センサ性能情報保持用
        self.b_fov = FOV(0,0,nBNum,'./sensor/sensor_blue.csv')
        self.r_fov = FOV(0,0,nRNum,'./sensor/sensor_red.csv')
        #self.b_fov = FOV(0,0,nBNum)
        #self.r_fov = FOV(0,0,nRNum)
        
        #個別FOV設定サンプル
        #fov_val = self.b_fov.GetFov(1)
        #print("before az_",fov_val[0],"el_",fov_val[1])
        #self.b_fov.SetFov(1,10,15)
        #fov_val = self.b_fov.GetFov(1)
        #print("az_",fov_val[0],"el_",fov_val[1])
        
        print(">>sub-start")
        #相手機:red→blue算出
        print("red -> blue")
        for i in range(nBNum):
            #自機-位置
            opp_lla =LLA(self.s_blue[i].lat,self.s_blue[i].lon,self.s_blue[i].alt)
            #自機-機体姿勢
            opp_att =AC_ATTITUDE(self.s_blue[i].roll,self.s_blue[i].pitch,self.s_blue[i].yaw)
            for n in range(nRNum):
                #相手機:red-位置
                own_lla = LLA(self.s_red[n].lat,self.s_red[n].lon,self.s_red[i].alt)
                #red：機体姿勢
                own_att = AC_ATTITUDE(self.s_red[i].roll,self.s_red[i].pitch,self.s_red[i].yaw)
                
                fov_val = self.b_fov.GetFov(n)
                sens_att =SENSOR_ATTITUDE(0,fov_val[1],fov_val[0]) #スピン,El,Az
                sensor_range = self.b_fov.GetRange(n)
                print("fov_val",fov_val,"range",sensor_range)
                
                # ↓実行にて、self.r_radar[i].sensor_range_effected更新
                self.r_radar[i * nBNum + n].get_rf_detection_level(own_lla, own_att,sens_att, opp_lla, opp_att,sensor_range,fov_val)
        
        #相手機:blue→red算出
        print("blue -> red")
        for i in range(nRNum):
            #自機
            opp_lla =LLA(self.s_red[i].lat,self.s_red[i].lon,self.s_red[i].alt)
            opp_att =AC_ATTITUDE(0,0,0)
            for n in range(nBNum):
                #相手機:blue
                own_lla = LLA(self.s_blue[n].lat,self.s_blue[n].lon,self.s_blue[i].alt)
                own_att = AC_ATTITUDE(0,0,0)
                
                fov_val = self.r_fov.GetFov(n)
                sens_att =SENSOR_ATTITUDE(0,fov_val[1],fov_val[0]) #スピン,El,Az
                sensor_range = self.r_fov.GetRange(n)
                print("fov_val",fov_val,"range",sensor_range)

                # ↓実行にて、self.b_radar[i].sensor_range_effected更新
                self.b_radar[i * nRNum + n].get_rf_detection_level(own_lla, own_att,sens_att, opp_lla, opp_att,sensor_range,fov_val)

        for i in range(nSize):
            print("r-detect",self.r_radar[i].detected,"b-detect",self.b_radar[i].detected)
        
        
    ### blue,red情報更新
    ###　入力：blue
    ###　入力：red
    def update(self,blue,red):
        print("old_blue:",self.s_blue[0].lat,"in", blue[0].lat)
        #print("old_blue:",self.blue[1].lat,self.blue[1].lon)
        #print("old_red:",self.red[0].lat,self.red[0].lon)
        #print("old_red:",self.red[1].lat,self.red[1].lon)

        self.s_blue = [0] * len(blue)
        self.s_red = [0] * len(red)
        self.s_blue = copy.deepcopy(blue)
        self.s_red = copy.deepcopy(red)

        #更新処理
        #相手機:red→blue算出
        nBNum = len(blue)
        nRNum = len(red)
        for i in range(nBNum):
            #自機
            opp_lla =LLA(self.s_blue[i].lat,self.s_blue[i].lon,self.s_blue[i].alt)
            opp_att =AC_ATTITUDE(self.s_blue[i].roll,self.s_blue[i].pitch,self.s_blue[i].yaw)
            for n in range(nRNum):
                #相手機:red
                own_lla = LLA(self.s_red[n].lat,self.s_red[n].lon,self.s_red[n].alt)
                own_att = AC_ATTITUDE(self.s_red[i].roll,self.s_red[i].pitch,self.s_red[i].yaw)
                
                fov_val = self.b_fov.GetFov(n)
                sens_att =SENSOR_ATTITUDE(0,fov_val[1],fov_val[0]) #スピン,El,Az
                print("up:fov_val",fov_val)
                
                sensor_range = self.b_fov.GetRange(n)
                print("up:sensor_range",sensor_range)
                
                self.r_radar[i * nBNum + n].get_rf_detection_level(own_lla, own_att,sens_att, opp_lla, opp_att,sensor_range,fov_val)

        #相手機:blue→red算出
        for i in range(nRNum):
            #自機
            opp_lla =LLA(self.s_red[i].lat,self.s_red[i].lon,self.s_red[i].alt)
            opp_att =AC_ATTITUDE(self.s_red[i].roll,self.s_red[i].pitch,self.s_red[i].yaw)
            for n in range(nBNum):
                #相手機:red
                own_lla = LLA(self.s_blue[n].lat,self.s_blue[n].lon,self.s_blue[n].alt)
                own_att = AC_ATTITUDE(self.s_blue[i].roll,self.s_blue[i].pitch,self.s_blue[i].yaw)
                
                fov_val = self.r_fov.GetFov(n)
                print("up:fov_val",fov_val)
                sens_att =SENSOR_ATTITUDE(0,fov_val[1],fov_val[0]) #スピン,El,Az
                sensor_range = self.r_fov.GetRange(n)
                print("up:sensor_range",sensor_range)

                self.b_radar[i * nRNum + n].get_rf_detection_level(own_lla, own_att,sens_att, opp_lla, opp_att,sensor_range,fov_val)
                
        nSize = nBNum * nRNum
        for i in range(nSize):
            print("up:r-detect",self.r_radar[i].detected,"up:b-detect",self.b_radar[i].detected)
                
        
        #ダミー値で確認
        #self.s_blue[0].lat = 99
        #self.s_red[0].lat = 99
        
        #print("b-pos:",self.blue[0].pos)
        #print("lat_lon:",self.blue[0].lat,self.blue[0].lon)
        #print("lat_lon:",self.blue[1].lat,self.blue[1].lon)

        #print("r-pos:",self.red[0].pos)
        #print("lat_lon:",self.red[0].lat,self.red[0].lon)
        #print("lat_lon:",self.red[1].lat,self.red[1].lon)
 
        return self.s_blue,self.s_red


# In[ ]:




