#!/usr/bin/env python
# coding: utf-8

# # RF回線計算（レーダ照射、RCS反射）

# In[2]:


import numpy as np
import quaternion

import sys
sys.path.append('../')
sys.path.append('../utility')

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
#センサモデルクラスのインポート
from SENSOR_MODEL import SENSOR_RECOGNITION


# ### クラス：アンテナ利得（σi)<br>
# 入力：Focal_Az、Focal_El<br>
# 出力：σi　単位なし（無次元量）<br>
# $$
#  \sigma_{i} (Az,El)
# $$

# In[3]:


class AntGain:
    #コンストラクタ
    def __init__(self):
        self.dSumAz = 0
        self.dSumEl = 0
        self.Focal_Az = []
        self.Focal_El = []
    
    #加算（リセット処理あり）
    #入力：Focal_Az
    #入力：Focal_El
    #入力:flag<0>:前回値に加算、flag<1>：前回値クリア後加算
    def SetGain(self,focal_ang=FOCAL_ANGLE(),flag=0):
        if (flag == 1):
            self.dSumAz = 0
            self.dSumEl = 0
            self.Focal_Az = []
            self.Focal_El = []

        self.dSumAz += focal_ang.focal_az
        self.dSumEl += focal_ang.focal_el
        self.Focal_Az.append(focal_ang.focal_az)
        self.Focal_El.append(focal_ang.focal_el)
        
        print("Az",self.dSumAz,"El",self.dSumEl)
        #print("Az[]",self.Focal_Az[len(self.Focal_Az)-1],"El[]",self.Focal_El[len(self.Focal_El)-1])

    #ゲイン取得
    #出力 AntGain
    def GetGain(self):
        #暫定　結果：1
        self.dSumAz = 1
        self.dSumEl = 1
        return self


# ### クラス：ＲＣＳ(σrcs)
# 入力：到来Az、到来El<br>
# 出力：σrcs　単位なし（無次元量）<br>
# $$
#  \sigma_{RCS} (Az,El)
# $$

# In[5]:


class Rcs:
    #コンストラクタ
    def __init__(self):
        self.dSumRcs = 0
        self.dAz = []
        self.dEl = []
    
    #加算（リセット処理あり）
    #入力：Focal_Az
    #入力：Focal_El
    #入力:flag<0>:前回値に加算、flag<1>：前回値クリア後加算
    def SetRcs(self,focal_ang=FOCAL_ANGLE(),flag=0):
        if (flag == 1):
            self.dSumRcs = 0
            self.dAz = []
            self.dEl = []

        #計算式　現状未定
        #self.dSumRcs += 
        self.dAz.append(focal_ang.focal_az)
        self.dEl.append(focal_ang.focal_el)
        
        print("Rcs",self.dSumRcs)

    #Rcs取得
    #出力
    def GetRcs(self):
        #暫定　結果：1
        self.dSumRcs = 1
        return self.dSumRcs



# ### クラス：ＲＦ損失計算

# メソッド：RF大気減衰(τ）<br>
# 入力：Tempreture　単位 K（ケルビン）<br>
# 入力：distance 単位m<br>
# 入力：altitude 単位m <br>
# 出力：1 (今は実装なし）<br>
# $$
#  \tau(Tempretur, distance, altitude) = 1
# $$

# In[37]:


import math

class RfLos():
    #コンストラクタ
    def __init__(self,dTemp=0,dDis=0,dAlt=0):
        self.dTemp = dTemp
        self.dDis = dDis
        self.dAlt = dAlt
        self.dPtxsm = 0
        self.dPref = 0
        self.dPrx = 0
    
    #取得
    def GetRfLos(self):
        return self
    
    #RF対気減衰
    #入力：Temp
    #入力：Dis
    #入力：Alt
    def CalcRf(self,dTemp=0,dDis=0,dAlt=0):
        print("Rf",1)        
        return 1

    #メソッド：伝達電力
    def CalcTrancePow(self,dPtx=0,dGain=0,dDis=0,dRfLos=0):
        self.dPtxsm = (math.tau * dGain)/(4 * math.pi * math.pow(dDis,2)) * dPtx
        print("Ptxsm",self.dPtxsm)
        return self.dPtxsm 
    
    #メソッド：受信電力
    def CalcRefPow(self,dRcs=0):
        self.dPref = dRcs * self.dPtxsm
        print("Pref",self.dPref)
        return self.dPref
    
    #メソッド：受信電力
    def CalcRcvPow(self,dGain=0,dDis=0):
        self.dPrx = (math.tau * dGain) / (4 * math.pi * math.pow(dDis,2)) * self.dPref
        
        #print("tau",math.tau,math.tau*dGain)
        #print(" ",(4 * math.pi * math.pow(dDis,2)))
        #print("Pref",self.dPref)
        print("Prx",self.dPrx)
        return self.dPrx
