#!/usr/bin/env python
# coding: utf-8

# # センサー性能クラス
# ## masu 2021.09.30
# ##### クラス作成時の引数、nAz、nElは、他クラスで使用してたのでそのまま残し
# ##### 追加でnSizeで、機体数によるセンサ関連の値保持を可能とする。
# ##### 現状、Az,El,range,取付角Az,取付角Elを管理。
# ##### センサの情報は、このクラス内に現状書いているが、ファイル等から呼び出す場合は
# ##### 変更可能

# In[18]:


import random
import csv

class FOV:
    def __init__(self,nAz=0,nEl=0,nSize=0,file=''):
        #センサ名、az値、el値
        self.fovlist = [["name1", 50, 11, 10000],[ "name2", 40, 22, 10000],[ "name3", 30, 33, 10000]]
        
        #初期化
        self.az = nAz
        self.el = nEl
        self.nSize = nSize
        self.range = []
        self.fov = []

        #csv入力時
        if (file != ""):
            #csv
            #print(file)
            csv_file = open(file,"r")
            f = csv.reader(csv_file,delimiter=",",doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            header = next(f)
            #print ("head",header)
            self.fovlist = []
            for row in f:
                for i in range(1,4):
                    #文字→int型
                    row[i] = int(row[i])
                self.fovlist.append(row)
                print(row)
            
            for i in range(self.nSize):
                if (0 < self.nSize):
                    self.fov.append(self.fovlist[i])
                    self.range.append(self.fov[i][3])
                    print ("fov-csv:",self.fov[i],"range:",self.range[i])
                else:
                    print ("none:",i)
        else:
        #csv入力なし
            for i in range(self.nSize):
                if (0 < self.nSize):
                    #とりあえず、ランダムで値設定し領域確保
                    self.fov.append(self.fovlist[random.randrange(3)])
                    self.range.append(random.randrange(10000,15000))
                    print ("fov:",self.fov[i],"range:",self.range[i])
                else:
                    print ("none:",i)

    #FOV値設定更新
    #nNo : idx
    #nAz : Az値
    #nEl : El値
    #戻り：成功：1 失敗：-1
    def SetFov(self, nNo, nAz = 0 ,nEl = 0):
        if (nNo < len(self.fov)):
            self.fov[nNo][1] = nAz
            self.fov[nNo][2] = nEl
        
            print("no:",nNo,"az:",self.fov[nNo][1],"el:",self.fov[nNo][2])
            nRet = 1
        else:
            nRet = -1
        return nRet

    #FOV値取得
    #nNo : idx
    #out : Az,El 範囲外は、0を返す
    def GetFov(self,nNo = 0):
        if (nNo < len(self.fov)):
            # [1]:Az [2]:El
            return self.fov[nNo][1],self.fov[nNo][2]
        else:
            return 0,0
            
    #range値設定更新
    #nNp : idx
    #nRange : Range値
    #戻り：成功：1 失敗：-1
    def SetRange(self,nNo,nRange = 0):
        if (nNo < len(self.fov) ):
            self.range[nNo] = nRange
            nRet = 1
        else:
            nRet = -1
        return nRet
    
    #range値取得
    #nNo : idx
    def GetRange(self,nNo = 0):
        if (nNo < len(self.range)):
            return self.range[nNo]
        else:
            return 0
       
    


# ### 使用例

# In[20]:


#Sensorfov = FOV(nAz=0,nEl=0,nSize=3)
#Sensorfov = FOV(nAz=0,nEl=0,nSize=3,file="./sensor_red.csv")


# In[3]:


#Sensorfov.SetFov(0,20,30)
#fov_val = Sensorfov.GetFov(3)
#Sensorfov.SetRange(0,12000)

#print("fov:",fov_val)
#print("Az:",fov_val[0],"El:",fov_val[1])

#range = Sensorfov.GetRange(0)
#print("range:",range)

#Sensorfov.SetRange(1,12345)
#range = Sensorfov.GetRange(1)
#print("range:",range)

