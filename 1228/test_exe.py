# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:07:52 2021

@author: ookit1
"""

import mmap
import time
from struct import *
import ctypes
from multiprocessing import Process
#外部ﾌﾟﾛｸﾞﾗﾑ呼び出し
import subprocess

Kernel32 = ctypes.windll.Kernel32
mutex = Kernel32.CreateMutexA(0,0,"Global/UAV_MUTEX")


def SetPyToSimData(flag,stepcnt):
    shmem2 = mmap.mmap(0,232,"PYTHON_SIM_MEM")

    BlueData = [[0.0,1.0,2,3,4,5,1,2],[6,7,8,9,10,11,1,2]]
    RedData = [[0,1,2,3,4,5,1,2],[6,7,8,9,10,11,1,2]]

    if stepcnt > 0:
        BlueData[0][3] = -1
        BlueData[0][4] = -2
        BlueData[0][5] = -3

    OutValue = pack('ddddddiiddddddiiddddddiiddddddiiii',
                1.0,2.0,3.0,4.0,5.0,6.0,1,2,
                7.0,8.0,9.0,10.0,11.0,12.0,1,2,
                #        BlueData[0][0], BlueData[0][1], BlueData[0][2], BlueData[0][3], BlueData[0][4], BlueData[0][5],
                #        BlueData[1][0], BlueData[1][1], BlueData[1][2], BlueData[1][3], BlueData[1][4], BlueData[1][5],
                        RedData[0][0], RedData[0][1], RedData[0][2], RedData[0][3], RedData[0][4], RedData[0][5], RedData[0][6],RedData[0][7],
                        RedData[1][0], RedData[1][1], RedData[1][2], RedData[1][3], RedData[1][4], RedData[1][5], RedData[1][6],RedData[1][7],
                        flag,stepcnt)    
    
    shmem2.write(OutValue)
    print(BlueData[0][0],BlueData[0][1],BlueData[0][2],BlueData[0][3],BlueData[0][4],BlueData[0][5], RedData[0][6],)
    #print(OutValue)
    #shmem2.close()

def GetSimToPyData():
        
    #Uav[0] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw,dN,dE,dH
    #Uav[1] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw,dN,dE,dH
    #Uav[2] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw,dN,dE,dH
    #Uav[3] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw,dN,dE,dH
    #int Python指令
    InValue = unpack('ddddddddddddddddddddddddddddddddddddii',shmem)
    #print(InValue[37])
    return InValue[37]
    
if __name__ == '__main__':
    #Kernel32.WaitForSingleObject(mutex,-1)
    #サイズは、8の倍数
    shmem = mmap.mmap(0,296,"UAV_SIM_MEM")
    
    #起動
    print("exe起動")
    #subprocess.Popen(r"C:\Users\UAV\Desktop\Ai-DMU\社研4機シミュレーション環境_20211122\Release\UCAV1123.exe")
    subprocess.Popen(r"./UCAV.exe")
    #subprocess.Popen(r"C:\Users\UAV\Desktop\Ai-DMU\社研4機シミュレーション環境_masu\release\UCAV_ShaeMem.exe")
    time.sleep(3)

    #初期化
    SetPyToSimData(1,0)

    nOldCnt =-1
    nCnt = 0
    for n in range(1,100,1):
        SetPyToSimData(1,n)
        print("step ",n)

        while True:
            time.sleep(0.02)
            nCnt = GetSimToPyData()
            if nOldCnt != nCnt:
                nOldCnt = nCnt
                break;
        
    
    SetPyToSimData(1,-1)
    print("end")

    shmem.close()
    
    #Kernel32.ReleaseMutex(mutex)