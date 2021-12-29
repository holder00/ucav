# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:33:26 2021

@author: ookit1
"""

import PySimpleGUI as sg


sg.theme('DarkAmber')   # デザインテーマの設定

# ウィンドウに配置するコンポーネント
layout = [  [sg.Text('ここは1行目')],
            [sg.Text('ここは2行目：適当に文字を入力してください'), sg.InputText()],
            [sg.Button('OK'), sg.Button('キャンセル')] ]

# ウィンドウの生成
window = sg.Window('サンプルプログラム', layout)

# イベントループ
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'キャンセル':
        break
    elif event == 'OK':
        print('あなたが入力した値： ', values[0])

window.close()