# -*- coding: utf-8 -*-
"""
@author: taguchi
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams

import numpy as np


# 日本語のフォントを設定
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

####################################
# 散布図を描画する関数
# x,y: データ
# xlabel : x軸のラベル
# ylabel : y軸のラベル
####################################
def drawScatter(x,y,xlabel,ylabel,plot_labels = None):     
    plt.clf();

    # グラフの軸ラベル等の設定
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # plot_labelsが指定されない場合は，1～データ数の数字にする
    if plot_labels is None:
        plot_labels = range(1,x.shape[0]+1)
        
    # 散布図を描画
    for (i,j,k) in zip(x,y,plot_labels):
        plt.plot(i,j,'o')
        plt.annotate(k, xy=(i, j))

####################################
# ３次元の散布図を描画する関数
# x,y,z: データ
# xlabel : x軸のラベル
# ylabel : y軸のラベル
# zlabel : z軸のラベル
# dlabel : データ点のラベル
####################################
def drawScatter3D(x,y,z,xlabel,ylabel,zlabel,dlabel):     

    # グラフの消去（必要に応じて利用）
    # plt.clf();
    
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(111, projection='3d')
    
    # X,Y,Z軸にラベルを設定
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    # 散布図を描画
    ax.plot(x,y,z,marker="o",linestyle='None')
    
    # ラベル位置のマージン
    m = 0.001
    
    # ラベルを描画
    for i, j, k , l in zip(x, y, z, dlabel):
        ax.text(i+m, j+m, k+m, l, None)
    
    plt.show()

####################################
# テキストファイルを読み込む関数
# fn: ファイル名
####################################
    
def loadText(fn):     
    f = open(fn, 'r')
    
    text = f.readlines() # １行ごとリストになる
    
    f.close()
    return text
    
     
#################################### 
# メイン関数
####################################    
if __name__ == "__main__":
    # テスト用のデータ
    #x = [0.1,0.2,0.3]
    #y = [0.2,0.3,0.4]
    #z = [0.1,0.3,0.1]

    #　各プロットにつけるラベル
    #var_label = ["A","B","C"]
    #var_label = ["い","ろ","は"]
    
    # 各プロットにサンプル番号（1～n）のラベルを付ける場合
    #n = 3
    #sample_label = range(1,n+1)        

    # テストしたいラベルを選択
    #dlabel = var_label
    #dlabel = sample_label

    # 3次元プロット（各プロットに任意のラベルを付ける）
    #drawScatter3D(x,y,z,"x","y","z",dlabel)      
    
    fn = "lec06_data_variable_name_sjis.txt"
    variable_name_list = loadText(fn)

    d = pd.read_excel("lec06_data(18).xlsx")
    
    #print(d)

    data = d.values
    A = data[:61, 1:19]    
    
    N = A.shape[0]
    M = A.shape[1]
    
    Fx = A.sum(axis=0)
    Fy = A.sum(axis=1)
        
    B = np.diag(Fx)
    C = np.diag(Fy)
    print(C)
    
    Bri = np.diag(Fx**(-0.5))
    Ci = np.diag(1/Fy)
    
    H = Bri @ A.T @ Ci @ A @ Bri   
    print(H)
    H = H.astype(float)
    
    w,v = np.linalg.eig(H)
    
    sort_index = np.argsort(w)[::-1]
    
    sort_w = w[sort_index]
    sort_v = v[:,sort_index]
    
    #for i in range(1,4):
        #print(i,"の固有値", sort_w[i])
        #print(i,"の固有ベクトル",sort_v[:,i])
    
    ccr = np.cumsum(sort_w[1:]) / np.sum(sort_w[1:])
    #print(ccr)
    "print("hello")"
    x = np.zeros((3, M))
    y = np.zeros((3, N))
    for i in range(1,4):
        X = Bri @ sort_v[:,i]
        Y = (sort_w[i] ** (-0.5)) * (Ci @ A @ X)    

        x[i-1,:] = X
        y[i-1,:] = Y
    #print(y)

    #変数スコア
    drawScatter(x[0, :], x[1, :], "x1", "x2", plot_labels=variable_name_list)
    drawScatter(x[0, :], x[2, :], "x1", "x3", plot_labels=variable_name_list)
    drawScatter(x[1, :], x[2, :], "x2", "x3", plot_labels=variable_name_list)
    
    #サンプルスコア
    drawScatter(y[0, :], y[1, :], "y1", "y2")
    drawScatter(y[0, :], y[2, :], "y1", "y3")
    drawScatter(y[1, :], y[2, :], "y2", "y3")
    
    #3D変数スコア
    drawScatter3D(x[0,:], x[1,:], x[2,:], "x1", "x2", "x3", variable_name_list)
    
    #3Dサンプルスコア
    sample_label = range(1,N+1) 
    drawScatter3D(y[0,:], y[1,:], y[2,:], "y1", "y2", "y3", sample_label)