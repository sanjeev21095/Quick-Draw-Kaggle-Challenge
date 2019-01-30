# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:27:33 2018

@author: Sanjeev Narayanan
"""

import pandas as pd
import os
import numpy as np
import ast
import matplotlib.pyplot as plt
from PIL import Image
path = "C:/Users/jnarays5/Desktop/Quick Draw/train_simplified/"
main_data = pd.DataFrame()
for i in os.listdir(path):
    main_data = pd.read_csv(path+i).head(100)
#    main_data.append(df,ignore_index=True)
    break

for i in os.listdir(path)[1:]:
    df = pd.read_csv(path+i)
    new = df.head(100)
    val = [main_data,new]
    main_data = pd.concat(val,ignore_index=True)
    print(i)


def get_array(df):
    vecs = df['drawing']
    flag = ast.literal_eval(vecs)
    new_flag = [item for sublist in flag for item in sublist]
    new_flag1 = new_flag[0::2]
    new_flag2 =  new_flag[1::2]
    out1 = []
    out2 = []
    for i,j in zip(new_flag1,new_flag2):
        out1+=i
        out2+=j

    plt.figure()
    plt.plot(out1,out2)
    plt.savefig ("my_img.png" )
    plt.close()
    plt.gcf().clear()
    nparray = np.array(Image.open("my_img.png").convert("L").resize((224,224),Image.ANTIALIAS))
    return nparray

main_data['Numpyarrays'] = '' 
for index, row in main_data.iterrows():
#    get_array(main_data.loc[index],index)
    main_data.at[index,'Numpyarrays'] =  get_array(main_data.loc[index])
    print(index)
    
