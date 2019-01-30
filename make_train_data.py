# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 08:18:22 2018

@author: Sanjeev Narayanan
"""

import pickle
import numpy as np
f = open('all_data100.pckl', 'rb')
main_data = pickle.load(f)
f.close()

y_train = main_data['word'].values.tolist()
y_train=sorted(set(y_train), key=y_train.index)
word_to_int = dict((w, i) for i, w in enumerate(y_train))
int_to_word = dict((i, w) for i, w in enumerate(y_train))


y = np.eye((len(y_train)))
y_data = np.zeros((len(main_data),len(y_train)))
for i in range(len(y_data)):
    y_data[i] = y[word_to_int[main_data['word'][i]]]
    
x_train = np.array(main_data['Numpyarrays'].values.tolist())
for i in range(len(x_train)):
    x_train[i] = x_train[i]/255

f = open('x_train.pckl', 'wb')
pickle.dump(x_train, f)
f.close()
f = open('y_train.pckl', 'wb')
pickle.dump(y_data, f)
f.close()