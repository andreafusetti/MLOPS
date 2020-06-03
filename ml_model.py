#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:27:55 2020

@author: andreafusetti
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats

path_train = '/Users/andreafusetti/Desktop/IE/TERM_3/MLOPS/Individual_assignment/train.csv'
path_test = '/Users/andreafusetti/Desktop/IE/TERM_3/MLOPS/Individual_assignment/test.csv'
train_backup = pd.read_csv(path_train)
test_backup = pd.read_csv(path_test)

target = 'Survived'

def split_dataset(df, size = 0.2):
    y = df[target]
    X = df.drop(target,axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state = 0)
    return X_train, X_test, y_train, y_test

def remove_outliers(df, cols, treshold=3):
    z = np.abs(stats.zscore(df[cols]))
    return df[(z < treshold).all(axis=1)]


def onehot_encode(df, cols):
    df = df.copy()
    for i in cols:
        one_hot_route = pd.get_dummies(df[i], prefix = i)
        del df[i]
        new_dummy_names = []
        c=0
        for colname in one_hot_route.columns:
            new_dummy_names.append(str(i) + '_'+str(c))
            c+=1
        one_hot_route.columns = new_dummy_names
        df = pd.concat([df,one_hot_route],axis=1)
        
    return df

def remove_nulls(df_orig):
    df = df_orig.copy()
    for i in [x for x in list(df.columns) if x != target]:
        if (len(df[i][df[i].isnull()])/df.shape[0] > 0.8):
            df = df.drop(i, axis = 1)
        elif df[i].astype == 'category':
            df[i] = df[i].fillna('unknown')
        else:
            df[i] = df[i].fillna(0)
    return df

train = train_backup.copy()
test = test_backup.copy()

df = train.append(test)

df = remove_nulls(df)

num = ['Fare', 'Age']
cat = ['Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']

df = onehot_encode(df, cat)

df = remove_outliers(df, num)

train = df[0:890]
test = df[891::]
X_train = train.drop(target, axis = 1)
X_test = test.drop(target, axis = 1)
y_train = train[target]

#X_train, X_test, y_train, y_test = split_dataset(df)

model = RandomForestClassifier()
model.fit(X_train,y_train)

model.predict(X_test)

#NEW FUNCTION

def score(y_pred, y_test):
    return accuracy_score(self.y_test, y_pred)