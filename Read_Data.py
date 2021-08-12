import pandas as pd
import numpy as np
from numpy import nan


def read_data(name):
    # simple label encoding to the data
    df = pd.read_csv(name, names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l','m','n','o','p','q','r','s','t','u','v'])
    features = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l','m','n','o','p','q','r','s','t','u','v']
    featuresd={'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21}
    list_d = [{'e': 0, 'p': 1},
              {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
              {'f': 0, 'g': 1, 'y': 2, 's': 3},
              {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9},
              {'t': 0, 'f': 1},
              {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
              {'a': 0, 'd': 1, 'f': 2, 'n': 3},
              {'c': 0, 'w': 1, 'd': 2},
              {'b': 0, 'n': 1},
              {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11},
              {'e': 0, 't': 1},
              {'f': 0, 'y': 1, 'k': 2, 's': 3},
              {'f': 0, 'y': 1, 'k': 2, 's': 3},
              {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
              {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
              {'p': 0, 'u': 1},
              {'n': 0, 'o': 1, 'w': 2, 'y': 3},
              {'n': 0, 'o': 1, 't': 2},
              {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7},
              {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
              {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
              {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6}]
    for name in features:
        df[name] = df[name].map(list_d[featuresd.get(name)])
    usefeatures = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j','k','l','m','n','o','p','q','r','s','t','u','v']
    target = 'f'
    return df, usefeatures, target


def get_features(data_set, features):
    # get the features
    return data_set[features]

def get_featuretags():
    # get feature tags for label encoding
    usefeatures = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                   'v']
    return usefeatures


def get_targettag():
    # get target tag for label encoding
    usefeatures = ['f']
    return usefeatures

def get_targetop():
   return ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']


def get_target(data_set, target):
    # get the features
    return data_set[target]


def get_odor(data, r):
    # finds where is the 1 in a one hot vector
    row = data[r]
    for c in range(len(row)):
        if (row[c] == 1):
            return c


def reverse_odor(data):
    # reverses one hot encoded odor to labeled
    list=[]
    for r in range(len(data)):
        list.append(get_odor(data, r))
    reversed=np.array(list)
    return reversed


def read_data2(name):
    # one hot encoding to the data
    df = pd.read_csv(name,
                     names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't', 'u', 'v'])
    features = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v']

    inverted = pd.get_dummies(df)
    all_features = inverted.columns.values
    usefeatures = np.delete(all_features, [24,25,26,27,28,29,30,31,32])
    target = all_features[24:33]
    return inverted, usefeatures, target


def read_data3(name):
    # simple label encoding to the features only
    df = pd.read_csv(name,
                     names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't', 'u', 'v'])
    usefeatures_tags = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                   'v']
    target_tag = 'f'
    features=df[usefeatures_tags]
    target=df[target_tag]
    features = pd.get_dummies(features)
    dic = {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8}
    target = target.map(dic)
    return features, target


def read_both_data(name,name2):
    # reads both the missing and regular dada and one hot encodes them
    df = pd.read_csv(name,
                     names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't', 'u', 'v'])
    df2 = pd.read_csv(name2,
                     names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't', 'u', 'v'])
    usefeatures_tags = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                        'u',
                        'v']
    target_tag = 'f'
    dfr = df.append(df2, ignore_index=True)
    dfr = dfr.replace('-', nan)
    features = dfr[usefeatures_tags]
    target = dfr[target_tag]
    features = pd.get_dummies(features)
    dic = {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8}
    target = target.map(dic)
    return features, target
