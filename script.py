#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:46:22 2017

@author: celio
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import itertools

import time


def Entropy(data, columns):
    ent=0
    
    discrete=list(set(columns)-set(["Age","Height","Weight"]))
    
    columnsContinuos=list(set(columns)-set(discrete))

    columnsDiscrete=list(set(columns)-set(columnsContinuos))
   
    count_v=(data.groupby(columnsDiscrete+[pd.cut(data[col], np.linspace(data[col].min()-1e-3, data[col].max(), 5)) for col in columnsContinuos]).size()/data.shape[0])

    for xy in count_v:
        if xy>1e-5: # se p(xy)=0 então -p(xy)log(xy)=0 por definição
            ent-=xy*np.log2(xy)   
            
    return ent



def MutualInformation(data, columns):
    if len(columns)==2:
        return Entropy(data,[columns[0]])+Entropy(data,[columns[1]])-Entropy(data,columns)
            
    ent=0
    for m in range(1,len(columns)+1):
        for subset in itertools.combinations(columns,m):
            if (len(columns)-m) % 2==0:
                ent-=Entropy(data,subset)
            else:
                ent+=Entropy(data,subset)
            
    return ent


def DependenceCriterion(data, K,C): #I(K;C)
    return Entropy(data,K)+Entropy(data,C)-Entropy(data,C+K)


def GreadySearchDependenceCriterion(data, C,size=1e10,eps=1e-8):
    K=[]
    S=list(set(data.columns)-set(C))
    mutualInfK=0
    for k in range(0,size):
        max=0
        Fi=[]
        for fi in S:
            aux=DependenceCriterion(data, [fi]+K, C)
            if aux>max:
                max=aux
                Fi=fi
        if max>mutualInfK+eps:
            K.append(Fi)
            S.remove(Fi)
            mutualInfK=max
        else:
            break
        
    return K
        

def mRMRFunc(data,Fi, K,C):
    sum=0
    if K:
        for fj in K:
            sum+=MutualInformation(data,[Fi]+[fj])
    
        return MutualInformation(data,[Fi]+C)-(1/len(K))*sum
    else:
        return MutualInformation(data,[Fi]+C)
    

def GreadySearchmRMR(data, C,size=1e10, eps=1e-8):
    K=[]
    S=list(set(data.columns)-set(C))
    for k in range(0,size):
        max=0
        Fi=[]
        for fi in S:
            aux=mRMRFunc(data, fi, K, C)
            if aux>max:
                max=aux
                Fi=fi
        if max>0+eps:
            K.append(Fi)
            S.remove(Fi)
        else:
            break
    return K

def GreedyInformationGain(data,C,size, eps=1e-8):
    K=[]
    S=list(set(data.columns)-set(C))
    for k in range(0,size):
        max=0
        Fi=[]
        for fi in S:
            aux=MutualInformation(data, [fi]+C)
            if aux>max:
                max=aux
                Fi=fi
        if max>0+eps:
            K.append(Fi)
            S.remove(Fi)
        else:
            break
    return K


def GreedyGainRatio(data,C,size, eps=1e-8):
    K=[]
    S=list(set(data.columns)-set(C))
    for k in range(0,size):
        max=0
        Fi=[]
        for fi in S:
            aux=(MutualInformation(data, [fi]+C)/Entropy(data,[fi]))
            if aux>max:
                max=aux
                Fi=fi
        if max>0+eps:
            K.append(Fi)
            S.remove(Fi)
        else:
            break
    return K


def printOutput(data, K,clf,class_out):    
    print(DependenceCriterion(data,K,[class_out]))

    print(K)

    print(data.shape[1])

    print(len(K))

    data_dummie=pd.get_dummies(data[K])

    scores = cross_val_score(clf, data_dummie, data[class_out], cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), 2*scores.std()))

    


data=pd.read_csv("youngHabitsresponses.csv")
data=data.dropna()

class_out_vec=["Only child", "Village - town","PC","Science and technology", "Internet"]

clf = svm.SVC()

outfile = open('AtributosSelecionados.txt', 'w')

for class_out in class_out_vec:
    outfile.write("\n Class Out: "+class_out+"\n\n")
    
    entropy_class=Entropy(data,[class_out])
    
    start_time = time.time()
    
    K=list(set(data.columns)-set([class_out]))
    
    entropy_allAttr=DependenceCriterion(data,K,[class_out])
    
    data_dummie=pd.get_dummies(data[K])

    scores = cross_val_score(clf, data_dummie, data[class_out], cv=5)
    
    acuracy_allAttr=scores.mean()  
    
    EntropySet=[]
    
    AcuracySet=[]
    
    TimeSet=[]
    
    for elements_count in [1,2,3,4,5,6,7,8,9,10]:
    
        EntropyPerSize=[]
        
        AcuracyPerSize=[]
        
        TimePerSize=[]
        
        start_time = time.time()
        
        K=GreadySearchmRMR(data,[class_out],elements_count)
        
        if elements_count==10:
            outfile.write("Mrmr: [")
            outfile.write("\t".join(K))
            outfile.write("] \n")
            
        if len(K)==elements_count:
        
            EntropyPerSize.append(DependenceCriterion(data,K,[class_out]))
            
            data_dummie=pd.get_dummies(data[K])
        
            scores = cross_val_score(clf, data_dummie, data[class_out], cv=5)
        
            AcuracyPerSize.append(scores.mean())
            
            TimePerSize.append(time.time() - start_time)
            
        else:
            EntropyPerSize.append(None)
            AcuracyPerSize.append(None)
            TimePerSize.append(None)
            
        
        start_time = time.time()
        
        K=GreadySearchDependenceCriterion(data,[class_out],elements_count)
        
        if elements_count==10:
            outfile.write("Dependence Criterion: [")
            outfile.write("\t".join(K))
            outfile.write("] \n")

        if len(K)==elements_count:
        
            EntropyPerSize.append(DependenceCriterion(data,K,[class_out]))
            
            data_dummie=pd.get_dummies(data[K])
        
            scores = cross_val_score(clf, data_dummie, data[class_out], cv=5)
        
            AcuracyPerSize.append(scores.mean())
            
            TimePerSize.append(time.time() - start_time)
        
        else:
            EntropyPerSize.append(None)
            AcuracyPerSize.append(None)
            TimePerSize.append(None)
            
        
        start_time = time.time()
        
        K=GreedyInformationGain(data,[class_out],elements_count)
        
        if elements_count==10:
            outfile.write("Information Gain: [")
            outfile.write("\t".join(K))
            outfile.write("] \n")
        
        if len(K)==elements_count:
        
            EntropyPerSize.append(DependenceCriterion(data,K,[class_out]))
            
            data_dummie=pd.get_dummies(data[K])
        
            scores = cross_val_score(clf, data_dummie, data[class_out], cv=5)
        
            AcuracyPerSize.append(scores.mean())
            
            TimePerSize.append(time.time() - start_time)
            
        else:
            EntropyPerSize.append(None)
            AcuracyPerSize.append(None)
            TimePerSize.append(None)
        
        
        start_time = time.time()
        
        K=GreedyGainRatio(data,[class_out],elements_count)
        
        if elements_count==10:
            outfile.write("Gain Ration: [")
            outfile.write("\t".join(K))
            outfile.write("] \n")
        
        if len(K)==elements_count:
        
            EntropyPerSize.append(DependenceCriterion(data,K,[class_out]))
            
            data_dummie=pd.get_dummies(data[K])
        
            scores = cross_val_score(clf, data_dummie, data[class_out], cv=5)
        
            AcuracyPerSize.append(scores.mean())
            
            TimePerSize.append(time.time() - start_time)
            
        else:
            EntropyPerSize.append(None)
            AcuracyPerSize.append(None)
            TimePerSize.append(None)
    
    
        AcuracySet.append(AcuracyPerSize)
        
        EntropySet.append(EntropyPerSize)
        
        TimeSet.append(TimePerSize)
    
    
    EntropySet=list(zip(*EntropySet))
    
    AcuracySet=list(zip(*AcuracySet))
    
    TimeSet=list(zip(*TimeSet))
    
    plt.axhline(entropy_class,linestyle='--', color="yellow", xmin=0, xmax=4, label="Entropy Class Output")
    
    marker = itertools.cycle(('v', 's', '*', 'o')) 
    
    for entropyList,technique in zip(EntropySet,["mRMR","Dependence Criterion","Information Gain","Gain Ration"]):
        plt.plot(entropyList,linestyle='--', marker=next(marker), label=technique)
        
    x = np.arange(0,10,1)
    
    my_xticks = np.arange(1,11,1)
    
    plt.title("Entropia Mútua I(K;C) - C=\""+class_out+"\"")
    
    plt.ylabel("Entropia (bits)")
    
    plt.xlabel("Número de atributos (|K|)")
    
    plt.xticks(x, my_xticks)    

    plt.legend()
    
    plt.savefig("Apresentação/figuras/Entropy_"+class_out+".png")
    
    plt.show()
    
    
    
    plt.axhline(acuracy_allAttr,linestyle='--', color="yellow", xmin=0, xmax=4, label="Acuracy All Attr")

    for acuracyList,technique in zip(AcuracySet,["mRMR","Dependence Criterion","Information Gain","Gain Ration"]):
        plt.plot(acuracyList,linestyle='--', marker=next(marker),label=technique)
        
    x = np.arange(0,10,1)
    
    my_xticks = np.arange(1,11,1)
    
    plt.title("Acurácia da classificação - C=\""+class_out+"\"")
    
    plt.ylabel("Acurácia")
    
    plt.xlabel("Número de atributos (|K|)")
    
    plt.xticks(x, my_xticks)    
        
    plt.legend()
    
    plt.savefig("Apresentação/figuras/Acuracy_"+class_out+".png")
    
    plt.show()
    
    
    
    for timeList,technique in zip(TimeSet,["mRMR","Dependence Criterion","Information Gain","Gain Ration"]):
        plt.plot(timeList,linestyle='--', marker=next(marker),label=technique)
        
    x = np.arange(0,10,1)
    
    my_xticks = np.arange(1,11,1)
    
    plt.title("Tempo de execução - C=\""+class_out+"\"")
    
    plt.ylabel("Tempo (s)")
    
    plt.xlabel("Número de atributos (|K|)")
    
    plt.xticks(x, my_xticks)    
    
    plt.legend()
    
    plt.savefig("Apresentação/figuras/Time_"+class_out+".png")
    
    plt.show()
    

outfile.close()

    
