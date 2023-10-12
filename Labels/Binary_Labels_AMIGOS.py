#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def All_labels(choice):                           ###  Creating all labels
    # Load the labels
    labels = np.load('Amigos_labels_608.npy')

    # Seperate each Trait
    extraversion_scores = labels[:, 0]
    Agreeableness_scores=labels[:,1]
    Consc_scores=labels[:,2]
    Neuro_scores=labels[:,3]
    Openness_scores=labels[:,4]


    # Calculate the median (also try the fixed median, that is more natural)
    median_score_Ex = np.median(extraversion_scores)
    median_score_Op=np.median(Openness_scores)
    median_score_Ag=np.median(Agreeableness_scores)
    median_score_Con=np.median(Consc_scores)
    median_score_Neu=np.median(Neuro_scores)

    # Create binary labels
    Ex_binary_labels = (extraversion_scores > median_score_Ex).astype(int)
    Ag_binary_labels = (Agreeableness_scores > median_score_Ag).astype(int)
    Con_binary_labels = (Consc_scores > median_score_Con).astype(int)
    Neu_binary_labels = (Neuro_scores > median_score_Neu).astype(int)
    Op_binary_labels = (Openness_scores > median_score_Op).astype(int)

    All_binary_labels=[]
    All_binary_labels.append(Op_binary_labels)
    All_binary_labels.append(Con_binary_labels)
    All_binary_labels.append(Ex_binary_labels)
    All_binary_labels.append(Ag_binary_labels)
    All_binary_labels.append(Neu_binary_labels)
    if choice==0:
         return Op_binary_labels
    elif choice==1:
         return Con_binary_labels
    elif choice==2:
         return Ex_binary_labels
    elif choice==3:
         return Ag_binary_labels
    elif choice==4:
         return Neu_binary_labels
         
    return All_binary_labels

