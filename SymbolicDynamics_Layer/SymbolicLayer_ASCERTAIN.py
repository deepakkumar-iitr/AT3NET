#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Input, LSTM, MultiHeadAttention, GlobalMaxPooling1D, BatchNormalization, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

###########################        Symbolic Transform        #############################

def symbolic_transform(X, N=20, N2=10, symbolic_features=8):
    """
    Transform the data into a symbolic sequence.
    """
    num_samples, num_points, num_features = X.shape
    symbolic_X = np.zeros_like(X)

    for feature in range(num_features):
        x_min = np.min(X[:, :, feature])
        x_max = np.max(X[:, :, feature])

        # First 8 features
        if feature < symbolic_features:
            Q = (x_max - x_min) / N
            for sample in range(num_samples):
                symbolic_X[sample, :, feature] = np.ceil((X[sample, :, feature] - x_min) / Q)

        # 9th and 10th features
        elif feature in [8, 9]:
            Q1 = (1- eta) * (x_max - x_min) / (N2 - 1)
            Q2 = eta * (x_max - x_min)
            threshold = x_min + 0.1 * (x_max - x_min)

            for sample in range(num_samples):
                for point in range(num_points):
                    if X[sample, point, feature] <= threshold:
                        symbolic_X[sample, point, feature] = np.ceil((X[sample, point, feature] - x_min) / Q1)
                    else:
                        symbolic_X[sample, point, feature] = N2  # the last symbol for top of the signal

        # 11th, 12th and 13th feature remains unchanged
        elif feature in [10,11,12]:
            symbolic_X[:, :, feature] = X[:, :, feature]

    return symbolic_X

##########################################################  One Hot Encoding  ###########################################
def one_hot_encoding(symbolic_X, N=20, N2=10):
    """
    Convert the symbolic representation into a one-hot encoded format.
    """
    num_samples, num_points, _ = symbolic_X.shape
    #encoded_shape = num_samples, num_points, 8*N + 2*N2 + 1
    encoded_shape = num_samples, num_points, 8*N + 2*N2 + 3
    encoded_data = np.zeros(encoded_shape, dtype=np.float32)

    for feature in range(13):
        if feature < 8:
            temp = to_categorical(symbolic_X[:, :, feature] - 1, num_classes=N)
            encoded_data[:, :, feature*N:(feature+1)*N] = temp

        elif feature in [8, 9]:
            temp = to_categorical(symbolic_X[:, :, feature] - 1, num_classes=N2)
            encoded_data[:, :, 8*N + (feature-8)*N2: 8*N + (feature-7)*N2] = temp

        # Feature 11, 12, 13 remains unchanged
        else:
            encoded_data[:, :, -1] = symbolic_X[:, :, -1]

    return encoded_data

