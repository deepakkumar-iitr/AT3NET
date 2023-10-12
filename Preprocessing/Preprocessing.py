#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Load the data
file_path="Amigos_GSR_EEG_3840.npy"
data = np.load(file_path)
print('Shape of the data:', data.shape)
print('First sample:\n', data[0])

# Load the labels
labels = np.loadtxt('labels.csv', delimiter=',', skiprows=1)

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

# Check for nan or inf in data
print("NaN in data:", np.isnan(data).any())
print("Inf in data:", np.isinf(data).any())

# Check for nan or inf in labels
print("NaN in labels:", np.isnan(labels).any())
print("Inf in labels:", np.isinf(labels).any())

# Calculate the mean of the non-NaN values in data
mean_value = np.nanmean(data)

# Replace NaN values in data with the mean
data = np.nan_to_num(data, nan=mean_value)

