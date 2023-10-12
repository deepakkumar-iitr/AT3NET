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

def create_model():
        # Input layer
        inputs = Input(shape=(3840, 3))

        # LSTM layers
        lstm = LSTM(32, return_sequences=True)(inputs)
        

        # Multi-head attention layers
        attention = MultiHeadAttention(num_heads=8, key_dim=3)(inputs, inputs, inputs)
        attention = MultiHeadAttention(num_heads=8, key_dim=3)(attention, attention, attention)

        attention = GlobalMaxPooling1D()(attention)
        attention = BatchNormalization()(attention)

        # Concatenatinh LSTM and Attention outputs
        print(attention.shape)
        lstm = Dense(32)(lstm[:, -1, :])  # Corrected time dimension
        print(lstm.shape)
        lstm = BatchNormalization()(lstm)

        x = concatenate([lstm, attention])

        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        #x = Dense(16, activation='relu')(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
X=data  #Contain the Data from all the signals

Traits=['Openness','Conscientiousness,','Extraversion','Agreeableness','Neuroticism']

n_of_fold=5
n_of_rep=5
random_state = 12883823
resultList = []
for index,values in enumerate(All_binary_labels):
    te_acc= []
    te_f1 = []
    repeat=1
    y=values   # Labels coming after converting into classification problem
    model=create_model()    # Creating model for each Trait

    rkf = RepeatedKFold(n_splits=n_of_fold, n_repeats=n_of_rep, random_state=random_state)      #repeat kfold function
    for train_idx, test_idx in rkf.split(X):
        print("Repeat {0}".format(repeat))
        print("---------------------------")
        repeat=repeat+1
        x_train, x_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        
        model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=20, batch_size=4)
        
        y_pred_test = model.predict(x_test)
        Binary_y_pred_test= [0 if i<0.5 else  1 for i in y_pred_test ]

        te_acc.append(accuracy_score(y_test,Binary_y_pred_test))

        te_f1.append(f1_score(y_test,Binary_y_pred_test))

    r1, r2 = round(np.array(te_acc).mean(),3), round(np.array(te_acc).std(),2)
    f1, f2 = round(np.array(te_f1).mean(),3),round(np.array(te_f1).std(),2)
    resultList.append([ [r1, r2], [f1, f2]  ])

    print("##############################################################################################")
    print("Length of Accuracy List: ",len(te_acc))
    print("Length of F1 Score List: ",len(te_f1))
    print("Results for Trait: ", Traits[index])
    print("Test accuracy {0} ± {1}" .format(r1, r2))
    print("Test F1 {0} ± {1}" .format(f1,f2))
    print("##############################################################################################")

