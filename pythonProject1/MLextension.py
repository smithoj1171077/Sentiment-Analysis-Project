import pandas as pd
import sklearn as sk
import sklearn.ensemble as ens
import sklearn.metrics as metrics
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import joblib 
import tensorflow 
import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D

class CNN:
 def __init__(self):
    self.model = self.get_model()

 def get_model(self):
    try:
       joblib.load('accuracy')
    except FileNotFoundError:
       raw_training_data = pd.read_csv('TRAIN.csv')
       training_labels = raw_training_data['rating']



       # get the SBERT embedded text
       embedded_training_df = pd.read_csv('384EMBEDDINGS_TRAIN.csv')
       embedded_validation_df = pd.read_csv('384EMBEDDINGS_VALIDATION.csv')
       embedded_training_df.drop(columns=embedded_training_df.columns[0],axis=1,inplace=True)

       # set up keras Neural Network, the number of nodes matches the 384 dimensions of the SBERT embedding
       from keras.models import Sequential
       from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Flatten 

       model = Sequential()
       model.add(Conv1D(384, 20, activation='relu', input_shape=(384,1)))
       model.add(MaxPooling1D(5))
       model.add(Conv1D(196, 20, activation='relu',input_shape=(384,1)))
       model.add(GlobalMaxPooling1D())
       model.add(Flatten())
       model.add(Dense(100, activation='relu'))
       model.add(Dense(100, activation='relu'))
       model.add(Dense(1, activation='sigmoid'))
    

       model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

       X = embedded_training_df.to_numpy()
       X_val = embedded_validation_df.to_numpy()
       X_tensor = tensorflow.constant(X)


       counts = training_labels.value_counts()
       print(counts)

       # accoutning for imbalanced data
       nPos = counts[1]
       nNeg = counts[-1]
       total = nPos + nNeg
       weight0 = (1 / nPos) * (total / 2.0)
       weight1 = (1 / nNeg) * (total / 2.0)

       class_weights = {1: weight0, 0: weight1}
       print(class_weights)

       input_training_labels = training_labels.apply(lambda x: 0 if x == -1 else 1)

       X__tensor = tensorflow.constant(embedded_training_df.to_numpy())
       model.fit(X_tensor, input_training_labels,class_weight=class_weights)

       # do predictions
       raw_validation_data = pd.read_csv('VALIDATION.csv')
       validation_labels = raw_validation_data['rating']
       validation_labels = validation_labels.apply(lambda x: 0 if x == -1 else 1)

       # extract the validation label predictions
       embedded_validation_data = pd.read_csv('384EMBEDDINGS_VALIDATION.csv')
       embedded_validation_data.drop(columns=embedded_validation_data.columns[0],axis=1,inplace=True)
       X_val = embedded_validation_data.to_numpy()
       X_val = tensorflow.constant(X_val)
       predictions = pd.DataFrame(model.predict(X_val))
       predictions = predictions.iloc[:,0].apply(lambda x: 0 if x < 0.5 else 1)
       print(validation_labels)
       print(predictions)
       accuracy = accuracy_score(validation_labels,predictions)
       joblib.dump(accuracy,'accuracy')
       print(accuracy)
    else:
       print("validation set accuracy: ", joblib.load('accuracy'))
       
   
 


     
cnn = CNN()
  

