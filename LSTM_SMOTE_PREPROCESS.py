# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:05:25 2022

@author: Orlando
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import nltk
nltk.download('punkt')
from nltk import word_tokenize

nltk.download("stopwords")
from nltk.corpus import stopwords
stop = list(stopwords.words("english"))

from sklearn.model_selection import train_test_split

import string

max_tokens = 1000

textdf = pd.read_csv("claimskg.csv")
textdf.head()

#concaténer le titre, les keywords à la fin des séquences pour ne pas inférer avec
            #la lecture de la première partie (une phrase cohérente)
            #(on le fait en même temps que le reste, dan sl a boucle ci-dessous)
    

#preprocessing du texte :
newDocs = []
for i in range(len(textdf)):
    doc = textdf.text[i]
    doc = doc+' '+str(textdf.author[i])
    doc = doc+' '+str(textdf.keywords[i])
    doc = doc+' '+str(textdf.headline[i])
    doc = doc+' '+str(textdf.source[i])
    #suppression de la ponctuation :
    new_doc = doc.translate(str.maketrans('','',string.punctuation))
    #suppression des stop words
    #new_doc = " ".join([word for word in new_doc.split(" ") if word not in stop])
    #suppression des nombres
    new_doc = "".join([word for word in new_doc if not word.isdigit()])
    #utilisation porterStemmer
    ps=nltk.stem.porter.PorterStemmer()
    #new_doc = " ".join([ps.stem(word) for word in new_doc.split(" ")])
    newDocs.append(new_doc)
textdf["text"]=newDocs

#numérisation
mytokens = Tokenizer(num_words=max_tokens, lower=False)
mytokens.fit_on_texts(textdf.text)
tokenized_text = mytokens.texts_to_sequences(textdf.text)

int_to_word = dict([(i,w) for (w,i) in mytokens.word_index.items()])


encoded_docs = mytokens.texts_to_sequences(textdf.text)

padded_docs = pad_sequences(encoded_docs,maxlen=np.max([len(doc) for doc in encoded_docs]))
print("max length of sequences : ",np.max([len(p) for p in padded_docs]))
print("min length of sequences : ",np.min([len(p) for p in padded_docs]))
print("number of sequences : ",len(padded_docs),"initial number : ",len(textdf.text))


#sanity check

  
#reshaping data
#X = np.reshape(padded_docs,(len(padded_docs),len(padded_docs[0]),1))
#X = X / float(len(mytokens.word_index))

truthvalues = [0 if i==1 else 1 for i in textdf.truthRating]

#séparer avant la génération de données pour éviter un biais
X_train, X_test,y_train,y_test = train_test_split(padded_docs,truthvalues,test_size=0.3,shuffle=True)

#générer séparément
from imblearn.over_sampling import SMOTE
X_train,y_train = SMOTE().fit_resample(X_train,y_train)
X_test,y_test = SMOTE().fit_resample(X_test,y_test)

print("after resampling, train value counts : \n0 :",len([i for i in y_train if i==0]),
      "\n1 :",len([i for i in y_train if i==1]))

#passer en catégorique, ça marche mieux :
X = np_utils.to_categorical(X_train)
y = np_utils.to_categorical(y_train,num_classes=2)
print(X.shape)
print(y.shape)

#model :
model = Sequential()
#model.add(LSTM(255,input_shape=(X.shape[1], X.shape[2]), return_sequences=(True)))
#model.add(Dropout(0.4))
model.add(LSTM(250))
model.add(Dropout(0.4))
model.add(Dense(y.shape[1],activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=3, batch_size=10)



#test it on test data
#first prepare the test data :
print("--------------------------------testing--------------------")
X = np_utils.to_categorical(X_test)
y = np_utils.to_categorical(y_test,num_classes=2)
print(X.shape)
print(y.shape)
model.evaluate(X,y)