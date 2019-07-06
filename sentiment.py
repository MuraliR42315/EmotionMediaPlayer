import pandas as pd
import shutil
import os
import csv
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import re
import keras
from nltk.stem import WordNetLemmatizer
from keras.utils import np_utils, plot_model
from keras.models import model_from_json
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.layers.merge import concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, RNN, MaxPooling1D,GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
import keras.backend as b
from keras import initializers, regularizers, constraints, optimizers, layers
from wordcloud import WordCloud
stop_words=set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()

def clean_text(text):
	text=re.sub(r'[^\w\s]','',text,re.UNICODE)
	text=text.lower()
	text=[lemmatizer.lemmatize(token) for token in text.split(" ")]
	text=[lemmatizer.lemmatize(token,"v") for token in text]
	text=[word for word in text if not word in stop_words]
	text=" ".join(text)
	return text

f=pd.read_csv("/home/muralidhar/Documents/isear2.csv",names=['Field1','SIT'])
f['Field1']=f['Field1'].map({'neutral':0,'joy':1,'fear':2,'sad':4,'disgust':6,'anger':3,'shame':5,'guilt':7,'surprise':8})
f['processed_SIT']=f.SIT.apply(lambda x: clean_text(x))
#print(f['processed_SIT'])
all_words=' '.join([text for text in f['processed_SIT']])
#print (all_words)
wordcloud=WordCloud(width=300,height=250, random_state=21,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
#plt.show()
plt.close('all')
max_features=7000
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(f['processed_SIT'])
list_tokenized_train = tokenizer.texts_to_sequences(f['processed_SIT'])
maxlen=512
x_t= pad_sequences(list_tokenized_train, maxlen=maxlen)
y=f['Field1']
print(x_t.shape)
#print(y)

dummy_y=y
#print(dummy_y)
embed_size=1024
Model=Sequential()
Model.add(Embedding(max_features, embed_size))
Model.add(Conv1D(128, kernel_size=3, activation='relu'))
Model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
Model.add(Bidirectional(LSTM(60, return_sequences = True)))
Model.add(GlobalMaxPooling1D())
Model.add(Dense(32,activation='relu'))
Model.add(Dense(9, activation='softmax'))

plot_model(Model,to_file='mlp_mnist.png',show_shapes=True)
Model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Model.summary()
batch_size = 100
epochs = 3
Model.fit(x_t,dummy_y, batch_size=batch_size, epochs=epochs, validation_split=0.2)	
Model_json = model.to_json()
with open("model.json", "w") as json_file:
    	json_file.write(model_json)
	# serialize weights to HDF5
model.save_weights("model.h5")
b.clear_session()
print("Saved model to disk")
	
