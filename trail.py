import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np
import pandas as pd
import shutil
import os
import csv
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from keras.utils import np_utils, plot_model
from keras.models import model_from_json
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D,MaxPooling1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
import keras
import keras.backend as b
from keras import initializers, regularizers, constraints, optimizers, layers
from wordcloud import WordCloud
stop_words=set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()


# Initialize session
sess = tf.Session()
K.set_session(sess)


def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)

  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))

  return train_df, test_df

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

#train_df, test_df = download_and_load_datasets()
#train_df.head()

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

#def clean_text(text):
#  text=re.sub(r'[^\w\s]','',text,re.UNICODE)
#  text=text.lower()
#  text=[lemmatizer.lemmatize(token) for token in text.split(" ")]
#  text=[lemmatizer.lemmatize(token,"v") for token in text]
#  text=[word for word in text if not word in stop_words]
#  text=" ".join(text)
#  return text



def build_model(): 
  input_text = layers.Input(shape=(1,), dtype="string")
  embedding = ElmoEmbeddingLayer()(input_text)
  pred = layers.Dense(128, activation='sigmoid',kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
  pred1 = layers.Dense(9, activation='softmax')(pred)
  model = Model(inputs=[input_text], outputs=pred1)
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()  
  return model

#plot_model(model,to_file='mlp_mnist1.png',show_shapes=True)
f=pd.read_csv("/home/muralidhar/Documents/isear2.csv",names=['Field1','SIT'])
f['Field1']=f['Field1'].map({'neutral':0,'joy':1,'fear':2,'sad':4,'disgust':6,'anger':3,'shame':5,'guilt':7,'surprise':8})

max_features=7000

maxlen=512
train_text =f['SIT'].tolist()
train_text = [' '.join(t.split()[0:150]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
#x_t= pad_sequences(list_tokenized_train, maxlen=maxlen)
#print(x_t.shape)
y=f['Field1']
dummy_y=y


model = build_model()
plot_model(model,to_file='mlp_mnist1.png',show_shapes=True)

#batch_size = 100
#epochs = 3
model.fit(train_text,dummy_y, batch_size=32, epochs=3, validation_split=0.2)	
#model.fit(train_text, 
#          train_label,
#          validation_data=(test_text, test_label),
#          epochs=1,
#          batch_size=32)	
