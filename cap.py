from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug import secure_filename
try:
	import MySQLdb
except:
	import pymysql as MySQLdb
import pandas as pd
import shutil
import os
import csv
from pathlib import Path
import time
import gmplot as gm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import webbrowser
app = Flask(__name__, template_folder='templates')
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug import secure_filename
try:
	import MySQLdb
except:
	import pymysql as MySQLdb
import pandas as pd
import shutil
import os
import random
import csv
from pathlib import Path
import time
import matplotlib.pyplot as plt
#import pandas as pd
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from wordcloud import WordCloud
from pytube import YouTube
@app.route("/")
def home():
	return render_template("Home.html")

@app.route("/register", methods=['POST','GET'])	
def register():
	if request.method=='POST':
		print(request.form)
		print("hi")
		uname=request.form['usrnm']
		#pwd=request.form['psw']
		print(uname)
	yt=YouTube("https://www.youtube.com/watch?v=c3NTGRgcj2c")
	caption=yt.captions.get_by_language_code('en')
	captions=caption.generate_srt_captions()
	with open('cap.srt','w') as p:
		p.write(captions)

	import pysrt
	try:
		subs=pysrt.open('cap.srt',encoding="iso-8859-1")
		print(len(subs))
		print(subs[0])
		print(subs[0].text)
		print(subs[0].start)
		print(subs[0].end)
	except: 
		pass

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
	file1="/home/muralidhar/new datagen/data/model.h5"
	file2="/home/muralidhar/new datagen/data/model.json"
	json_file = open(file2, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(file1)
	print("Loaded model from disk")

	ynew=[]
	for i in range(len(subs)):
		x={}
		x['text']=subs[i].text
		x=pd.DataFrame.from_dict(x, orient='index', columns=['text'])
		x['text']=x.text.apply(lambda x: clean_text(x))	
		max_features=6000
		tokenizer=Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(x['text'])
		list_tokenized_test = tokenizer.texts_to_sequences(x['text'])
		maxlen=130
		x_t2= pad_sequences(list_tokenized_test, maxlen=maxlen)
		#print(x_t2)
		ynew.append(loaded_model.predict_classes(x_t2))
	lc=[]
	ch=random.choice([7,9,11,13,17,23,27,45,30])
	b= len(subs)//random.randint(ch-2,ch)
	for i in range(ch):
		cl=[]
		for z in range(b):
			color=random.randint(0,8)
			cl.append(color)
		lc.append(cl)
	po=0
	pc=0
	perc=[]
	for i in range(len(subs)):
		if po==b:
			po=0
			endtime=subs[i].end
			endt=subs[-1].end
			enl=str(endtime).split(':')
			endl=str(endt).split(':')
			enls=enl[2].split(',')	
			endls=endl[2].split(',')
			timl=int(enl[0])*60*60+int(enl[1])*60+int(enls[0])
			timel=int(endl[0])*60*60+int(endl[1])*60+int(endls[0])
			per=int((timl/timel)*100)
			#print(per)
			perc.append(per)
			pc=pc+1
		po=po+1


	def most_frequent(List): 
		counter = 0
		num = List[0] 
	      
		for i in List: 
			curr_frequency = List.count(i) 
			if(curr_frequency> counter): 
				counter = curr_frequency 
				num = i 
  
		return num 

	color=[]
	for i in lc:
#	print(i)
		mf=most_frequent(i)
		if mf ==0:
			color.append('#FFFFFF')
		elif mf==1:
			color.append('#008000')
		elif mf==2:
			color.append('#800080')
		elif mf==3:
			color.append('#0000FF')
		elif mf==4:
			color.append('#FF0000')
		elif mf==5:
			color.append('#FFFF00')
		elif mf==6:
			color.append('#A52A2A')
		else:
			color.append('#FFC0CB')
	filename="/home/muralidhar/new datagen/static/css/video.css"
	fp=open(filename,"w")
	st="main {\nmargin-left: 1em;\nmargin-right: 1em;\n}\n"
	st+="\nmain > section{ margin-bottom: 1em; }\n"
	st+="\nmain > section div.framed > div {\n border: 1px solid black;\n margin-bottom: 1ex;\n padding: 1ex;\n}\n"

	st+="\nmain > section button { padding: 0.5ex 1em; }\n"
	st+="\nmain > section input[type=\"range\"] { width: 400px;}\n"
	st+="\n#YouTube-player-progress::-moz-range-track {\n "
	st+="\nbackground: linear-gradient(to right,\n" 
	st+=""+ color[0]+" "+str(perc[0])+"%, "
	for i in range(1,len(perc)-1):
		st+=""+color[i]+" "+str(perc[i-1])+"% "+str(perc[i])+"%,"
	st+=" "+color[-1]+" "+str(perc[-2])+"% "+"100%);}\n"
	st+="\nmain > section label[for^=\"YouTube-player-\"] { position: absolute; }\n"
	st+="\nmain > section textarea {\nheight: 9ex;\nmargin: 0;\npadding: 0;\nvertical-align: top;\nwidth: 99%;\n}"
	st+="\n.center { text-align: center; }\n"
	st+="\n.margin-left-m { margin-left: 1em; }"
	st+="\n.margin-right-m { margin-right: 1em; }"
	st+="\n.nowrap { white-space: nowrap; }"
	st+="\n#indicator-display {\nbackground-color: #f0f0f0;\nborder-radius: 3ex;\ndisplay: inline-block;\nfloat: right;\nfont-family: 		monospace;\nheight: 2ex;\nline-height: 2ex;\nmargin: 1ex 1ex 0 0;\npadding: 1ex;\ntext-align: center;\nvertical-align: middle;\nwidth: 2ex;\n}\n"
	st+="\n#YouTube-video-id { font-family: monospace; }\n"
	st+="\n@media (max-width: 600px) {\n main > section iframe {\nheight: 240px;\nwidth: 320px;\n}\nmain > section input[type=\"range\"] { width: 320px;}\n}"
	st+="\n@media (max-width: 400px) {\nmain > section iframe {\nheight: 200px;\nwidth: 200px;\n} \n   main > section input[type=\"range\"]\n { width: 200px;\n }\n}"
	print (st)
	fp.write(st)
	uname=uname.split("v=")[1]
	fp.close()
	return render_template("Home.html",data1=uname)
#print(len(perc))
#print(len(lc))
if __name__ == "__main__":
	app.run(debug=True)

"""	
	a=random.choice([1,2,3,4,5,6,7,8,9])
	if a==1:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==2:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==3:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==4:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==5:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==6:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==7:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==8:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"
	if a==9:
		file1="/home/muralidhar/new datagen/data/model.h5"
		file2="/home/muralidhar/new datagen/data/model.json"

	#json_file = open(file2, 'r')
	#loaded_model_json = json_file.read()
	#json_file.close()
	#loaded_model = model_from_json(loaded_model_json)
	#loaded_model.load_weights(file1)
	#print("Loaded model from disk")

	ynew=[]
	for i in range(len(subs)):
		x={}
		x['text']=subs[i].text
		x=pd.DataFrame.from_dict(x, orient='index', columns=['text'])
		x['text']=x.text.apply(lambda x: clean_text(x))	
		max_features=6000
		tokenizer=Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(x['text'])
		list_tokenized_test = tokenizer.texts_to_sequences(x['text'])
		maxlen=130
		x_t2= pad_sequences(list_tokenized_test, maxlen=maxlen)
		#print(x_t2)
		ynew.append(loaded_model.predict_classes(x_t2))

#print(ynew)
	dic={}
	for i in range(len(ynew)-1):
		if (ynew[i]!=ynew[i+1]):
			dic[i]=ynew[i]
"""	
	
