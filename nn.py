"""
Neural Networks INFO 557 
Graduate Project
Submitted by: Akshit Agarwal
Instructor: Dr. Steven Bethard
"""



import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import jaccard_similarity_score



################################################## Preprocessing of Tweets from Training Set ###############################################

data=pd.read_csv('2018-E-c-En-train.txt',sep='\t')
max_features = 1200 #1000 #1200 #2000
max_len = 50
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Tweet'].values)
X = tokenizer.texts_to_sequences(data['Tweet'].values)
X= pad_sequences(X,maxlen=max_len)

########################################### LSTM Neural Network ###############################################################################

embed_dim = 40
model=Sequential()
model.add(Embedding(max_features,embed_dim,input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(32,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(32,activation='tanh'))              #64
model.add(Dropout(0.2))  
model.add(Dense(64,activation='tanh'))              #128
model.add(Dropout(0.2))   
model.add(Dense(128,activation='tanh'))              #128
model.add(Dropout(0.9))   
model.add(Dense(11,activation='sigmoid'))      
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['categorical_accuracy'])

############################################ Training on Training Dataset ###########################################################################

Y = data.loc[:,['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

batch_size = 32

model.fit(X_train, Y_train, epochs = 50, batch_size=batch_size, verbose = 2, validation_data=(X_test, Y_test))

################################## Evaluating on Development Dataset #################################################################

dev_data = pd.read_csv('2018-E-c-En-dev.txt',sep='\t')
X = tokenizer.texts_to_sequences(dev_data['Tweet'].values)
X=pad_sequences(X,maxlen=max_len)
Y = dev_data.loc[:,['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)


s,a=model.evaluate(X_train, Y_train, batch_size=batch_size, verbose = 2)      #Overall Categorical Score and Categorical Accuracy

print("score - %.2f" % (s))
print("accuracy - %.2f" % (a))

prediction=model.predict(X,batch_size=batch_size) #Prediction Matrix

############################################ Creating the Prediction File ###############################################################################

data={}
index=0

with open('2018-E-c-En-dev.txt',encoding='utf8') as f:
	for line in f:
		raw = line.split('\t')
		raw = raw[0: -11]
		data[index]=raw
		index+=1

index=1
output='ID\tTweet\tanger\tanticipation\tdisgust\tfear\tjoy\tlove\toptimism\tpessimism\tsadness\tsurprise\ttrust\n'
for pred in prediction:
	output += data[index][0]
	output += '\t'
	output += data[index][1]
	output += '\t'
	for i in range(11):
		if pred[i]<=0.5:
			output += '0'
		else:
			output  += '1'

		if(i != 10):
			output += '\t'
	output += '\n'
	index+=1

text_file = open("E-C_en_pred.txt", "w",encoding='utf8')
text_file.write("%s" % output)
text_file.close()


