# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:17:39 2018

@author: Acer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tensorflow as tf
from keras.preprocessing import image, sequence
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, merge, Input
from keras.models import Sequential, Model
from keras.optimizers import Nadam
from keras.layers.merge import Concatenate, concatenate, Dot
import cv2
import keras
import PIL.Image

imagepath="./data/images/"
imagedir=glob.glob(imagepath+'*.png')
cap_path="./data/captions/combined.txt"
train_path="./data/captions/train_set.txt"
test_path="./data/captions/testing_set.txt"


captions=open(cap_path,'r', encoding="utf-8").read().split("\n")
x_train=open(train_path,'r').read().split("\n")
x_test=open(test_path,'r').read().split("\n")


tokens={}

for ix in range (len(captions)):
    temp=captions[ix].split(",")
    if temp[0] in tokens:
        tokens[temp[0]].append(temp[1:]);
    else:
        tokens[temp[0]]=[temp[1:]]
temp=captions[100].split(",")

    
print("number of training images {}".format(len(x_train)))

vgg=VGG16(weights="imagenet", include_top=True, input_shape=(224,224,3))

def preprocess_input(img):
    img=img[:,:,:,::-1]
    img[:, :, :, 0] -= 103.939
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 123.68
    return img

def preprocessing(img_path):
    print(img_path)
    i=cv2.imread(img_path)
    im=cv2.resize(i,(224,224), interpolation=cv2.INTER_AREA)
    im=image.img_to_array(im)
    im=np.expand_dims(im, axis=0)
    im=preprocess_input(im)
    return im

    
vgg=Model(inputs=vgg.input, outputs=vgg.layers[-2].output)

def get_encoding(model, img):
    image=preprocessing(imagepath+img)
    pred=model.predict(image)
    pred=np.reshape(pred, pred.shape[1])
    detail=pred.shape
    print(detail)
    return pred

#BUILDING VOCABULARY#
    
pd_ds=pd.read_csv("./data/captions/train_ds.txt", delimiter="\t")
ds=pd_ds.values

sentences=[]
for ix in range(ds.shape[0]):
    sentences.append(ds[ix,1])
    
print(len(sentences))

words=[i.split() for i in sentences]

unique=[]

for i in words:
    unique.extend(i)
    
print(len(unique))

unique = list(set(unique))
print(len(unique))

vocab_size = len(unique)

word_2_indices = {val:index for index, val in enumerate(unique)}
indices_2_word = {index:val for index, val in enumerate(unique)}

print(word_2_indices['<start>'])
print(indices_2_word[0])

max_len=0
for i in sentences:
    i=i.split()
    if len(i)>max_len:
        max_len=len(i)
    
print(max_len)

#Model in action

captions=np.load('./resource/captions.npy')
next_words=np.load('./resource/next_words.npy')

print(captions.shape)
print(next_words.shape)

images=np.load('./resource/images.npy')
print(images.shape)
print(vocab_size)
images_n=np.load('./resource/image_names.npy')
print(images_n.shape)

embedding_size=128

#Image Model

i1=Input((4096,))
iD=Dense(embedding_size, activation='relu')(i1)
image_model=RepeatVector(max_len)(iD)

#image_model.summary()

#Language Model

l1=Input((max_len,))
lE= Embedding(input_dim=vocab_size, output_dim=embedding_size)(l1)
lL=LSTM(256, return_sequences=True)(lE)
language_model=TimeDistributed(Dense(embedding_size))(lL)

#language_model.summary()






#Model


result=Concatenate()([image_model,language_model])

output_layer1=LSTM(1000,return_sequences=False)(result)
output_layer2=Dense(vocab_size)(output_layer1)
output_layer3=Activation('softmax')(output_layer2)

model=Model(inputs=[i1, l1], outputs=output_layer3)

model.compile(loss='categorical_crossentropy', optimizer=Nadam())

model.fit([images, captions], next_words, batch_size=5, epochs=2)
model.summary()

model.save_weights("C:/Users/Acer/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

#Testing
model.load_weights("./models/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
img="2106.png"
test_img=get_encoding(vgg, img)
def predict_captions(image):
    start_word=["<start>"]
    while True:
        par_caps=[word_2_indices[i] for i in start_word]
        par_caps=sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        par_caps=np.array(par_caps)
        image=np.array(image)
        preds=model.predict([image.reshape(1, 4096), par_caps.reshape(1,max_len)])
        word_pred=indices_2_word[np.argmax(preds[0])]
        
        start_word.append(word_pred)
        
        
        if word_pred=="<end>" or len(start_word)>max_len:
            break
        
    return ' '.join(start_word[1:-1])

Argmax_Search=predict_captions(test_img)
print(Argmax_Search)
def beam_search_pred(image, beam_index=3):
    start=[word_2_indices["<start>"]]
    start_word=[[start,0.0]]
    
    while len(start_word[0][0])<max_len:
        temp=[]
        for s in start_word:
            par_caps=sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            par_caps=np.array(par_caps)
            image=np.array(image)
            preds=model.predict([image.reshape(1,4096), par_caps.reshape(1,max_len)])
            preds=model.predict([np.array([image]), np.array(par_caps)])
            
            word_preds=np.argsort(preds[0])[-beam_index:]
            
            for w in word_preds:
                next_cap,prob=s[0][:], s[1]
                next_cap.append(w)
                prob+=preds[0][w]
                temp.append([next_cap,prob])
        start_word=temp
        start_word=sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word=start_word[-beam_index:]
    start_word=start_word[-1][0]
    inter_cap=[indices_2_word[i] for i in start_word]
    
    final_cap=[]
    for i in inter_cap:
        if i!='<end>':
            final_cap.append(i)
        else:
            break
    final_cap=' '.join(final_cap[1:])
    return final_cap

z=PIL.Image.open(imagepath+img)
z.show()
Beam_Search_index_3= beam_search_pred(test_img, beam_index=3)
Beam_Search_index_5= beam_search_pred(test_img, beam_index=5)
Beam_Search_index_7= beam_search_pred(test_img, beam_index=7)    
print("Argmax Prediction: "+Argmax_Search)
print("Beam Search 3: "+Beam_Search_index_3)

print("Beam Search 5: "+Beam_Search_index_5)
print("Beam Search 7: "+Beam_Search_index_7)

