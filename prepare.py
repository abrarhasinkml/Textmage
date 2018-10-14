# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 00:31:07 2018

@author: Acer
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import glob
from keras.preprocessing import image, sequence
from keras.applications import VGG16
from keras.layers import Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector
from keras.models import Sequential, Model
from keras.optimizers import Nadam
import cv2




#directory=os.fsencode("./data/images/")
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

#from IPython.display import Image, display
#
#z=Image(filename=imagepath+temp[0])
#display(z)
#
#for ix in range(len(tokens[temp[0]])):
#    print(tokens[temp[0]][ix])
#    
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
    im/=255
    return im
#
#x=preprocessing(imagepath+temp[0])
#print(x.shape)
#
#plt.figure(0)
#plt.imshow(np.squeeze(x, axis=0))
#plt.show()
vgg=Model(inputs=vgg.input, outputs=vgg.layers[-2].output)

vgg.summary()

def get_encoding(model, img):
    image=preprocessing(imagepath+img)
    pred=model.predict(image)
    pred=np.reshape(pred, pred.shape[1])
    return pred

#print(temp[0])
#print(get_encoding(vgg, temp[0]).shape)

train_ds=open("./data/captions/train_ds.txt", 'w', encoding="utf-8")
#train_ds.write("image_id\tcaptions\n")

val_ds=open("./data/captions/val_ds.txt", "w", encoding="utf-8")
#val_ds.write("image_id\tcaptions\n")

train_encoded_images={}

c_train=0

for img in x_train:
    train_encoded_images[img]=get_encoding(vgg,img)
    for cap in tokens[img]:
        #print(tokens[img])
        caption="<start> "+ str(cap) +" <end>"
        train_ds.write(img+"\t"+caption+"\n")
        train_ds.flush()
        c_train+=1
train_ds.close()
with open("./resource/train_encoded_images.p", "wb") as pickle_f:
    pickle.dump(train_encoded_images, pickle_f)
 

test_encoded_images={}

c_test=0

for img in x_test:
    test_encoded_images[img]=get_encoding(vgg,img)
    for cap in tokens[img]:
        caption="<start> "+str(cap)+" <end>"
        val_ds.write(img+"\t"+caption+"\n")
        val_ds.flush()
        c_test+=1
val_ds.close()

   
with open("./resource/test_encoded_images.p", "wb") as pickle_f:
    pickle.dump(test_encoded_images, pickle_f)

#
pd_ds=pd.read_csv("./data/captions/train_ds.txt", delimiter="\t")
ds=pd_ds.values
#print(ds.shape)
sentences=[]
for ix in range(ds.shape[0]):
    sentences.append(ds[ix, 1])

print(len(sentences))
print(sentences[1])
words=[i.split() for i in sentences]
#chars=[j for i in words for j in i]
#alphabets=[j for i in chars for j in i]
#print(alphabets[0])
print(words[0])
print(len(words))
#alph=[]
unique=[]
for i in words:
    unique.extend(i)
unique=list(set(unique))
#alph=list(set(alph))
vocab_size=len(unique)

word_2_indices={val:index for index, val in enumerate(unique)}
indices_2_word={index:val for index, val in enumerate(unique)}

max_len=0
for i in sentences:
    i=i.split()
    if len(i)>max_len:
        max_len=len(i)
        
padded_sequences, subsequent_words = [], []

for ix in range(ds.shape[0]):
    partial_seqs = []
    next_words = []
    text = ds[ix, 1].split()
    text = [word_2_indices[i] for i in text]
    for i in range(1, len(text)):
        partial_seqs.append(text[:i])
        next_words.append(text[i])
    padded_partial_seqs = sequence.pad_sequences(partial_seqs, max_len, padding='post')

    next_words_1hot = np.zeros([len(next_words), vocab_size], dtype=np.bool)
    
    #Vectorization
    for i,next_word in enumerate(next_words):
        next_words_1hot[i, next_word] = 1
        
    padded_sequences.append(padded_partial_seqs)
    subsequent_words.append(next_words_1hot)
    
padded_sequences = np.asarray(padded_sequences)
subsequent_words = np.asarray(subsequent_words)
print(padded_sequences[0])
print(padded_sequences.shape)
#print subsequent_words.shape

for ix in range(len(padded_sequences[0])):
    for iy in range(max_len):
        print(indices_2_word[padded_sequences[0][ix][iy]]),
    print("\n")
print(len(padded_sequences[0]))

with open('./resource/train_encoded_images.p', 'rb') as f:
    encoded_images = pickle.load(f)

imgs = []

for ix in range(ds.shape[0]):
    imgs.append(encoded_images[ds[ix, 0]])

imgs = np.asarray(imgs)
print(imgs.shape)
number_of_images = 1500
captions = np.zeros([0, max_len])
next_words = np.zeros([0, vocab_size])

for ix in range(number_of_images):#img_to_padded_seqs.shape[0]):
    captions = np.concatenate([captions, padded_sequences[ix]])
    next_words = np.concatenate([next_words, subsequent_words[ix]])

np.save("./resource/captions.npy", captions)
np.save("./resource/next_words.npy", next_words)

images = []

for ix in range(number_of_images):
    for iy in range(padded_sequences[ix].shape[0]):
        images.append(imgs[ix])
        
images = np.asarray(images)

np.save("./resource/images.npy", images)

image_names=[]

for ix in range(number_of_images):
    for iy in range(padded_sequences[ix].shape[0]):
        image_names.append(ds[ix,0])
image_names=np.asarray(image_names)

np.save("./resource/image_names.npy",image_names)
print(len(image_names))
