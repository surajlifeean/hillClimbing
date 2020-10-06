# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 10:30:22 2020

@author: 
"""


import numpy as np
from numpy.random import seed
from numpy.random import randint
import functools
import operator
import matplotlib
import matplotlib.pyplot as plt
import itertools
import random
import os
import sys
import numpy
import itertools
import cv2

#varaibles

currentState =  randint(0, 2, 100)  #our random vector
per=2 #defines the percentage by which mutation needs to be done

face=cv2.imread('face.png')
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

W=face.shape[1] #width of the image 
H=face.shape[0] #height of the image



def img2vector(img_arr):
     fv = np.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))
     return fv

#storing face vector in a variable
faceVector=img2vector(face)/255


imgplot = plt.imshow(face)


    
#convert vector to matrix
def convert2img(vec, imgShape):
    imgArr = np.reshape(a=vec, newshape=imgShape)
    return imgArr

#Evaluate to see how close the current state is to the target vector
def evaluate(X,faceVector):
    error=-1*sum(abs(X-faceVector))
    return error




#mutate the current state
def mutate(currentState):
    #copy the current state to make changes in ot
    newState=currentState.copy()
    #randomly select some positions to be changed
    rndSelection =  randint(0,W*H, per)
    
    #Compliment values present in those positions
    for i in range(len(newState)):
        if i in rndSelection:
            #print(i)
            newState[i]=newState[i]^1
    return newState

#show the images after a specific interval
def showImages(currIteration,population, imShape, save_point):
    if(np.mod(currIteration, save_point)==0):
        best_solution_img = convert2img(population, imShape)
        #show constructed image
        fig,a =  plt.subplots(1,2)
        a[0].imshow(best_solution_img)
        a[1].imshow(face)
        
        

for i in range(8500):
    #evaluate the current solution
    evRes1=evaluate(currentState,faceVector)
    #new state after mutation
    newState=mutate(currentState)
    #evaluate the mutated solution
    evRes2=evaluate(newState,faceVector)
    #if the new state is better than the previous one then replace the current state with the new state
    if evRes2>evRes1:
        currentState=newState
        print("ev1-{}, ev2-{}".format(evRes1,evRes2))
    print("Outloop: ev1-{}, ev2-{}".format(evRes1,evRes2))
    print("---------------------------------")
    #See the constructed image
    showImages(i,currentState,(H,W), 
                     save_point=100)
