#Reinforcement Learning - UCB
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
#Importing data set
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
#Implementing UCB
import math
N=10000
d=10
numberOfSelections=[0]*d
sumOfRewards=[0]*d
adSelected=[]
for n in range(0,N):
    maxUpperBound=0
    ad=0
    for i in range(0,d):
        if(numberOfSelections[i]>0):
            avgReward=sumOfRewards[i]*numberOfSelections[i]
            deltaI=math.sqrt(3/2*math.log(n+1)/numberOfSelections[i])
            upperBound=avgReward+deltaI
        else:
            upperBound=1e400
        if upperBound>maxUpperBound:
            maxUpperBound=upperBound
            ad=i
    adSelected.append(ad)
    numberOfSelections[ad]=numberOfSelections[ad]+1