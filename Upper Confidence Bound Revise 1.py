#Upper Confidence Bound Revise 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
import math
N=10000
d=10
ads_selected=[]
noOfSelections=[0]*d
sumOfRewards=[0]*d
totalReward=0
for n in range(0,N):
    ad=0
    maxUpperBound=0
    for i in range(0,d):
        if(noOfSelections[i]>0):
            avgReward=sumOfRewards[i]/noOfSelections[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/noOfSelections[i])
            upperBound=avgReward+delta_i
        else:
            upperBound=1e400
        if upperBound>maxUpperBound:
            maxUpperBound=upperBound
            ad=i
    ads_selected.append(ad)
    noOfSelections[ad]=noOfSelections[ad]+1
    reward=dataset.values[n,ad]
    sumOfRewards[ad]=sumOfRewards[ad]+reward
    totalReward=totalReward+reward
plot.hist(ads_selected)
plot.title('Histogram')
plot.xlabel('Ads')
plot.ylabel('No of times')
plot.show()s