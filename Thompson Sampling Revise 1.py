#Thompson Sampling Revise 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
import random
N=10000
d=10
adsSelected=[]
numbersOfRewards_1=[0]*d
numbersOfRewards_0=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(numbersOfRewards_1[i]+1,numbersOfRewards_0[i]+1)
        if(random_beta>max_random):
            max_random=random_beta
            ad=i
    adsSelected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        numbersOfRewards_1[ad]=numbersOfRewards_1[ad]+1
    else:
        numbersOfRewards_0[ad]=numbersOfRewards_0[ad]+1
    total_reward=total_reward+reward