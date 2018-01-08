# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:23:06 2017

@author: Aakash
"""

import random
import numpy as np

P = np.array([[0,0.67,0.33],[0.5,0,0.5],[0.33,0.67,0]])
Q = np.array([[-1.0,0.67,0.33],[0.5,-1.0,0.5],[0.33,0.67,-1]])

def RM_to_TPM(Q):
    k = len(Q)
    P = np.zeros((k,k))
    for i in range(0,k):
        for j in range(0,k):
            if j==i and Q[i][i] == 0:
                P[i][j] = 1
            if j!=i and Q[i][i]!=0:
                P[i][j] = Q[i][j]/-Q[i][i]
            if j!=i and Q[i][i]==0:
                P[i][j]=0
            if j==i and Q[i][i]!=0:
                P[i][j]=0
    print("Transition Probability Matrix is: ", P)
    P1 = np.zeros((3,3))
    #def Cum_Matrix(P,k):
    P1[:,0] += P[:,0]
    for j in range(1,3):
        P1[:,j] = P1[:,j-1]+ P[:,j]
    return P1


'''
Function to define the state transition in each step
#Defining Function F with arguement
k-is the number of states
i-current state
x-random numbers from uniform distribution
'''

states = [0,1,2]
'''
Function to define the state transition in each step
'''
k=3
def F(x,i):
    for j in range(0,k):
        if j==0:
            if x <= P1[i][0]:
                return 0
        elif 0<j<=k-1:
            if P1[i][j-1]<x<=P1[i][j]:
                return j
        else:
            return k-1

def CTMC(Q,start,T):
    '''
    Code to simulate a continuous-time Markov Chain on a finite sample space
    ## Q is 3x3 matrix as provided
    ## start is the starting state(a number between 1 an 3)
    ## T is the total time to run the simulation
    ##
    ##Output is:
    ## ec = embedded chain (sequence of states)
    ## t = time at each state

    '''
    t = [0]
    n=0
    X =  [start]
    # drawing from exponential distribution
    H0 = np.random.exponential(scale = -1/Q[X[n]][X[n]])
    t.append(t[0]+H0)
    while max(t)<T:
        U = np.random.uniform(0,1)
        X.append(F(U,X[n]))
        Hn = np.random.exponential(scale = -1/Q[X[n]][X[n]])
        t.append(max(t) + Hn)
        n+=1
    t.remove(t[0])
    last = t[-1]
    t.remove(t[-1])
    X.remove(X[0])
    print(t,"\n")
    print("Last element of time sequence is: ", last)
    
    return X

print("State Space considered is : [0, 1, 2]")
initial_state = int(input("Enter initial state: "))
time_horizon = int(input("Enter Time-Horizon: "))
print("Time sequence is: \n ")
print("Embeded Chain for given Q-matrix is :\n\n",CTMC(initial_state,time_horizon))

