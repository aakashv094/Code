# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:23:06 2017

@author: Aakash
"""

import random
import numpy as np

#Defining function To Convert Rate Matrix 'Q' To TPM Matrix 'P'

def RM_to_TPM(Q):
    k = len(Q)
    P = np.zeros((k,k))
    for i in range(0,k):
            if Q[i][i] == 0:
                P[i][i] = 1
            if Q[i][i]!=0:
                for j in range(0,k):
                    if j!=i:
                        P[i][j] = Q[i][j]/-Q[i][i]

    return P

#Defining function to find Cumulative Sum Matrix for TPM matrix P
    
def Cum_Sum(Q):
    k = len(Q)
    P1 = np.zeros((k,k))
    #def Cum_Matrix(P,k):
    P1[:,0] = RM_to_TPM(Q)[:,0]
    for j in range(1,k):
        P1[:,j] = P1[:,j-1]+ RM_to_TPM(Q)[:,j]
    return P1


'''
Function to define the state transition in each step
#Defining Function F with arguement
k-is the number of states
i-current state
x-random numbers from uniform distribution
'''

#states = [0,1,2,........,k-1]
'''
Function to define the state transition in each step
'''
def F(Q,x,i):
    k = len(Q)
    for j in range(0,k):
        if x <= Cum_Sum(Q)[i][0]:
            return 0
        if Cum_Sum(Q)[i][j-1]<x<=Cum_Sum(Q)[i][j]:
            return j

def CTMC(Q,start,T):
    '''
    Code to simulate a continuous-time Markov Chain on a finite sample space
    ## Q is kxk matrix as provided, where k = dim(Q)
    ## start is the starting state(a number between 0 an k-1)
    ## T is the total time to run the simulation
    ##
    ##Output is:
    ## ec = embedded chain (sequence of states)
    ## t = time at each state

    '''
    k = len(Q)
    t = [0]
    n=0
    X =  [start]
    # drawing from exponential distribution
    H0 = np.random.exponential(scale = -1/Q[X[n]][X[n]])
    t.append(t[0]+H0)
    while max(t)<T:
        U = np.random.uniform(0,1)
        X.append(F(Q,U,X[n]))
        Hn = np.random.exponential(scale = -1/Q[X[n]][X[n]])
        t.append(max(t) + Hn)
        n+=1
#    t.remove(t[0])
    last = t[-1]
    t.remove(t[-1])
#    X.remove(X[0])
#    print(t,"\n")
#    print("Last element of time sequence is: ", last,"\n")
    Hold_t_0 = []
    Hold_t_1 = []
    if X[0]==0 and len(X)%2 == 0:
            Hold_t_0 = np.diff(t)[::2]
            Hold_t_1 = np.diff(t)[1::][::2]
    elif X[0] ==1 and len(X)%2==0:
            Hold_t_1 = np.diff(t)[::2]
            Hold_t_0 = np.diff(t)[1::][::2]
    else:
            Hold_t_0 = np.diff(t)[::2]
            Hold_t_1 = np.diff(t)[1::][::2]
    return t, X, Hold_t_0, Hold_t_1

#----------------------------------Code for GBM and MMGBM
print("\nState Space considered is : [0, 1]","\n")
Q_mat = np.array([[-0.67,0.67],[0.5,-0.5]])
print("Q-matrix entered is : \n", Q_mat,"\n")
print("TPM-matrix for given Q-matrix is: \n", RM_to_TPM(Q_mat))
initial_state = int(input("Enter initial state: "))
time_horizon = int(input("Enter Time-Horizon: "))
CTMC_call = CTMC(Q_mat,initial_state,time_horizon)
tn = CTMC_call[0]
Xi = CTMC_call[1]
print("\n","Time sequence is: \n ", tn, "\n")
print("Embeded Chain for given Q-matrix is :\n", Xi, "\n")


S0 = 20
mu = [0.01,0.04]
variance=[0.0001,0.0009]
sigma = np.sqrt(variance)
Hold_t_0 = CTMC_call[2]
Hold_t_1 = CTMC_call[3]
h=0.01
mu_t_0 = np.multiply(mu[0], Hold_t_0)
print("Holding time for 0",Hold_t_0, mu_t_0)
mu_t_1 = np.multiply(mu[1], Hold_t_1)
print("Holding time for 1",Hold_t_1, mu_t_1)
mu_t = np.concatenate((mu_t_0, mu_t_1))
sigma_t_0 = np.multiply(sigma[0], Hold_t_0)
sigma_t_1 = np.multiply(sigma[1], Hold_t_1)
sigma_t = np.concatenate((sigma_t_0, sigma_t_1))

#print(mu_t, sigma_t, len(mu_t))

#from browmot_drift import *
#def MMGBM(t):
#    S_t = np.zeros(len(Xi)-1)
#    B_t = brn_motion(t)[1]
#    for r in range(1,len(Xi)-1):
#        S_t[r] = S_t[r-1]*(np.exp((mu_t[r-1] - (0.5*(sigma[r-1]**2)*h)) + sigma_t[-1]*(B_t[r] - B_t[r-1])))
#    return S_t[r]
#
##print("mu_t = ", mu_t,"sigma_{t} =", sigma_t)
#    
#
#print(MMGBM(time_horizon))

