import numpy as np
import pandas as pd
def lm(Y,X):
	Xbar = np.append([np.ones(len(Y))], X, axis = 0)
	A = np.dot(Xbar,Xbar.T) 
	theta = np.dot(np.linalg.pinv(A),np.dot(Xbar,Y))
	return theta

data = pd.read_csv("NhuCauXeBus.csv")
Y = data['Y'].values
X2 = data['X2'].values
X3 = data['X3'].values
X4 = data['X4'].values
theta = lm(Y,(X2,X3,X4))
print(theta)
