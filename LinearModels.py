import numpy as np
import pandas as pd
def lm(Y,X):
	print(X)
	# Y = np.array[y1,y2..,yn] : yi : value
	# X = np.array[X1,X2..,Xn] : Xi : vector row
	Xbar = np.append([np.ones(len(Y))], X, axis = 0)
	# theta = ((XX^T)^t)XY
	A = np.dot(Xbar,Xbar.T) 
	theta = np.dot(np.linalg.pinv(A),np.dot(Xbar,Y))
	return theta

data = pd.read_csv("Data/NhuCauXeBus.csv")
Y = data['Y'].values
X2 = data['X2'].values
X3 = data['X3'].values
X4 = data['X4'].values
theta = lm(Y,(X2,X3,X4))
print(theta)
