import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pathName="D:\projects\Learning\Machine Learning Projects\Regrission Projects\Regression with one variable\Salary_Data.csv"

data=pd.read_csv(pathName)




#____ 2- Data Cleaning ____
data=data.drop_duplicates()
sumOfNullValues=data.isnull().sum()
# print(sumOfNullValues)
#____ 3- split the Data ____
data.insert(0,'Ones',1)
cols=data.shape[1]
yearsOfExperience=data.iloc[:,:cols-1].values
salary=data.iloc[:,cols-1:].values

# ___ split the data to train and test ____
x_train,x_test,y_train,y_test=train_test_split(yearsOfExperience,salary,test_size=1/3,random_state=0)

yearsOfExperience_train = np.matrix(x_train)
salary_train = np.matrix(y_train)
theta=np.matrix(np.array([0,0]))

def computeCost(x,y,theta):
    result=np.power(((x*theta.T)-y),2)
    return np.sum(result) / (2*len(result))

print(computeCost(yearsOfExperience_train,salary_train,theta))

def gradientDescent(yearsOfExperince,salary,learningRate,numOfIterations,thetas):
    temp=np.matrix(np.zeros(thetas.shape))
    parameters=int(thetas.ravel().shape[1])
    cost=np.zeros(numOfIterations)
    for i in range(numOfIterations):
        error=(yearsOfExperince*thetas.T)-salary
        for j in range(parameters):
            term=np.multiply(error,yearsOfExperince[:,j])
            thetas[0, j] = thetas[0, j] - (learningRate / len(yearsOfExperince)) * np.sum(term)
        cost[i] = computeCost(yearsOfExperince, salary, thetas)
    return thetas,cost

numOfIteration=1000
learningRate=0.01
thetas,cost=gradientDescent(yearsOfExperience_train,salary_train,learningRate,numOfIteration,theta)

xs = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)
bestFitLine = theta[0, 0] + (theta[0, 1] * xs)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_train[:, 1], y_train, label='Training Data')
ax.plot(xs, bestFitLine, 'r')
ax.set_xlabel('Experience (years)')
ax.set_ylabel('Salary($)')
ax.set_title('Salary Vs Years Of Experience (Training Data)')


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_test[:, 1], y_test)
ax.plot(xs, bestFitLine, 'r')
ax.set_xlabel('Experience (years)')
ax.set_ylabel('Salary($)')
ax.set_title('Salary Vs Years Of Experience (Test Data)')
plt.show()


