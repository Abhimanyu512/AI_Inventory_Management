import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

#Read data from table
data = pd.read_csv("salesweekly.csv")
data.drop(0)
data.drop("datum",axis=1,inplace=True)
med1 = data[["M01AB"]]
med2 = data[["M01AE"]]
med3 = data[["N02BA"]]
med4 = data[["N02BE"]]
med5 = data[["N05B"]]
med6 = data[["N05C"]]
med7 = data[["R03"]]
med8 = data[["R06"]]

#----------------------------------------------------------------------------------------------------

#Take inputs
product = ['#M01AB', '#M01AE', '#N02BA', '#N02BE', '#N05B', '#N05C', '#R03', '#R06']
df = pd.read_csv(r'lead_time.csv', index_col=0)
index = int(input("Enter index="))
print("Enter sales")
currentsales = float(input())
print("Finding data for "+product[index]+"....")

#----------------------------------------------------------------------------------------------------

#Choose model based on index
if index == 0:
    x = np.array(med1.loc[len(med1)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model1.sav'
    accuracy = 0.82

if index == 1:
    x = np.array(med2.loc[len(med2)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model2.sav'
    accuracy = 0.77

if index == 2:
    x = np.array(med3.loc[len(med3)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model3.sav'
    accuracy = 0.86

if index == 3:
    x = np.array(med4.loc[len(med4)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model4.sav'
    accuracy = 0.94

if index == 4:
    x = np.array(med5.loc[len(med5)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model5.sav'
    accuracy = 0.86

if index == 5:
    x = np.array(med6.loc[len(med6)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model6.sav'
    accuracy = 0.77

if index == 6:
    x = np.array(med7.loc[len(med7)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model7.sav'
    accuracy = 0.87

if index == 7:
    x = np.array(med8.loc[len(med8)-9:])
    x = np.append(x,currentsales).reshape(1,10)
    filename = 'model8.sav'
    accuracy = 0.93

#----------------------------------------------------------------------------------------------------

#Calculate Demand
model = pickle.load(open(filename, 'rb'))
demand = model.predict(x)
print("The predicted demand is: ")
print(demand[0])
mean1 = np.mean(x)
std1 = np.std(x)

#Inventory
def lt_mean(index):
    #print(df['Actual Lead Time'][str(product[index])])
    return df['Actual Lead Time'][str(product[index])].mean()

def lt_std(index):
    return df['Actual Lead Time'][str(product[index])].std()

def ss_cal(z,prod_mean,prod_std,mean1,std1):
    ss = (z*std1*np.sqrt(prod_mean))+(z*mean1*prod_std)
    return ss

def qor_cal(demand, accuracy, ss):
    qor = demand+((1-accuracy)*demand)+ss
    return qor

prod_mean=lt_mean(index)
prod_std=lt_std(index)

z=1.28

safety_stock= np.ceil(ss_cal(z,prod_mean,prod_std,mean1,std1))
print("Required Safety Stock=")
print(safety_stock)

qor = np.ceil(qor_cal(demand, accuracy, safety_stock))
print("Required Quantity of Reorder=")
print(qor[0])