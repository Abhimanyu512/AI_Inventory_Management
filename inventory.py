import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from sklearn.ensemble import RandomForestRegressor

#Read data from table
data = pd.read_csv("salesweekly.csv")
date = data["datum"]
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

#Array Creation for med1
x1 = list()
y1 = list()
for i in range(len(med1)-10):
    x = np.array(med1.loc[i:i+9])
    y = np.array(med1.loc[i+10])
    x1.append(x)
    y1.append(y)
x1 = np.array(x1).reshape(292,10)
y1 = np.array(y1).reshape(292,)

#Random Forests 85%
reg = RandomForestRegressor(max_depth=10,random_state=10)
reg.fit(x1,y1)
print(reg.score(x1,y1))
p = reg.predict(x1)

#Save model1
filename = 'model1.sav'
pickle.dump(reg, open(filename, 'wb'))
model1 = pickle.load(open(filename, 'rb'))

#----------------------------------------------------------------------------------------------------

# Array Creation for med2
x2 = list()
y2 = list()
for i in range(len(med2)-10):
    x = np.array(med2.loc[i:i+9])
    y = np.array(med2.loc[i+10])
    x2.append(x)
    y2.append(y)
x2 = np.array(x2).reshape(292,10)
y2 = np.array(y2).reshape(292,)

#Random Forests 
reg2 = RandomForestRegressor(max_depth=10,random_state=10)
reg2.fit(x2,y2)
print(reg2.score(x2,y2))

#Save model2
filename = 'model2.sav'
pickle.dump(reg2, open(filename, 'wb'))
model2 = pickle.load(open(filename, 'rb'))

# #----------------------------------------------------------------------------------------------------

#Array Creation for med3
x3 = list()
y3 = list()
for i in range(len(med3)-10):
    x = np.array(med3.loc[i:i+9])
    y = np.array(med3.loc[i+10])
    x3.append(x)
    y3.append(y)
x3 = np.array(x3).reshape(292,10)
y3 = np.array(y3).reshape(292,)

#Random Forests 
reg3 = RandomForestRegressor(max_depth=10,random_state=10)
reg3.fit(x3,y3)
print(reg3.score(x3,y3))

#Save model3
filename = 'model3.sav'
pickle.dump(reg3, open(filename, 'wb'))
model3 = pickle.load(open(filename, 'rb'))

# #----------------------------------------------------------------------------------------------------

#Array Creation for med4
x4 = list()
y4 = list()
for i in range(len(med4)-10):
    x = np.array(med4.loc[i:i+9])
    y = np.array(med4.loc[i+10])
    x4.append(x)
    y4.append(y)
x4 = np.array(x4).reshape(292,10)
y4 = np.array(y4).reshape(292,)

#Random Forests 
reg4 = RandomForestRegressor(max_depth=10,random_state=10)
reg4.fit(x4,y4)
print(reg4.score(x4,y4))

#Save model4
filename = 'model4.sav'
pickle.dump(reg4, open(filename, 'wb'))
model4 = pickle.load(open(filename, 'rb'))

# #----------------------------------------------------------------------------------------------------

#Array Creation for med5
x5 = list()
y5 = list()
for i in range(len(med5)-10):
    x = np.array(med5.loc[i:i+9])
    y = np.array(med5.loc[i+10])
    x5.append(x)
    y5.append(y)
x5 = np.array(x5).reshape(292,10)
y5 = np.array(y5).reshape(292,)

#Random Forests 
reg5 = RandomForestRegressor(max_depth=10,random_state=10)
reg5.fit(x5,y5)
print(reg5.score(x5,y5))

#Save model5
filename = 'model5.sav'
pickle.dump(reg5, open(filename, 'wb'))
model5 = pickle.load(open(filename, 'rb'))

# #----------------------------------------------------------------------------------------------------

#Array Creation for med6
x6 = list()
y6 = list()
for i in range(len(med6)-10):
    x = np.array(med6.loc[i:i+9])
    y = np.array(med6.loc[i+10])
    x6.append(x)
    y6.append(y)
x6 = np.array(x6).reshape(292,10)
y6 = np.array(y6).reshape(292,)

#Random Forests
reg6 = RandomForestRegressor(max_depth=10,random_state=10)
reg6.fit(x6,y6)
print(reg6.score(x6,y6))

#Save model6
filename = 'model6.sav'
pickle.dump(reg6, open(filename, 'wb'))
model6 = pickle.load(open(filename, 'rb'))

# #----------------------------------------------------------------------------------------------------

#Array Creation for med7
x7 = list()
y7 = list()
for i in range(len(med7)-10):
    x = np.array(med7.loc[i:i+9])
    y = np.array(med7.loc[i+10])
    x7.append(x)
    y7.append(y)
x7 = np.array(x7).reshape(292,10)
y7 = np.array(y7).reshape(292,)

#Random Forests
reg7 = RandomForestRegressor(max_depth=10,random_state=10)
reg7.fit(x7,y7)
print(reg7.score(x7,y7))

#Save model7
filename = 'model7.sav'
pickle.dump(reg7, open(filename, 'wb'))
model7 = pickle.load(open(filename, 'rb'))

# #----------------------------------------------------------------------------------------------------

#Array Creation for med8
x8 = list()
y8 = list()
for i in range(len(med8)-10):
    x = np.array(med8.loc[i:i+9])
    y = np.array(med8.loc[i+10])
    x8.append(x)
    y8.append(y)
x8 = np.array(x8).reshape(292,10)
y8 = np.array(y8).reshape(292,)

#Random Forests
reg8 = RandomForestRegressor(max_depth=10,random_state=10)
reg8.fit(x8,y8)
print(reg8.score(x8,y8))

#Save model8
filename = 'model8.sav'
pickle.dump(reg8, open(filename, 'wb'))
model8 = pickle.load(open(filename, 'rb'))

#----------------------------------------------------------------------------------------------------

#Plot bar graph
labels = ['W0','W1','W2','W3','W4','W5','W6','W7','W8','W9','W10']
x = np.arange(len(labels))
width = 0.35
actual = np.round(y1[:11])
predicted = np.round(p[:11])
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, actual, width, label='Actual')
rects2 = ax.bar(x + width/2, predicted, width, label='Predicted')
ax.set_ylabel('Sales')
ax.set_title('Actual sales versus predicted sales')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

#----------------------------------------------------------------------------------------------------

