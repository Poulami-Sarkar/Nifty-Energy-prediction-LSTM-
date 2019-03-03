#import packages
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

#to plot within notebook
import matplotlib.pyplot as plt
#matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('d.csv')

#print the head
print(df.head())
#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')
df.index = df['Date']

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

#splitting into train and validation
train = new_data[:24]
valid = new_data[24:]
print(valid.shape)
#make predictions
preds = []
for i in range(0,6):
    a = train['Close'][len(train)-6+i:].sum() + sum(preds)
    b = a/6
    preds.append(b)
#plot
#plt.figure(figsize=(16,8))
#plt.plot(df['Close'], label='Close Price history')
#plt.show()