# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 08:36:01 2021

@author: SINCHANA BS
"""


#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt



#Reading the Data From Your Files
data = pd.read_csv("advertising.csv")
data.head()



#To Visualise Data
fig , axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])



# Creating X and Y for Regression(transforming)
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales



#Importing Linear Regression ALGO for Simple Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)



result = 6.9748214882298925 + 0.05546477*50
print(result)



#create a DataFrame with MIN and MAX value of the table
x_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()

preds = lr.predict(x_new)
preds

data.plot(kind = 'scatter', x='TV',y='Sales')

plt.plot(x_new,preds,c='red',linewidth=3)


import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lm.conf_int()

#finding the probablity values
lm.pvalues


#finding the R-Squared values
lm.rsquared


#muliti Linear Regression
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales


lr = LinearRegression()
lr.fit(X,y)



print(lr.intercept_)
print(lr.coef_)


lm = smf.ols(formula='Sales ~TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()

lm = smf.ols(formula='Sales ~TV+Radio',data=data).fit()
lm.conf_int()
lm.summary()















