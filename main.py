import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

file_path = "housing.csv"
df = pd.read_csv(file_path)

ms = df.isnull().sum()
msp = (ms/len(df))*100

cleanDf = df.dropna()

q1 = cleanDf['median_house_value'].quantile(0.25)
q2 = cleanDf['median_house_value'].quantile(0.75)
iq = q2 -q1

lb = q1 - 1.5*iq
ub = q2 + 1.5*iq
nodf = cleanDf[(cleanDf['median_house_value']>= lb) & (cleanDf['median_house_value']<=ub)]

Q1 = nodf['median_income'].quantile(0.25)
Q2 = nodf['median_income'].quantile(0.75)
IQ = Q2 -Q1

lb2 = Q1 - 1.5*IQ
ub2 = Q2 + 1.5*IQ
nodf2 = nodf[(nodf['median_income']>= lb2) & (nodf['median_income']<=ub2)]

data = nodf2
data = data.drop("total_bedrooms" , axis = 1)

odummy = pd.get_dummies(data['ocean_proximity'] , prefix = 'ocean_proximity')
data = pd.concat([data.drop('ocean_proximity', axis =1), odummy] , axis = 1)
data = data.drop('ocean_proximity_ISLAND', axis = 1)

features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'population', 'households', 'median_income', 
       'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
target = ['median_house_value']
X = data[features]
y = data[target]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1111)
X_train = X_train.applymap(lambda x: int(x) if isinstance(x, bool) else x)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
print(X_train.isnull().sum())
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]  
X_train_constant = sm.add_constant(X_train)
model_fitted = sm.OLS(y_train,X_train_constant).fit()
print(model_fitted.summary())
X_train_constant = sm.add_constant(X_test)
test_predictions = model_fitted.predict(X_train_constant)
scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_scaled,y_train)
y_pred = lr.predict(X_test_scaled)
mse = mean_squared_error(y_test,y_pred)
rmse = sqrt(mse)

print(f'MSE on test set: {mse}')
print(f'RMSE on test set: {rmse}')
