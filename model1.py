# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Boston.csv')

X = dataset.iloc[:,:3]

y = dataset.iloc[:,3]

# Data Transformation
# Feature Selection
# Hyperparameter tuning


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model1.pkl','rb'))
print(model.predict(X.iloc[0,:].values.reshape(1,-1)))