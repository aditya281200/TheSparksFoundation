from sklearn.linear_model import LinearRegression
import pandas as pd 
import numpy as np
import requests
data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values
regressor = LinearRegression()
regressor.fit(X, Y)
t = [[9.25]]
print(regressor.predict(t))
