import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


#DATA HANDLING
data = pd.read_csv('housing_price_dataset.csv')
print(data.head())
print(data.describe())
X = data[["SquareFeet"]]
Y = data["Price"]


#DATA ANALYSIS
plt.scatter(X['SquareFeet'], Y, color='b')
plt.xlabel('SquareFeet')  
plt.ylabel('Price') 
plt.show()


#OBSERVATIONS


#LINEAR REGRESSION
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[6.575]])
print("Predicted value (LR): ",pred[0])
print("Accuracy (LR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['SquareFeet'], Y, color='b')
plt.plot(X['SquareFeet'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('SquareFeet')  
plt.ylabel('Price') 
plt.show()

#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators=100,max_depth=6)
mdl.fit(X, Y)
pred = mdl.predict([[6.575]])
print("Predicted value (RFR): ",pred[0])
print("Accuracy (RFR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['SquareFeet'], Y, color='b')
plt.plot(X['SquareFeet'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('SquareFeet')  
plt.ylabel('Price') 
plt.show()
