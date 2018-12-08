from __future__                import division
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection   import cross_val_score
from sklearn.linear_model      import LinearRegression
from sklearn.preprocessing     import PolynomialFeatures
from sklearn.pipeline          import make_pipeline
from sklearn.model_selection   import KFold
from sklearn.metrics           import mean_squared_error



#LOAD IN FILE
btcExcel = "btc.xlsx"
btc = pd.ExcelFile(btcExcel)

#MAKE DATAFRAME
bitcoin = btc.parse("btc (2)")

print bitcoin

#PLOT PRICES AGAINST TIME (for training data)

bitcoin.plot(x='date', y='price(USD)')
plt.title('Price of Bitcoin (4/28/13 - 9/16/18)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

btc = bitcoin.tail(n=10)
btc.plot(x='date', y='price(USD)')
plt.title('Price of Bitcoin Over Ten Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


#maybe we want some other time frames too - smaller and larger



"""avg_kfold_errors = []
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for p in range(10):#looping through degrees
    #transforming X to fit the degree we test
    poly = PolynomialFeatures(degree=p)
    Xtf = poly.fit_transform(X)

    avg = 0
    for train_index, test_index in kf.split(Xtf):
                        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = Xtf[train_index], Xtf[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        regressor = LinearRegression()
        model = regressor.fit(X_train, Y_train)
        y_pred = regressor.predict(X_test)

            avg += mean_squared_error(Y_test, y_pred)

        avg = avg/5
        avg_kfold_errors.append((avg,p))

        ans = avg_kfold_errors[0]"""
