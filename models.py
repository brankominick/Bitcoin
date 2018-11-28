from __future__                import division
import numpy
import pandas                  as pd
import matplotlib.pyplot       as plt

from sklearn.model_selection   import cross_val_score
from sklearn.model_selection   import KFold
from sklearn.metrics           import accuracy_score
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.decomposition     import PCA
from sklearn.neural_network    import MLPClassifier
from sklearn.svm               import SVC




def GetData():

    #LOAD IN FILE
    btcExcel = "btc.xlsx"
    btc = pd.ExcelFile(btcExcel)

    #MAKE DATAFRAME
    bitcoin = btc.parse("btc (2)")

    #TARGET CLASS: 1 if decrease in price, 0 if increase
    y = []
    for i in range(len(bitcoin['price(USD)'])):
        if (i == len(bitcoin['price(USD)'])-1 or bitcoin['price(USD)'][i+1]>bitcoin['price(USD)'][i]):
            y.append(1)
        else:
            y.append(0)
    y = pd.DataFrame({'y':y})



    bitcoin = bitcoin.drop(columns=['date'])
    bitcoin = pd.concat([bitcoin,y],axis=1)

    """#Lets also add in 10 day moving average of price
    MA = pd.DataFrame({'MA':bitcoin['price(USD)'].rolling(window=10).mean()})
    print MA
    bitcoin = pd.concat([bitcoin,MA],axis=1)
    bitcoin['MA'].fillna(bitcoin['price(USD)'],inplace=True)"""

    #bitcoin now contains all attributes we will use including target class

    #for i in range(len(bitcoin)):
    #    print (bitcoin['price(USD)'].iloc[i],bitcoin['y'].iloc[i])

    #Lets make some FOLDS!
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(bitcoin):
        train, test = bitcoin.iloc[train_index], bitcoin.iloc[test_index]


    #Now split into x and y
    Y_train, Y_test = train.y, test.y
    X_train, X_test = train.drop(columns='y'), test.drop(columns='y')

    #print "Xtrain", X_train, "X_test", X_test, "Y_train", Y_train, "Y_test", Y_test

    return [X_train,X_test,Y_train,Y_test]


    #So we have our training and testing data...
    #Time to make some models
    #Lets start with the forest




def Forests(X_train,X_test,Y_train,Y_test):
    results = []
    #t_results = []
    for n in range(30,100):#for number of estimators
        #print n
        #build forest
        rfc = RandomForestClassifier(n_estimators=n)
        rfc.fit(X_train,Y_train)


        #test it out
        Y_pred = rfc.predict(X_test)#out of sample data
        #Yt_pred = rfc.predict(X_train)#with training data

        #get some metrics
        accuracy = accuracy_score(Y_test,Y_pred)#out of sample accuracy
        #t_accuracy = accuracy_score(Y_train,Yt_pred)#training data accuracy

        feature_imp = pd.Series(rfc.feature_importances_,index=X_train.columns.values)

        results.append((n,accuracy,feature_imp))
        #t_results.append((n,t_accuracy))





    x = []
    y = []
    tx = []
    ty = []
    for i in range(len(results)):
        #print results[i][0:2]
        x.append(results[i][0])#number of estimators
        y.append(results[i][1])#accuracy
    #for i in range(len(t_results)):
    #    tx.append(t_results[i][0])
    #    ty.append(t_results[i][1])


    plt.figure(1)#Accuracy and number of estimators
    plt.plot(x[:100],y[:100])
    plt.suptitle('Accuracy and Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.figure(2)#Feauture importances from most recent
    plt.suptitle('Feature Importances')
    plt.barh(X_train.columns.values,rfc.feature_importances_)
    plt.show()


#max from results is 37?
#feature importance lowest is median fee then average difficulty

    return


def Neighbors(X_train,X_test,Y_train,Y_test):
    results_uni = []
    results_dis = []

    #pca doesn't work too well
    for n in range(1,len(X_train.columns)):#number of components to be tested

        #first scale our attributes
        scaler = StandardScaler()
        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)
        pca = PCA(n_components=n)
        pca.fit(x_train)
        train_components = pca.transform(x_train)#now reduce dimensionality
        test_components = pca.transform(x_test)

        neigh_uni = KNeighborsClassifier(n_neighbors=3)#uniform weights
        neigh_dis = KNeighborsClassifier(n_neighbors=3,weights='distance')#weights based on distance

        neigh_uni.fit(train_components,Y_train)
        neigh_dis.fit(train_components,Y_train)

        Yu_pred = neigh_uni.predict(test_components)
        acc_uni = accuracy_score(Y_test,Yu_pred)
        Yd_pred = neigh_dis.predict(test_components)
        acc_dis = accuracy_score(Y_test,Yd_pred)

        print "N:",n, "Uniform accuracy:", acc_uni, "Distance accuracy:", acc_dis

        results_uni.append(acc_uni)
        results_dis.append(acc_dis)

    n = [i for i in range(1,101)]
    plt.figure(1)#Accuracy and number of neighbors
    plt.plot(n[:15],results_uni,color='b')
    plt.plot(n[:15],results_dis,color='r')
    plt.suptitle('Accuracy and Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.show()
#testing number of neighbors and uniform vs distance weighting - distance weighting looks slightly better
    for n in range(1,101):#number of neighbors to be tested

        neigh_uni = KNeighborsClassifier(n_neighbors=n)#uniform weights
        neigh_dis = KNeighborsClassifier(n_neighbors=n,weights='distance')#weights based on distance

        neigh_uni.fit(X_train,Y_train)
        neigh_dis.fit(X_train,Y_train)

        Yu_pred = neigh_uni.predict(X_test)
        acc_uni = accuracy_score(Y_test,Yu_pred)
        Yd_pred = neigh_dis.predict(X_test)
        acc_dis = accuracy_score(Y_test,Yd_pred)

        print "N:",n, "Uniform accuracy:", acc_uni, "Distance accuracy:", acc_dis

        results_uni.append(acc_uni)
        results_dis.append(acc_dis)
        #3 Neighbors is the best for uniform! sharp drop from 3-4 for uniform, 3-4 for distance is good

    n = [i for i in range(1,101)]
    plt.figure(1)#Accuracy and number of neighbors
    plt.plot(n[:100],results_uni[:100],color='b')
    plt.plot(n[:100],results_dis[:100],color='r')
    plt.suptitle('Accuracy and Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
#tested kd and ball tree, no difference with leaf size
    for n in range(1,101):#number of leaf size to be tested

        neigh_uni = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',leaf_size=n)#uniform weights
        neigh_dis = KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='kd_tree',leaf_size=n)#weights based on distance

        neigh_uni.fit(X_train,Y_train)
        neigh_dis.fit(X_train,Y_train)

        Yu_pred = neigh_uni.predict(X_test)
        acc_uni = accuracy_score(Y_test,Yu_pred)
        Yd_pred = neigh_dis.predict(X_test)
        acc_dis = accuracy_score(Y_test,Yd_pred)

        print "N:",n, "Uniform accuracy:", acc_uni, "Distance accuracy:", acc_dis

        results_uni.append(acc_uni)
        results_dis.append(acc_dis)
        #3 Neighbors is the best for uniform! sharp drop from 3-4 for uniform, 3-4 for distance is good

    n = [i for i in range(1,101)]
    plt.figure(1)#Accuracy and number of neighbors
    plt.plot(n[:100],results_uni[:100],color='b')
    plt.plot(n[:100],results_dis[:100],color='r')
    plt.suptitle('Accuracy and Leaf Size (3 Neighbors)')
    plt.xlabel('Leaf Size')
    plt.ylabel('Accuracy')
    plt.show()



    return


def Neural(X_train,X_test,Y_train,Y_test):
    results = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #Testing number of hidden layers and nuerons
    for n in range(1,len(X_train.columns)):
        #1
        """ann = MLPClassifier((n),solver='lbfgs',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #2
        ann = MLPClassifier((n,n),solver='lbfgs',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #3
        ann = MLPClassifier((n,n,n),solver='lbfgs',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #4
        ann = MLPClassifier((n,n,n,n),solver='lbfgs',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #5
        ann = MLPClassifier((n,n,n,n),solver='lbfgs',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(acc)
        results.append(n)"""

        ann = MLPClassifier((n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #2
        ann = MLPClassifier((n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #3
        ann = MLPClassifier((n,n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #4
        ann = MLPClassifier((n,n,n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(n)
        results.append(acc)
        #5
        ann = MLPClassifier((n,n,n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)
        acc = accuracy_score(Y_test,Y_pred)
        print "Nuerons:", n, "Accuracy:", acc
        results.append(acc)
        results.append(n)

        """res = 0
        for i in range(len(results)):
            if results[i][1] > res:
                res = results[i][0]

        for i in range(len(results)):
            if results[i][0] == i:
                print results[i]"""


            #test 'lbfgs', sgd, and adam for solver; logistic and relu for activation; hidden layer sizes
            #max iterations 500?

    return


def SVM(X_train,X_test,Y_train,Y_test):
    results = []
    #for i in range(1,11):
    svm = SVC(kernel='sigmoid',gamma='scale')
    svm.fit(X_train,Y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(Y_test,y_pred)
    print acc
    #    results.append((acc,i))
    #   print results
    #print max(results)
    #best results are with sigmoid scale
    return




def RunModels():
#INCLUDES RANDOM FOREST AND NEAREST NEIGHBORS (sklearn.neighbors)
#RUNS WITH 5 FOLDS AND COMPARES ACCURACY
#TO DO: PICK OUT A FEW MORE MODELS, TEST WITH DIFFERENT PARAMETERS (# Folds?, # trees? etc)

    data = GetData() #list of [X_train,X_test,Y_train,Y_test]
    X_train,X_test,Y_train,Y_test = data[0],data[1],data[2],data[3]

    Forests(X_train,X_test,Y_train,Y_test)
    Neighbors(X_train,X_test,Y_train,Y_test)
    Neural(X_train,X_test,Y_train,Y_test)
    SVM(X_train,X_test,Y_train,Y_test)

    #maybe try having the functions return their raw results - predictions
    #add repeated hold out with a for loop
    #Make some dataframes in this main function
    #run all predictions through a forest?

RunModels()




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
