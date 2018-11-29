from __future__                import division
import numpy                   as np
import pandas                  as pd
import matplotlib.pyplot       as plt
import matplotlib.patches      as mpatches

from sklearn.model_selection   import cross_val_score
from sklearn.model_selection   import KFold
from sklearn.metrics           import accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.decomposition     import PCA
from sklearn.neural_network    import MLPClassifier
from sklearn.svm               import SVC




def GetData():#returns dataframe of all data including target class

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

    return bitcoin

def PrepData(dataframe):#splits data into folds and returns test and train sets

    #Lets make some FOLDS!
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(dataframe):
        train, test = dataframe.iloc[train_index], dataframe.iloc[test_index]


    #Now split into x and y
    Y_train, Y_test = train.y, test.y
    X_train, X_test = train.drop(columns='y'), test.drop(columns='y')

    #print "Xtrain", X_train, "X_test", X_test, "Y_train", Y_train, "Y_test", Y_test

    return [X_train,X_test,Y_train,Y_test]


#So we have our training and testing data...
#Time to make some models
#Lets start with the forest



def ForestTester(X_train,X_test,Y_train,Y_test):
    accuracy = []
    auroc = []
    kappa = []
    #t_results = []
    for n in range(1,101):  #for number of estimators
        #build forest
        rfc = RandomForestClassifier(n_estimators=n)
        rfc.fit(X_train,Y_train)


        #test it out
        Y_pred = rfc.predict(X_test)#out of sample data
        #Yt_pred = rfc.predict(X_train)#with training data

        y_score = rfc.predict_proba(X_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)
        #print "n:",n, acc
        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)

    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]


    return results


#KNeighbors testing for good number of neighbors - 28
def NumberNeighborsTester(X_train,X_test,Y_train,Y_test):
    accuracy = []
    auroc = []
    kappa = []

    for n in range(1,101):#number of neighbors to be tested

        neigh_uni = KNeighborsClassifier(n_neighbors=n)#uniform weights
        #neigh_dis = KNeighborsClassifier(n_neighbors=n,weights='distance')#weights based on distance

        neigh_uni.fit(X_train,Y_train)


        Y_pred = neigh_uni.predict(X_test)

        y_score = neigh_uni.predict_proba(X_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)
        #print "n:",n, acc
        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)


    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]


    return results

#testing for uniform vs weighted
def WeightsNeighborsTester(X_train,X_test,Y_train,Y_test):
#tested kd and ball tree, no difference with leaf size
    #for n in range(2):

    neigh_uni = KNeighborsClassifier(n_neighbors=28,weights='uniform')#uniform weights
    neigh_dis = KNeighborsClassifier(n_neighbors=28,weights='distance')#weights based on distance

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
    plt.plot(n[:],results_uni[:],color='b')
    plt.plot(n[:],results_dis[:],color='r')
    plt.suptitle('Accuracy and Leaf Size (3 Neighbors)')
    plt.xlabel('Leaf Size')
    plt.ylabel('Accuracy')
    plt.show()



    return
#testing for good number of PCA components - 14
def PCANeighborsTester(X_train,X_test,Y_train,Y_test):
    #pca
    accuracy = []
    auroc = []
    kappa = []
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

        #neigh_uni = KNeighborsClassifier(n_neighbors=28)#uniform weights
        neigh_dis = KNeighborsClassifier(n_neighbors=28,weights='distance')#weights based on distance
        neigh_dis.fit(train_components,Y_train)
        Y_pred = neigh_dis.predict(test_components)

        y_score = neigh_dis.predict_proba(test_components)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)

        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)



    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]

    return results




        #print "N:",n, "Uniform accuracy:", acc_uni, "Distance accuracy:", acc_dis

"""
    n = [i for i in range(1,101)]
    plt.figure(1)#Accuracy and number of neighbors
    plt.plot(n[:15],results_uni,color='b')
    plt.plot(n[:15],results_dis,color='r')
    plt.suptitle('Accuracy and Number of Components with PCA')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.show()"""



def NeuralTester1(X_train,X_test,Y_train,Y_test):
    results = []
    accuracy = []
    auroc = []
    kappa = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #testing number of neurons
    for n in range(1,len(X_train.columns)):
        ann = MLPClassifier((n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)

        y_score = ann.predict_proba(x_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)

        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)


    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]

    return results

def NeuralTester2(X_train,X_test,Y_train,Y_test):
    results = []
    accuracy = []
    auroc = []
    kappa = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #testing number of neurons
    for n in range(1,len(X_train.columns)):
        ann = MLPClassifier((n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)

        y_score = ann.predict_proba(x_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)

        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)


    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]

    return results

def NeuralTester3(X_train,X_test,Y_train,Y_test):
    results = []
    accuracy = []
    auroc = []
    kappa = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #testing number of neurons
    for n in range(1,len(X_train.columns)):
        ann = MLPClassifier((n,n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)

        y_score = ann.predict_proba(x_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)

        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)


    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]

    return results

def NeuralTester4(X_train,X_test,Y_train,Y_test):
    results = []
    accuracy = []
    auroc = []
    kappa = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #testing number of neurons
    for n in range(1,len(X_train.columns)):
        ann = MLPClassifier((n,n,n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)

        y_score = ann.predict_proba(x_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)

        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)


    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]

    return results

def NeuralTester5(X_train,X_test,Y_train,Y_test):
    results = []
    accuracy = []
    auroc = []
    kappa = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #testing number of neurons
    for n in range(1,len(X_train.columns)):
        ann = MLPClassifier((n,n,n,n,n),solver='sgd',activation='logistic',max_iter=500)
        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)

        y_score = ann.predict_proba(x_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)

        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)


    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]

    return results

def NeuralTesterFin(X_train,X_test,Y_train,Y_test,n1,n2,n3,n4,n5):
    results = []
    accuracy = []
    auroc = []
    kappa = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #testing number of neurons
    for i in range(1,6):
        if (i==1):
            ann = MLPClassifier((n1),solver='sgd',activation='logistic',max_iter=500)

        elif (i==2):
            ann = MLPClassifier((n2,n2),solver='sgd',activation='logistic',max_iter=500)

        elif (i==3):
            ann = MLPClassifier((n3,n3,n3),solver='sgd',activation='logistic',max_iter=500)

        elif (i==4):
            ann = MLPClassifier((n4,n4,n4,n4),solver='sgd',activation='logistic',max_iter=500)

        elif (i==5):
            ann = MLPClassifier((n5,n5,n5,n5,n5),solver='sgd',activation='logistic',max_iter=500)

        ann.fit(x_train,Y_train)
        Y_pred = ann.predict(x_test)

        y_score = ann.predict_proba(x_test)#scores for stats
        acc = accuracy_score(Y_test,Y_pred)
        auc = roc_auc_score(Y_test,y_score[:,1])
        kappa_score = cohen_kappa_score(Y_test,Y_pred)

        accuracy.append(acc)
        auroc.append(auc)
        kappa.append(kappa_score)


    results = [np.asarray(accuracy),np.asarray(auroc),np.asarray(kappa)]
    print results

    return results



def SVMTester(X_train,X_test,Y_train,Y_test):
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


def EvalResults(result_list,title='',xlab='',ylab='Accuracy'):#pass in list with [[acc],[auroc],[kappa]]
#show Plots
#return best option
    #print "result_list",result_list
    accuracies = np.asarray(result_list[0][0])
    auroc = np.asarray(result_list[0][1])
    kappa = np.asarray(result_list[0][2])
    #print "Accuracies from assignment:",accuracies
    div = len(result_list)
    for i in range(1,div):
        accuracies += result_list[i][0]
        auroc += result_list[i][1]
        kappa += result_list[i][2]

    #print "Accuracies before dividing:",accuracies
    accuracies = accuracies/div
    auroc = auroc/div
    kappa = kappa/div

    #accuracies = np.asarray(accuracies)
    #print "Accuracies after dividing:",accuracies
    x=[i for i in range(1,len(accuracies)+1)]
    #x=[i for i in range(1,accuracies+1)]

    plt.plot(x,accuracies,'r')
    plt.plot(x,auroc,'b')
    plt.plot(x,kappa,'g')
    red_patch = mpatches.Patch(color='red', label='Accuracy')
    blue_patch = mpatches.Patch(color='blue', label='AUROC')
    green_patch = mpatches.Patch(color='green', label='Kappa')
    plt.legend(handles=[red_patch,blue_patch,green_patch])
    plt.suptitle(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.show()
    #now find optimal number
    maxVal = accuracies[0] + auroc[0] + kappa[0]
    ind = 0
    for i in range(len(accuracies)):
        new = accuracies[i] + auroc[i] + kappa[i]
        if new > maxVal:
            maxVal = new
            ind = i

    ind += 1

    return ind

def Forests(X_train,X_test,Y_train,Y_test,n):
    results = []
    rfc = RandomForestClassifier(n_estimators=n)
    rfc.fit(X_train,Y_train)


    #test it out
    Y_pred = rfc.predict(X_test)#out of sample data
    #Yt_pred = rfc.predict(X_train)#with training data

    y_score = rfc.predict_proba(X_test)#scores for stats
    acc = accuracy_score(Y_test,Y_pred)



    print "Accuracy for random forest with {} estimators: {}".format(n,acc)






    plt.figure(2)#Feauture importances from most recent
    plt.suptitle('Feature Importances')
    plt.barh(X_train.columns.values,rfc.feature_importances_)
    plt.show()




    return

def Neighbors(X_train,X_test,Y_train,Y_test,n_neighs,n_compons):
    results = []


        #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    pca = PCA(n_components=n_compons)
    pca.fit(x_train)
    train_components = pca.transform(x_train)#now reduce dimensionality
    test_components = pca.transform(x_test)

    neigh_dis = KNeighborsClassifier(n_neighbors=n_neighs,weights='distance')#weights based on distance

    neigh_dis.fit(train_components,Y_train)

    Yd_pred = neigh_dis.predict(test_components)
    acc_dis = accuracy_score(Y_test,Yd_pred)

    print "Accuracy for nearest neighbors with {} neighbors and {} principal components: {}".format(n_neighs,n_compons,acc_dis)


    """n = [i for i in range(1,101)]
    plt.figure(1)#Accuracy and number of neighbors
    plt.plot(n[:15],results_uni,color='b')
    plt.plot(n[:15],results_dis,color='r')
    plt.suptitle('Accuracy and Number of Components with PCA')
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


"""
    return

def Neural(X_train,X_test,Y_train,Y_test,layers,n):
    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    if (layers==1):
        ann = MLPClassifier((n),solver='lbfgs',max_iter=500)

    elif (layers==2):
        ann = MLPClassifier((n,n),solver='lbfgs',max_iter=500)

    elif (layers==3):
        ann = MLPClassifier((n,n,n),solver='lbfgs',max_iter=500)

    elif (layers==4):
        ann = MLPClassifier((n,n,n,n),solver='lbfgs',max_iter=500)

    elif (layers==5):
        ann = MLPClassifier((n,n,n,n,n),solver='lbfgs',max_iter=500)


    ann.fit(x_train,Y_train)
    Y_pred = ann.predict(x_test)
    acc = accuracy_score(Y_test,Y_pred)

    print "Accuracy for MLPClassifier with {} layers and {} neurons: {}".format(layers,n,acc)

    return
    """results = []

    #first scale our attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    #Testing number of hidden layers and nuerons
    for n in range(1,len(X_train.columns)):
        #1
        ann = MLPClassifier((n),solver='lbfgs',max_iter=500)
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
        results.append(n)

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



            #test 'lbfgs', sgd, and adam for solver; logistic and relu for activation; hidden layer sizes
            #max iterations 500?

    return"""


def SVM(X_train,X_test,Y_train,Y_test):
    results = []
    #for i in range(1,11):
    svm = SVC(kernel='sigmoid',gamma='scale')
    svm.fit(X_train,Y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(Y_test,y_pred)
    print "Accuracy for svm with sigmoid kernel:",acc
    #    results.append((acc,i))
    #   print results
    #print max(results)
    #best results are with sigmoid scale
    return



def RunModels():
    bitcoin = GetData() #dataframe with all features

#lets do 10 iterations of testing

    forestRes = []
    numberNeigh = []
    PCANeigh = []
    nn1 = []
    nn2 = []
    nn3 = []
    nn4 = []
    nn5 = []
    nn = []
    #nFin = []
    numNeur = []
    for i in range(10):
        data = PrepData(bitcoin) #list of [X_train,X_test,Y_train,Y_test]
        X_train,X_test,Y_train,Y_test = data[0],data[1],data[2],data[3]
        res = ForestTester(X_train,X_test,Y_train,Y_test)
        forestRes.append(res)
        a = NumberNeighborsTester(X_train,X_test,Y_train,Y_test)
        b = PCANeighborsTester(X_train,X_test,Y_train,Y_test)
        numberNeigh.append(a)
        PCANeigh.append(b)
        n1 = NeuralTester1(X_train,X_test,Y_train,Y_test)
        n2 = NeuralTester2(X_train,X_test,Y_train,Y_test)
        n3 = NeuralTester3(X_train,X_test,Y_train,Y_test)
        n4 = NeuralTester4(X_train,X_test,Y_train,Y_test)
        n5 = NeuralTester5(X_train,X_test,Y_train,Y_test)
        nn1.append(n1)
        nn2.append(n2)
        nn3.append(n3)
        nn4.append(n4)
        nn5.append(n5)
        if (i==9):
            an1 = EvalResults(nn1,'1 layer Neural Network','Number of Neurons')
            an2 = EvalResults(nn2,'2 layer Neural Network','Number of Neurons')
            an3 = EvalResults(nn3,'3 layer Neural Network','Number of Neurons')
            an4 = EvalResults(nn4,'4 layer Neural Network','Number of Neurons')
            an5 = EvalResults(nn5,'5 layer Neural Network','Number of Neurons')
            print "an1:",an1,"an2:",an2,"an3:",an3,"an4:",an4,"an5:",an5
            an = NeuralTesterFin(X_train,X_test,Y_train,Y_test,an1,an2,an3,an4,an5)
            nn.append(an)
            nFin = EvalResults(nn)
            print nFin
            numNeur.append(an1)
            numNeur.append(an2)
            numNeur.append(an3)
            numNeur.append(an4)
            numNeur.append(an5)





    ind = EvalResults(forestRes,'Number of Estimators in Forest','Number of Estimators')
    print "Optimal number of trees:",ind
    #print numberNeigh
    numNeigh = EvalResults(numberNeigh,'Number of Neighbors in KNeighbors','Number of Neighbors')
    print "Optimal number of neighbors:",numNeigh
    numPCA = EvalResults(PCANeigh,'Number of Components From PCA','Number of Components')
    print "Optimal number of components from PCA:",numPCA


    Forests(X_train,X_test,Y_train,Y_test,ind)
    Neighbors(X_train,X_test,Y_train,Y_test,numNeigh,numPCA)
    Neural(X_train,X_test,Y_train,Y_test,nFin,numNeur[nFin-1])
    SVM(X_train,X_test,Y_train,Y_test)



    """nn.append(nn1)
    nn.append(nn2)
    nn.append(nn3)
    nn.append(nn4)
    nn.append(nn5)
    ans = EvalResults(nn)
    print nn"""
    """an1 = EvalResults(nn1)
    an2 = EvalResults(nn2)
    an3 = EvalResults(nn3)
    an4 = EvalResults(nn4)
    an5 = EvalResults(nn5)
    print "an1:",an1,"an2:",an2,"an3:",an3,"an4:",an4,"an5:",an5"""




    #print 'accuracies',accuracies,'auroc',auroc,'kappa',kappa

    #print len(forestRes)
    #print len(forestRes[0])

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
