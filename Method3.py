import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import Pca
import NN
import csv
import Read_Data
import Confusion_Matrix
from Method1 import cluster3, predicts
from Method2 import mlp_clf


def getdata():
    # extracts data for the random forest
    data, features, target = Read_Data.read_data('mushrooms_data.csv')
    X = Read_Data.get_features(data, features)
    y = Read_Data.get_target(data, target)
    return data, X, y


def getdata2():
    # extracts data for the clustering
    data, features, target = Read_Data.read_data2('mushrooms_data.csv')
    X = Read_Data.get_features(data, features)
    y = Read_Data.get_target(data, target)
    return data, X, y


def getdata3():
    # extracts data for the mlp and neural network
    features, target = Read_Data.read_data3('mushrooms_data.csv')
    return features , target


def random_forest(X_train,y_train):
    # builds a random forest
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X_train,y_train)
    return clf


def predictsf(clf,X_test):
    # predicts results of a classifier
    predicted = clf.predict(X_test)
    return predicted


def feature_importance(clf):
    # creates a table of feature importance
    feature_imp = pd.Series(clf.feature_importances_, index=Read_Data.get_featuretags()).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()
    return feature_imp


def forest():
    # performs random forest classifying with important features
    data,X,y = getdata()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
    clf = random_forest(X_train,y_train)
    feature_imp = feature_importance(clf)[0:11]
    imp_features=feature_imp.index
    newX_train = Read_Data.get_features(X_train, imp_features)
    newX_test = Read_Data.get_features(X_test, imp_features)
    rforest = random_forest(newX_train, y_train)
    predicted = predictsf(rforest, newX_test)
    actual = y_test.values
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: Random Forest with Feature importance")
    result = [f1, recall, accuracy]
    with open('results_folder/forest_fi_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def cluster():
    # performs gmm clustering and classifying with pca
    data,X,y= getdata2()
    new_data= Pca.pca(data,35)
    clusters = cluster3(new_data, 13)
    predicted = predicts(clusters, y)
    actual = y.values
    actual = Read_Data.reverse_odor(actual)
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: GMM with PCA")
    result = [f1,recall,accuracy]
    with open('results_folder/cluster_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def mlp_pca():
    # performs mlp classifying with pca
    features, target = getdata3()
    new_features = Pca.pca(features, 35)
    new_features = pd.DataFrame(new_features)
    X_train, X_test, y_train, y_test = train_test_split(new_features, target, test_size=0.20, random_state=100)
    mlp = mlp_clf(X_train, y_train)
    predicted = predictsf(mlp, X_test)
    actual = y_test.values
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: MLP with PCA")
    result = [f1, recall, accuracy]
    with open('results_folder/mlp_pca_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def nn_pca():
    # performs neural network classifying with pca
    features, target = getdata3()
    new_features = Pca.pca(features,35)
    X_train, X_test, y_train, y_test = train_test_split(new_features, target, test_size=0.20, random_state=100)
    y_train = y_train.values
    y_test = y_test.values
    f1, recall, accuracy = NN.main(X_train, X_test, y_train, y_test, input_size=35)
    result = [f1, recall, accuracy]
    with open('results_folder/nn_pca_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def main():
    print('1-Random Forest with feature importance')
    print('2-Clustering with PCA')
    print('3-MLP with PCA')
    print('4-Neural Network with PCA')
    algo_number = input('Enter the number of the preferred method')
    algo_number = int(algo_number)
    if (algo_number == 1):
        forest()
    elif (algo_number == 2):
        cluster()
    elif (algo_number == 3):
        mlp_pca()
    elif (algo_number == 4):
        nn_pca()
    else:
        print('wrong number')
