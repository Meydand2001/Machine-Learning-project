import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import Read_Data
import Confusion_Matrix
from sklearn.model_selection import train_test_split
import NN


def getdata2():
    # extracts data for the mlp and neural network
    X, y = Read_Data.read_data3('mushrooms_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 100)
    return X_train, X_test, y_train, y_test


def getdata():
    # extracts data for the random forest
    data, features, target = Read_Data.read_data('mushrooms_data.csv')
    X = Read_Data.get_features(data, features)
    y = Read_Data.get_target(data, target)
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state = 100)
    return X_train, X_test, y_train, y_test


def random_forest(X_train,y_train):
    #builds random forest
    clf = RandomForestClassifier(n_estimators=15)
    clf = clf.fit(X_train,y_train)
    return clf


def predicts(clf, X_test):
    # predicts results of a classifier
    x = X_test.values
    predicted = clf.predict(x)
    return predicted


def random_forest_im():
    # performs classification with random forest
    X_train, X_test, y_train, y_test = getdata()
    rforest=random_forest(X_train, y_train)
    predicted = predicts(rforest,X_test)
    actual = y_test.values
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: Random Forest")
    result = [f1, recall, accuracy]
    with open('results_folder/forest_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def mlp_clf(X_train,y_train):
    # build mlp classifier
    mlp = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(200,30,30), random_state=100 ,max_iter=100)
    mlp.fit(X_train,y_train)
    return mlp


def mlp():
    # performs classification with mlp
    X_train, X_test, y_train, y_test = getdata2()
    mlp = mlp_clf(X_train, y_train)
    predicted = predicts(mlp, X_test)
    actual = y_test.values
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: MLP")
    result = [f1, recall, accuracy]
    with open('results_folder/mlp_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def nn():
    # performs classification with neural network
    X_train, X_test, y_train, y_test = getdata2()
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    f1, recall, accuracy = NN.main(X_train, X_test, y_train, y_test, input_size=101)
    result = [f1, recall, accuracy]
    with open('results_folder/nn_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def main():
    print('1-Random Forest')
    print('2-MLP')
    print('3-Neural Network')
    algo_number = input('Enter the number of the preferred classifier method')
    algo_number = int(algo_number)
    if (algo_number == 1):
        random_forest_im()
    elif (algo_number == 2):
        mlp()
    elif (algo_number == 3):
        nn()
    else:
        print('wrong number')
