from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
import Confusion_Matrix
import Read_Data
import Method2
from numpy import nan


def getdata():
    # extracts data from the files
    features, target = Read_Data.read_both_data('mushrooms_data.csv','mushrooms_data_missing.csv')
    target = target.replace(nan, -1)
    return features, target


def label_spr(features,target):
    # label spreading
    label_spread = LabelSpreading(kernel='rbf', alpha=0.8, max_iter=25)
    label_spread.fit(features,target)
    output_target = label_spread.transduction_
    return output_target


def label_prop(features,target):
    # label propagation
    label_propagation = LabelPropagation(kernel='rbf', max_iter=500)
    label_propagation.fit(features,target)
    output_target = label_propagation.transduction_
    return output_target


def semi_supervised_label_propagation():
    # performs learning after label propagation
    features,target = getdata()
    output_target = label_prop(features,target)
    X_train, X_test, y_train, y_test = train_test_split(features, output_target, test_size=0.20, random_state=100)
    mlp = Method2.mlp_clf(X_train,y_train)
    predicted = Method2.predicts(mlp, X_test)
    actual = y_test
    Confusion_Matrix.main(actual, predicted, "Confusion Matrix: Label_Propagation")


def semi_supervised_label_spreading():
    # performs learning after label spreading
    features,target = getdata()
    output_target = label_spr(features,target)
    X_train, X_test, y_train, y_test = train_test_split(features, output_target, test_size=0.20, random_state=100)
    mlp = Method2.mlp_clf(X_train,y_train)
    predicted = Method2.predicts(mlp, X_test)
    actual = y_test
    Confusion_Matrix.main(actual, predicted, "Confusion Matrix: Label_Spreading")


def self_training_clf(features,target):
    # builds a self training classifier
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 30, 30), random_state=100, max_iter=100)
    self_training_model = SelfTrainingClassifier(mlp)
    self_training_model.fit(features, target)
    return self_training_model


def split(features,target):
    # splits the data for the self learning classifier
    regfeatures, regtarget= features[:6498],target[:6498]
    missing_features, missing_target = features[6498:],target[6498:]
    X_train, X_test, y_train, y_test = train_test_split(regfeatures, regtarget, test_size=0.20,random_state=100)
    X_train = X_train.append(missing_features, ignore_index=True)
    y_train = y_train.append(missing_target, ignore_index=True)
    return X_train, X_test, y_train, y_test


def semi_supervised_self_training():
    # performs self learning
    features, target = getdata()
    X_train, X_test, y_train, y_test = split(features,target)
    self_training_model = self_training_clf(X_train,y_train)
    predicted = self_training_model.predict(X_test)
    actual = y_test.values
    Confusion_Matrix.main(actual, predicted, "Confusion Matrix: Self-Training")


def main():
    print('1-label-propagation')
    print('2-label-spreading')
    print('3-self learning')
    option = input('Enter the number of the preferred option')
    option = int(option)
    if (option == 1):
        semi_supervised_label_propagation()
    elif (option == 2):
        semi_supervised_label_spreading()
    elif (option == 3):
        semi_supervised_self_training()
    else:
        print('wrong number')
