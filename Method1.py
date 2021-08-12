import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import recall_score
from kmodes.kmodes import KModes
import Read_Data
import Confusion_Matrix
from sklearn.cluster import SpectralClustering
from scipy.stats import mode


def getdata():
    data, features, target = Read_Data.read_data2('mushrooms_data.csv')
    X = Read_Data.get_features(data, features)
    y = Read_Data.get_target(data, target)
    return data, X,y


def cluster1(data,number):
    # K-means
   kmean = KMeans(n_clusters=number, random_state=100)
   clusters = kmean.fit_predict(data)
   print('kmean')
   return clusters


def cluster2(data, number):
    # Spectral clustering
    spectral = SpectralClustering(n_clusters=number,random_state=100)
    model = spectral.fit(data)
    clusters = model.labels_
    print('spectral')
    return clusters


def cluster3(data, number):
    # GMM
    gmm = GaussianMixture(n_components=number,n_init=30,random_state=100)
    clusters = gmm.fit_predict(data)
    print('GMM')
    return clusters


def predicts(clusters,target):
    # predicts and classifies by the clustering
    labels = np.zeros_like(clusters)
    target= target.values
    target = Read_Data.reverse_odor(target)
    for i in range(9):
        mask = (clusters == i)
        labels[mask] = mode(target[mask])[0]
    return labels


def k_means():
    # performs classification with K-means
    data, X, y = getdata()
    clusters = cluster1(data, 8)
    predicted = predicts(clusters, y)
    actual = y.values
    actual = Read_Data.reverse_odor(actual)
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: K-Means")
    result = [f1, recall, accuracy]
    with open('results_folder/k_means_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def spectral():
    # performs classification with spectral clustering
    data, X, y = getdata()
    clusters = cluster2(data,12)
    predicted = predicts(clusters, y)
    actual = y.values
    actual = Read_Data.reverse_odor(actual)
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: Spectral Clustering")
    result = [f1, recall, accuracy]
    with open('results_folder/spectral_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)

def gmm():
    # performs classification with GMM
    data, X, y = getdata()
    clusters = cluster3(data, 13)
    predicted = predicts(clusters, y)
    actual = y.values
    actual = Read_Data.reverse_odor(actual)
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted, "Confusion Matrix: GMM")
    result = [f1, recall, accuracy]
    with open('results_folder/gmm_result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)


def main():
    print('1-Kmeans')
    print('2-Spectral clustering')
    print('3-GMM')
    algo_number = input('Enter the number of the preferred clustering method')
    algo_number = int(algo_number)
    if (algo_number == 1):
        k_means()
    elif(algo_number == 2):
        spectral()
    elif(algo_number == 3):
        gmm()
    else:
        print('wrong number')
