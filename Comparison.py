import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





def f1_comparison(f1_scores,method_list):
    # Creating a F1 comparison chart
    f1_score = pd.Series(f1_scores, index=method_list).sort_values(ascending=False)
    sns.barplot(x = f1_score, y = f1_score.index)
    print('f1_score')
    plt.xlabel('F1 Score')
    plt.ylabel('Methods')
    plt.title("Comparing F1 Scores")
    plt.legend()
    plt.show()


def recall_comparison(recall_scores,method_list):
    # Creating a recall comparison chart
    recall_score = pd.Series(recall_scores, index=method_list).sort_values(ascending=False)
    sns.barplot(x = recall_score, y = recall_score.index)
    print('recall_score')
    plt.xlabel('Recall Score')
    plt.ylabel('Methods')
    plt.title("Comparing Recall Scores")
    plt.legend()
    plt.show()


def accuracy_comparison(accuracy_scores,method_list):
    # Creating an accuracy comparison chart
    accuracy_score = pd.Series(accuracy_scores, index=method_list).sort_values(ascending=False)
    sns.barplot(x = accuracy_score, y = accuracy_score.index)
    print('accuracy_score')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Methods')
    plt.title("Comparing Accuracy Scores")
    plt.legend()
    plt.show()

def create_scores():
    # getting the scores from the results files
    method_list = ['K-means','Spectral Clustering','GMM','Random Forest','MLP','Neural Network',
                   'Random Forest with Importance','GMM with PCA','MLP with PCA','NN with PCA']
    f1_scores = []
    recall_scores = []
    accuracy_scores = []

    df1 = pd.read_csv('results_folder/k_means_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0, 0])
    recall_scores.append(df1[0, 1])
    accuracy_scores.append(df1[0, 2])

    df1 = pd.read_csv('results_folder/spectral_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0, 0])
    recall_scores.append(df1[0, 1])
    accuracy_scores.append(df1[0, 2])

    df1 = pd.read_csv('results_folder/gmm_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0, 0])
    recall_scores.append(df1[0, 1])
    accuracy_scores.append(df1[0, 2])

    df1 = pd.read_csv('results_folder/forest_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0, 0])
    recall_scores.append(df1[0, 1])
    accuracy_scores.append(df1[0, 2])

    df1 = pd.read_csv('results_folder/mlp_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0, 0])
    recall_scores.append(df1[0, 1])
    accuracy_scores.append(df1[0, 2])

    df1 = pd.read_csv('results_folder/nn_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0, 0])
    recall_scores.append(df1[0, 1])
    accuracy_scores.append(df1[0, 2])

    df1 = pd.read_csv('results_folder/forest_fi_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0, 0])
    recall_scores.append(df1[0, 1])
    accuracy_scores.append(df1[0, 2])

    df1 = pd.read_csv('results_folder/cluster_result.csv', names=range(3))
    df1 = df1.values
    f1_scores.append(df1[0,0])
    recall_scores.append(df1[0,1])
    accuracy_scores.append(df1[0,2])

    df2 = pd.read_csv('results_folder/mlp_pca_result.csv', names=range(3))
    df2 = df2.values
    f1_scores.append(df2[0,0])
    recall_scores.append(df2[0,1])
    accuracy_scores.append(df2[0,2])

    df3 = pd.read_csv('results_folder/nn_pca_result.csv', names=range(3))
    df3 = df3.values
    f1_scores.append(df3[0,0])
    recall_scores.append(df3[0,1])
    accuracy_scores.append(df3[0,2])
    return method_list, f1_scores, recall_scores, accuracy_scores


def main():
    method_list, f1_scores, recall_scores, accuracy_scores = create_scores()
    f1_comparison(f1_scores,method_list)
    recall_comparison(recall_scores, method_list)
    accuracy_comparison(accuracy_scores, method_list)


