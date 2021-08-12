import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.metrics import recall_score, f1_score, accuracy_score


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

def conf_matrix(actual, predicted):
    num_clusters = 9
    conf_mat = np.empty((num_clusters, num_clusters))
    for row in range(num_clusters):
        for col in range(num_clusters):
            counter = 0
            for i in range(len(actual)):
                if predicted[i] == row and actual[i] == col:
                    counter += 1
            conf_mat[row, col] = counter
    return conf_mat


def main(actual, predicted, name):
    fig, ax = plt.subplots()
    conf_mat = conf_matrix(actual,predicted)
    clusters = ['0','1', '2', '3', '4', '5', '6', '7', '8']
    im = heatmap(conf_mat, clusters, clusters, ax=ax, cmap="pink_r")
    annotate_heatmap(im, valfmt="{x:.1f}")
    plt.title(name)
    plt.xlabel('real values')
    plt.ylabel('predicted values')

    plt.show()
    print("Recall value:")
    recall = recall_score(actual, predicted, average='macro').round(3)
    print(recall)
    print("F1 value")
    f1 = f1_score(actual, predicted, average='macro')
    print(f1)
    print("Accuracy_score")
    accuracy = accuracy_score(actual, predicted)
    print(accuracy)

    return f1, recall, accuracy