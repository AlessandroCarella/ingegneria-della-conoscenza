import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample

from stampe import prGreenMoreString, prRedMoreString, prYellow


def resampleDataset(dataSet, differentialColumn):
    dataSet.drop(dataSet[(dataSet[differentialColumn] != 0) & (
        dataSet[differentialColumn] != 1)].index, inplace=True)
    df_minority = dataSet[dataSet[differentialColumn] == 0]
    df_majority = dataSet[dataSet[differentialColumn] == 1]

    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    dataSet = pd.concat([df_minority_upsampled, df_majority])

    prYellow("\nValue after Oversampling:")
    prGreenMoreString('Not liked songs: ', dataSet.songIsLiked.value_counts()[0],
                      '(% {:.2f})'.format(dataSet.songIsLiked.value_counts()[0] / dataSet.songIsLiked.count() * 100))
    prRedMoreString('Liked songs: ', dataSet.songIsLiked.value_counts()[1],
                    '(% {:.2f})'.format(dataSet.songIsLiked.value_counts()[1] / dataSet.songIsLiked.count() * 100))

    return dataSet


def visualizeAspectRatioChart(dataSet, differentialColumn, labels=["Liked songs", "Not liked songs"]):
    ax = dataSet[differentialColumn].value_counts().plot(
        kind='pie', figsize=(5, 5), labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of liked songs and not like songs")
    plt.legend(labels=labels, loc="best")
    plt.show()
    plt.clf()


def visualizeNumberOfSamplesForClasses(dataSet):
    prGreenMoreString('Not liked songs: ', dataSet.songIsLiked.value_counts()[0],
                      '(% {:.2f})'.format(dataSet.songIsLiked.value_counts()[0] / dataSet.songIsLiked.count() * 100))
    prRedMoreString('Liked songs: ', dataSet.songIsLiked.value_counts()[1],
                    '(% {:.2f})'.format(dataSet.songIsLiked.value_counts()[1] / dataSet.songIsLiked.count() * 100))


def undersampleDataset(dataSet, differentialColumn):
    dataSet.drop(dataSet[(dataSet[differentialColumn] != 0) & (
        dataSet[differentialColumn] != 1)].index, inplace=True)
    df_minority = dataSet[dataSet[differentialColumn] == 0]
    df_majority = dataSet[dataSet[differentialColumn] == 1]

    n_samples = min(len(df_minority), len(df_majority), len(dataSet)//15)
    n_samples_minority = int(n_samples / 2)
    n_samples_majority = int(n_samples / 2)

    df_minority_undersampled = df_minority.sample(
        n=n_samples_minority, random_state=42)
    df_majority_undersampled = df_majority.sample(
        n=n_samples_majority, random_state=42)
    dataSet = pd.concat([df_minority_undersampled, df_majority_undersampled])

    prYellow("\nValue after Undersampling:")
    prGreenMoreString('Not liked songs: ', dataSet.songIsLiked.value_counts()[0],
                      '(% {:.2f})'.format(dataSet.songIsLiked.value_counts()[0] / dataSet.songIsLiked.count() * 100))
    prRedMoreString('Liked songs: ', dataSet.songIsLiked.value_counts()[1],
                    '(% {:.2f})'.format(dataSet.songIsLiked.value_counts()[1] / dataSet.songIsLiked.count() * 100))

    return dataSet
