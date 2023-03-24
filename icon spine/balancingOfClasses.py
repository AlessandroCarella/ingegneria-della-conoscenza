import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample

from stampe import prYellow, prGreenMoreString, prRedMoreString


def resampleDataset(dataSet, differentialColumn):
    dataSet.drop(dataSet[(dataSet[differentialColumn] != 0) & (
        dataSet[differentialColumn] != 1)].index, inplace=True)
    df_minority = dataSet[dataSet[differentialColumn] == 1]
    df_majority = dataSet[dataSet[differentialColumn] == 0]

    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    dataSet = pd.concat([df_minority_upsampled, df_majority])

    prYellow("\nValue after Oversampling:")
    prGreenMoreString('Abnormal spine: ', dataSet.spine_state.value_counts()[0],
                      '(% {:.2f})'.format(dataSet.spine_state.value_counts()[0] / dataSet.spine_state.count() * 100))
    prRedMoreString('Normal spine: ', dataSet.spine_state.value_counts()[1],
                    '(% {:.2f})'.format(dataSet.spine_state.value_counts()[1] / dataSet.spine_state.count() * 100))

    return dataSet


def visualizeAspectRatioChart(dataSet, differentialColumn, labels=["Abnormal spine", "Normal spine"]):
    # Visualization of the aspect ratio chart
    ax = dataSet[differentialColumn].value_counts().plot(
        kind='pie', figsize=(5, 5), labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of Normal spine and Abnormal spine")
    plt.legend(labels=labels, loc="best")
    # plt.show()
    plt.clf()


def visualizeNumberOfSamplesForClasses(dataSet):
    # Proportion of Abnormal spine (0) and Normal spine (1):
    # [Number of Abnormal spine raws/Total number of Normal spine raws]
    prGreenMoreString('Abnormal spine: ', dataSet.spine_state.value_counts()[0],
                      '(% {:.2f})'.format(dataSet.spine_state.value_counts()[0] / dataSet.spine_state.count() * 100))
    prRedMoreString('Normal spine: ', dataSet.spine_state.value_counts()[1],
                    '(% {:.2f})'.format(dataSet.spine_state.value_counts()[1] / dataSet.spine_state.count() * 100))
