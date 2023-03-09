# dataset: https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset
# Main libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sys
import time
import multiprocessing
import itertools
import threading

# Classification models
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

# Machine learning
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold, cross_val_score

# Classification algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import warnings

import os

# utility


def autopct(pct):
    # shows only values of labers that are greater than 1%
    return ('%.2f' % pct + "%") if pct > 1 else ''


def prGreenMoreString(prt, prt2, prt3):
    print("\n\033[92m{}\033[00m".format(prt), prt2, prt3)


def prRedMoreString(prt, prt2, prt3):
    print("\033[91m{}\033[00m".format(prt), prt2, prt3)


def prYellow(prt):
    print("\033[93m{}\033[00m".format(prt))


def visualizeAspectRatioChart(dataSet, labels=["Abnormal spine", "Normal spine"]):
    # Visualization of the aspect ratio chart
    ax = dataSet[differentialColumn].value_counts().plot(
        kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of Normal spine and Abnormal spine")
    plt.legend(labels=labels, loc="best")
    plt.show()


def fun():
    # Proportion of Abnormal spine (0) and Normal spine (1):
    # [Number of Abnormal spine raws/Total number of Normal spine raws]
    prGreenMoreString('Abnormal spine: ', dataSet.Class_att.value_counts()[0],
                      '(% {:.2f})'.format(dataSet.Class_att.value_counts()[0] / dataSet.Class_att.count() * 100))
    prRedMoreString('Normal spine: ', dataSet.Class_att.value_counts()[1],
                    '(% {:.2f})'.format(dataSet.Class_att.value_counts()[1] / dataSet.Class_att.count() * 100))


if __name__ == '__main__':
    fileName = os.path.join(os.path.dirname(__file__), "Dataset_spine.csv")
    dataSet = pd.read_csv(fileName)
    differentialColumn = "Class_att"
    #String conversion
    dataSet[differentialColumn] = dataSet[differentialColumn].replace("Abnormal", 0)
    dataSet[differentialColumn] = dataSet[differentialColumn].replace("Normal", 1)
    #Removal of empty column
    dataSet = dataSet.loc[:, ~dataSet.columns.str.contains('^Unnamed')]
    #Renaming columns
    dataSet.rename(columns={
        'Col1': 'pelvic incidence',
        'Col2': 'pelvic tilt',
        'Col3': 'lumbar lordosis angle',
        'Col4': 'sacral slope',
        'Col5': 'pelvic radius',
        'Col6': 'degree spondylolisthesis',
        'Col7': 'pelvic slope',
        'Col8': 'Direct tilt',
        'Col9': 'thoracic slope',
        'Col10': 'cervical tilt',
        'Col11': 'sacrum angle',
        'Col12': 'scoliosis slope'
    },
          inplace=True, errors='raise')

    # Data overview
    print("\nDisplay (partial) of the dataframe:\n", dataSet.head())
    print("\nNumber of elements: ", len(dataSet.index) - 1)
    print("\nInfo dataset:\n", dataSet.describe())

    # Input dataset, eliminating the last column (needed for the output)
    X = dataSet.drop(differentialColumn, axis=1)
    Y = dataSet[differentialColumn]

    # BALANCING OF CLASSES
    visualizeAspectRatioChart(dataSet)

    fun()
    dataSet.drop(dataSet[(dataSet[differentialColumn] != 0) & (dataSet[differentialColumn] != 1)].index, inplace = True)
    df_minority = dataSet[dataSet[differentialColumn] == 1]
    df_majority = dataSet[dataSet[differentialColumn] == 0]

    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len (df_majority), random_state=42)
    df_spine = pd.concat([df_minority_upsampled, df_majority])
    
    prYellow("\nValue after Oversampling:")
    prGreenMoreString('Abnormal spine: ', dataSet.Class_att.value_counts()[0],
                      '(% {:.2f})'.format(dataSet.Class_att.value_counts()[0] / dataSet.Class_att.count() * 100))
    prRedMoreString('Normal spine: ', dataSet.Class_att.value_counts()[1],
                    '(% {:.2f})'.format(dataSet.Class_att.value_counts()[1] / dataSet.Class_att.count() * 100))

    print (df_spine)

    visualizeAspectRatioChart(df_spine)

    # Creation of X feature and target y
    X = df_spine.to_numpy()
    y = df_spine[differentialColumn].to_numpy()  # K-Fold Cross Validation

    kf = RepeatedKFold(n_splits=5, n_repeats=5)

    # Classifiers for the purpose of evaluation
    knn = KNeighborsClassifier()
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    svc = SVC()
    bnb = BernoulliNB()
    gnb = GaussianNB()

    # Score of metrics
    model = {
        'KNN': {'accuracy_list': 0.0,
                'precision_list': 0.0,
                'recall_list': 0.0,
                'f1_list': 0.0
                },

        'DecisionTree': {'accuracy_list': 0.0,
                         'precision_list': 0.0,
                         'recall_list': 0.0,
                         'f1_list': 0.0
                         },

        'RandomForest': {'accuracy_list': 0.0,
                         'precision_list': 0.0,
                         'recall_list': 0.0,
                         'f1_list': 0.0
                         },

        'SVM': {'accuracy_list': 0.0,
                'precision_list': 0.0,
                'recall_list': 0.0,
                'f1_list': 0.0
                },

        'BernoulliNB': {'accuracy_list': 0.0,
                        'precision_list': 0.0,
                        'recall_list': 0.0,
                        'f1_list': 0.0
                        },

        'GaussianNB': {'accuracy_list': 0.0,
                       'precision_list': 0.0,
                       'recall_list': 0.0,
                       'f1_list': 0.0
                       }

    }

    # K-Fold of the classifiers
    for train_index, test_index in kf.split(X, y):
        training_set, testing_set = X[train_index], X[test_index]

        # train data
        data_train = pd.DataFrame(training_set, columns=df_spine.columns)
        X_train = data_train.drop(differentialColumn, axis=1)
        y_train = data_train.Class_att

        # test data
        data_test = pd.DataFrame(testing_set, columns=df_spine.columns)
        X_test = data_test.drop(differentialColumn, axis=1)
        y_test = data_test.Class_att

        # classifier fit
        knn.fit(X_train, y_train)
        dtc.fit(X_train, y_train)
        rfc.fit(X_train, y_train)
        svc.fit(X_train, y_train)
        bnb.fit(X_train, y_train)
        gnb.fit(X_train, y_train)

        y_pred_knn = knn.predict(X_test)
        y_pred_dtc = dtc.predict(X_test)
        y_pred_rfc = rfc.predict(X_test)
        y_pred_SVM = svc.predict(X_test)
        y_pred_gnb = gnb.predict(X_test)
        y_pred_bnb = bnb.predict(X_test)

        # saving fold metrics in the dictionary
        model['KNN']['accuracy_list'] = (
            metrics.accuracy_score(y_test, y_pred_knn))
        model['KNN']['precision_list'] = (
            metrics.precision_score(y_test, y_pred_knn))
        model['KNN']['recall_list'] = (
            metrics.recall_score(y_test, y_pred_knn))
        model['KNN']['f1_list'] = (metrics.f1_score(y_test, y_pred_knn))

        model['DecisionTree']['accuracy_list'] = (
            metrics.accuracy_score(y_test, y_pred_dtc))
        model['DecisionTree']['precision_list'] = (
            metrics.precision_score(y_test, y_pred_dtc))
        model['DecisionTree']['recall_list'] = (
            metrics.recall_score(y_test, y_pred_dtc))
        model['DecisionTree']['f1_list'] = (
            metrics.f1_score(y_test, y_pred_knn))

        model['RandomForest']['accuracy_list'] = (
            metrics.accuracy_score(y_test, y_pred_rfc))
        model['RandomForest']['precision_list'] = (
            metrics.precision_score(y_test, y_pred_rfc))
        model['RandomForest']['recall_list'] = (
            metrics.recall_score(y_test, y_pred_rfc))
        model['RandomForest']['f1_list'] = (
            metrics.f1_score(y_test, y_pred_rfc))

        model['SVM']['accuracy_list'] = (
            metrics.accuracy_score(y_test, y_pred_SVM))
        model['SVM']['precision_list'] = (
            metrics.precision_score(y_test, y_pred_SVM))
        model['SVM']['recall_list'] = (
            metrics.recall_score(y_test, y_pred_SVM))
        model['SVM']['f1_list'] = (metrics.f1_score(y_test, y_pred_SVM))

        model['BernoulliNB']['accuracy_list'] = (
            metrics.accuracy_score(y_test, y_pred_bnb))
        model['BernoulliNB']['precision_list'] = (
            metrics.precision_score(y_test, y_pred_bnb))
        model['BernoulliNB']['recall_list'] = (
            metrics.recall_score(y_test, y_pred_bnb))
        model['BernoulliNB']['f1_list'] = (
            metrics.f1_score(y_test, y_pred_bnb))

        model['GaussianNB']['accuracy_list'] = (
            metrics.accuracy_score(y_test, y_pred_gnb))
        model['GaussianNB']['precision_list'] = (
            metrics.precision_score(y_test, y_pred_gnb))
        model['GaussianNB']['recall_list'] = (
            metrics.recall_score(y_test, y_pred_gnb))
        model['GaussianNB']['f1_list'] = (metrics.f1_score(y_test, y_pred_gnb))

        # report template

        def model_report(model):
            df_spine_models = []

            for clf in model:
                df_spine_models.append(pd.DataFrame({'model': [clf],
                                                     'accuracy': [np.mean(model[clf]['accuracy_list'])],
                                                     'precision': [np.mean(model[clf]['precision_list'])],
                                                     'recall': [np.mean(model[clf]['recall_list'])],
                                                     'f1score': [np.mean(model[clf]['f1_list'])]
                                                     }))

            return df_spine_models

    print ("----------------------------------------")

    # Visualization of the table with metrics and Graph
    df_spine_models_concat = pd.concat(model_report(model), axis=0).reset_index()  # concatenation of the models
    df_spine_models_concat = df_spine_models_concat.drop('index', axis=1)  # removal of the index
    print("\n", df_spine_models_concat)  # table display

    # Accuracy Graph
    x = df_spine_models_concat.model
    y = df_spine_models_concat.accuracy

    plt.bar(x, y)
    plt.title("Accuracy")
    plt.show()

    # Precision Graph
    x = df_spine_models_concat.model
    y = df_spine_models_concat.precision

    plt.bar(x, y)
    plt.title("Precision")
    plt.show()

    # Recall Graph
    x = df_spine_models_concat.model
    y = df_spine_models_concat.recall

    plt.bar(x, y)
    plt.title("Recall")
    plt.show()

    # F1score Graph
    x = df_spine_models_concat.model
    y = df_spine_models_concat.f1score

    plt.bar(x, y)
    plt.title("F1score")
    plt.show()

    # Standard deviation
    std_knn = np.std(cross_val_score(knn, X_test, y_test, cv=5, n_jobs=5))
    std_dtc = np.std(cross_val_score(dtc, X_test, y_test, cv=5, n_jobs=5))
    std_rfc = np.std(cross_val_score(rfc, X_test, y_test, cv=5, n_jobs=5))
    std_svc = np.std(cross_val_score(svc, X_test, y_test, cv=5, n_jobs=5))
    std_bnd = np.std(cross_val_score(bnb, X_test, y_test, cv=5, n_jobs=5))
    std_gnb = np.std(cross_val_score(gnb, X_test, y_test, cv=5, n_jobs=5))
    plt.plot(["KNN", "DecisionTree", "RandomForest", "SVM", "BernoulliNB", "GaussianNB"],
             [std_knn, std_dtc, std_rfc, std_svc, std_bnd, std_gnb])
    plt.title("Standard deviation")
    plt.ylabel("Standard deviation value")
    plt.xlabel("Classifiers")
    plt.show()
    print("\nStandard deviation for Knn:", std_knn)
    print("\nStandard deviation for DecisionTree:", std_dtc)
    print("\nStandard deviation for RandomForest:", std_rfc)
    print("\nStandard deviation for SVM:", std_svc)
    print("\nStandard deviation for BernoulliNB:", std_bnd)
    print("\nStandard deviation for GaussianNB:", std_gnb)

    # VERIFICATION OF THE IMPORTANCE OF FEATURES

    # Creation of X feature and target y
    X = df_spine.drop(differentialColumn, axis=1)
    y = df_spine[differentialColumn]

    # Classifier to be used for the search of the main features
    rfc = RandomForestClassifier(random_state=42, n_estimators=100)
    rfc_model = rfc.fit(X, y)

    # Tracking features based on their importance
    ax = (pd.Series(rfc_model.feature_importances_, index=X.columns)
          .nlargest(10)  # maximum number of features to display
          .plot(kind='pie', figsize=(6, 6), autopct=autopct)  # type of chart and size
          .invert_yaxis())  # to ensure a descending order

    # Visualization of the graph of the most important features
    plt.title("Top features derived by Random Forest")
    plt.ylabel("")
    plt.show()

    """
    documentazione
    vai a ricevimento e chiedi quali classificatori hanno più senso e cosa fare dopo

    """
    