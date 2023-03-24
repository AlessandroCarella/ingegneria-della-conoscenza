import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def createModel():
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

    return model


def saveFoldMetricsInModel(model, y_test, y_pred_knn, y_pred_dtc, y_pred_rfc, y_pred_SVM, y_pred_bnb, y_pred_gnb):
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

    return model


def trainModelKFold(dataSet, differentialColumn, model=createModel()):
    X = dataSet.to_numpy()
    y = dataSet[differentialColumn].to_numpy()  # K-Fold Cross Validation

    knn = KNeighborsClassifier()
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    svc = SVC()
    bnb = BernoulliNB()
    gnb = GaussianNB()

    kf = RepeatedKFold(n_splits=5, n_repeats=5)

    for train_index, test_index in kf.split(X, y):
        training_set, testing_set = X[train_index], X[test_index]

        data_train = pd.DataFrame(training_set, columns=dataSet.columns)
        X_train = data_train.drop(differentialColumn, axis=1)
        y_train = data_train.songIsLiked

        data_test = pd.DataFrame(testing_set, columns=dataSet.columns)
        X_test = data_test.drop(differentialColumn, axis=1)
        y_test = data_test.songIsLiked

        knn.fit(X_train, y_train)
        dtc.fit(X_train, y_train)
        rfc.fit(X_train, y_train)
        svc.fit(X_train, y_train)
        bnb.fit(X_train, y_train)
        gnb.fit(X_train, y_train)

        model = saveFoldMetricsInModel(model,
                                       y_test,
                                       knn.predict(X_test),
                                       dtc.predict(X_test),
                                       rfc.predict(X_test),
                                       svc.predict(X_test),
                                       bnb.predict(X_test),
                                       gnb.predict(X_test))

    return model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb


def model_report(model):
    dataSet_models = []

    for clf in model:
        dataSet_models.append(pd.DataFrame({'model': [clf],
                                            'accuracy': [np.mean(model[clf]['accuracy_list'])],
                                            'precision': [np.mean(model[clf]['precision_list'])],
                                            'recall': [np.mean(model[clf]['recall_list'])],
                                            'f1score': [np.mean(model[clf]['f1_list'])]
                                            }))

    return dataSet_models


def findStandardDeviation(X_test, y_test, knn, dtc, rfc, svc, bnb, gnb):
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
    plt.clf()
    plt.show()
    print("\nStandard deviation for Knn:", std_knn)
    print("\nStandard deviation for DecisionTree:", std_dtc)
    print("\nStandard deviation for RandomForest:", std_rfc)
    print("\nStandard deviation for SVM:", std_svc)
    print("\nStandard deviation for BernoulliNB:", std_bnd)
    print("\nStandard deviation for GaussianNB:", std_gnb)


def visualizeMetricsGraphs(model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb):
    dataSet_models_concat = pd.concat(model_report(
        model), axis=0).reset_index()
    dataSet_models_concat = dataSet_models_concat.drop(
        'index', axis=1)
    print("\n", dataSet_models_concat)

    # Accuracy Graph
    x = dataSet_models_concat.model
    y = dataSet_models_concat.accuracy

    plt.bar(x, y)
    plt.title("Accuracy")
    plt.clf()
    plt.show()

    # Precision Graph
    x = dataSet_models_concat.model
    y = dataSet_models_concat.precision

    plt.bar(x, y)
    plt.title("Precision")
    plt.clf()
    plt.show()

    # Recall Graph
    x = dataSet_models_concat.model
    y = dataSet_models_concat.recall

    plt.bar(x, y)
    plt.title("Recall")
    plt.clf()
    plt.show()

    # F1score Graph
    x = dataSet_models_concat.model
    y = dataSet_models_concat.f1score

    plt.bar(x, y)
    plt.title("F1score")
    plt.clf()
    plt.show()

    findStandardDeviation(X_test, y_test, knn, dtc, rfc, svc, bnb, gnb)
