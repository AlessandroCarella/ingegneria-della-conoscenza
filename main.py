from installPackages import installPackages
#installPackages ()

import os

import pandas as pd

from balancingOfClasses import resampleDataset, visualizeAspectRatioChart
from bayesianNetwork import bayesianNetwork
from outliersRemoval import softClusteringEMOutliersRemoval
from training import trainModelKFold, visualizeMetricsGraphs
from verificationFeaturesImportance import (createXfeatureAndyTarget,
                                            visualizeFeaturesImportances)


# DATASET CLEANING
fileName = os.path.join(os.path.dirname(__file__), "dataSet.csv")
dataSet = pd.read_csv(fileName).astype(int)
differentialColumn = "songIsLiked"

# LEARNING NON SUPERVISIONATO (CLUSTERING)
dataSet = softClusteringEMOutliersRemoval(dataSet)


# BALANCING OF CLASSES
visualizeAspectRatioChart(dataSet, differentialColumn)
# visualizeNumberOfSamplesForClasses(dataSet)

dataSet = resampleDataset(dataSet, differentialColumn)
# dataSet = undersampleDataset(dataSet, differentialColumn)

visualizeAspectRatioChart(dataSet, differentialColumn)


# TRAINING
model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb = trainModelKFold(
    dataSet, differentialColumn)
visualizeMetricsGraphs(model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb)


# VERIFICATION OF THE IMPORTANCE OF FEATURES
rfc_model, X = createXfeatureAndyTarget(dataSet, differentialColumn)

visualizeFeaturesImportances(rfc_model, X)


# BAYESIAN NETWORK
bayesianNetwork(dataSet, differentialColumn)
