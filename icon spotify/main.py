from installPackages import installPackages

#installPackages ()

# dataset: https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset
from bayesianNetwork import bayesianNetwork
import pandas as pd

import os

from stampe import histogramAndDensityPlot
from datasetCleaning import cleanAndRenameColumnsOfDataset, dataOverview
from balancingOfClasses import visualizeAspectRatioChart, visualizeNumberOfSamplesForClasses, resampleDataset, undersampleDataset
from outliersRemoval import softClusteringEMOutliersRemoval
from training import trainModelKFold, visualizeMetricsGraphs
from verificationFeaturesImportance import createXfeatureAndyTarget, visualizeFeaturesImportances

# DATASET CLEANING
fileName = os.path.join(os.path.dirname(__file__), "dataSet.csv")
dataSet = pd.read_csv(fileName).astype(int)
differentialColumn = "songIsLiked"
#dataSet = cleanAndRenameColumnsOfDataset(dataSet, differentialColumn)
"""
print(dataSet)

# LEARNING NON SUPERVISIONATO (CLUSTERING)
print("histogram and density plot before outliers removal with clustering")
histogramAndDensityPlot(dataSet, differentialColumn)

"""
dataSet = softClusteringEMOutliersRemoval(dataSet)
"""
print("histogram and density plot after outliers removal with clustering")
histogramAndDensityPlot(dataSet, differentialColumn)

dataOverview(dataSet)

# BALANCING OF CLASSES
visualizeAspectRatioChart(dataSet, differentialColumn)

visualizeNumberOfSamplesForClasses(dataSet)
"""

dataSet = resampleDataset(dataSet, differentialColumn)
#dataSet = undersampleDataset(dataSet, differentialColumn)

"""
print(dataSet)

visualizeAspectRatioChart(dataSet, differentialColumn)


# TRAINING

model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb = trainModelKFold(
    dataSet, differentialColumn)
"""
"""
for i in range (0, 2):
    model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb = trainModelKFold (dataSet, differentialColumn, model)

"""
"""
visualizeMetricsGraphs(model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb)
"""

"""
# VERIFICATION OF THE IMPORTANCE OF FEATURES
rfc_model, X = createXfeatureAndyTarget(dataSet, differentialColumn)

visualizeFeaturesImportances(rfc_model, X)
"""

# BAYESIAN NETWORK
bayesianNetwork(dataSet, differentialColumn)