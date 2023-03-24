from installPackages import installPackages

#installPackages ()

# dataset: https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset
from bayesianNetwork import bayesianNetwork
import pandas as pd
import matplotlib.pyplot as plt

import os

from stampe import histogramAndDensityPlot
from datasetCleaning import cleanAndRenameColumnsOfDataset, dataOverview
from balancingOfClasses import visualizeAspectRatioChart, visualizeNumberOfSamplesForClasses, resampleDataset
from outliersRemoval import softClusteringEMOutliersRemoval
from training import trainModelKFold, visualizeMetricsGraphs
from verificationFeaturesImportance import createXfeatureAndyTarget, visualizeFeaturesImportances

plt.gcf().set_size_inches(19.2, 10.8)

# DATASET CLEANING
fileName = os.path.join(os.path.dirname(__file__), "Dataset_spine.csv")
dataSet = pd.read_csv(fileName)
differentialColumn = "spine_state"
dataSet = cleanAndRenameColumnsOfDataset(dataSet, differentialColumn)

print(dataSet)

# LEARNING NON SUPERVISIONATO (CLUSTERING)
print("histogram and density plot before outliers removal with clustering")
histogramAndDensityPlot(dataSet, differentialColumn)


dataSet = softClusteringEMOutliersRemoval(dataSet)

dataSet.to_csv("cleaned_data_soft_clustering.csv", index=False)


print("histogram and density plot after outliers removal with clustering")
histogramAndDensityPlot(dataSet, differentialColumn)

dataOverview(dataSet)

# BALANCING OF CLASSES
visualizeAspectRatioChart(dataSet, differentialColumn)

visualizeNumberOfSamplesForClasses(dataSet)


dataSet = resampleDataset(dataSet, differentialColumn)

print(dataSet)

visualizeAspectRatioChart(dataSet, differentialColumn)


# TRAINING

model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb = trainModelKFold(
    dataSet, differentialColumn)
"""
for i in range (0, 2):
    model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb = trainModelKFold (dataSet, differentialColumn, model)

"""

visualizeMetricsGraphs(model, X_test, y_test, knn, dtc, rfc, svc, bnb, gnb)


# VERIFICATION OF THE IMPORTANCE OF FEATURES
rfc_model, X = createXfeatureAndyTarget(dataSet, differentialColumn)

visualizeFeaturesImportances(rfc_model, X)


# BAYESIAN NETWORK
bayesianNetwork(dataSet, differentialColumn)