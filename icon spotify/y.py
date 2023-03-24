import itertools
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

from datasetCleaning import cleanAndRenameColumnsOfDataset

setFeatures = ["pelvic incidence","pelvic tilt","lumbar lordosis angle","sacral slope","pelvic radius","degree spondylolisthesis","pelvic slope","direct tilt","thoracic slope","cervical tilt","sacrum angle","scoliosis slope","spine_state"]

combinations = list(itertools.combinations(setFeatures, 2))

fileName = os.path.join(os.path.dirname(__file__), "Dataset_spine.csv")
dataSet = pd.read_csv(fileName)
dataSet = cleanAndRenameColumnsOfDataset (dataSet, differentialColumn)


# Visualize the clusters
for feature1, feature2 in combinations:
    plt.scatter(dataSet[feature1], dataSet[feature2])
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    #plt.savefig(os.path.join((os.path.abspath ("")), "pic", "scatter plot", "scatter plot" + " " + feature1 + "-" + feature2 + " " +str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".jpeg"))