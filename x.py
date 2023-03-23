import os
from datasetCleaning import cleanAndRenameColumnsOfDataset
import pandas as pd

#DATASET CLEANING
fileName = os.path.join(os.path.dirname(__file__), "Dataset_spine.csv")
dataSet = pd.read_csv(fileName)
differentialColumn = "Class_att"
dataSet = cleanAndRenameColumnsOfDataset (dataSet, differentialColumn)

# define the list of columns to keep
keep_cols = ['thoracic slope', 'degree spondylolisthesis', 'lumbar lordosis angle', 'pelvic radius', 'sacrum angle', "Class_att"]

# drop any columns that are not in the list
dataSet = dataSet[keep_cols]

dataSet['Class_att'] = dataSet['Class_att'].replace(0, "Abnormal")
dataSet['Class_att'] = dataSet['Class_att'].replace(1, "Normal")

dataSet.to_csv("soloFeatureQuery.csv", index=False)
