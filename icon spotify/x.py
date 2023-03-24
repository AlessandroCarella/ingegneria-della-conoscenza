import os
from datasetCleaning import cleanAndRenameColumnsOfDataset
import pandas as pd

#DATASET CLEANING
fileName = os.path.join(os.path.dirname(__file__), "Dataset_spine.csv")
dataSet = pd.read_csv(fileName)
differentialColumn = "songIsLiked"
dataSet = cleanAndRenameColumnsOfDataset (dataSet, differentialColumn)

# define the list of columns to keep
keep_cols = ['thoracic slope', 'degree spondylolisthesis', 'lumbar lordosis angle', 'pelvic radius', 'sacrum angle', "songIsLiked"]

# drop any columns that are not in the list
dataSet = dataSet[keep_cols]

dataSet['songIsLiked'] = dataSet['songIsLiked'].replace(0, "not_liked_song")
dataSet['songIsLiked'] = dataSet['songIsLiked'].replace(1, "liked_song")

dataSet.to_csv("soloFeatureQuery.csv", index=False)
