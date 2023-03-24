import os
import pandas as pd

fileName = os.path.join(os.path.dirname(__file__), "dataSet.csv")
dataSet = pd.read_csv(fileName)
differentialColumn = "songIsLiked"

counts = dataSet['songIsLiked'].value_counts()

print (counts)