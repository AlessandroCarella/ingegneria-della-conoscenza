import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


def createXfeatureAndyTarget(dataSet, differentialColumn):
    X = dataSet.drop(differentialColumn, axis=1)
    y = dataSet[differentialColumn]

    rfc = RandomForestClassifier(random_state=42, n_estimators=100)
    rfc_model = rfc.fit(X, y)

    return rfc_model, X


def visualizeFeaturesImportances(rfc_model, X):
    # Tracking features based on their importance
    ax = (pd.Series(rfc_model.feature_importances_, index=X.columns)
          .nlargest(10)  # maximum number of features to display
          .plot(kind='pie', figsize=(6, 6))  # type of chart and size
          .invert_yaxis())  # to ensure a descending order

    # Visualization of the graph of the most important features
    plt.title("Top features derived by Random Forest")
    plt.ylabel("")
    #plt.clf()
    plt.show()
