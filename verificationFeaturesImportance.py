import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def createXfeatureAndyTarget(dataSet, differentialColumn):
    X = dataSet.drop(differentialColumn, axis=1)
    y = dataSet[differentialColumn]

    rfc = RandomForestClassifier(random_state=42, n_estimators=100)
    rfc_model = rfc.fit(X, y)

    return rfc_model, X


def visualizeFeaturesImportances(rfc_model, X):
    ax = (pd.Series(rfc_model.feature_importances_, index=X.columns)
          .nlargest(10)
          .plot(kind='pie', figsize=(6, 6))
          .invert_yaxis())

    plt.title("Top features derived by Random Forest")
    plt.ylabel("")
    plt.clf()
    plt.show()
