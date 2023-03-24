from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

import numpy as np
from sklearn.mixture import GaussianMixture


def softClusteringEMOutliersRemoval(dataSet):
    gmm = GaussianMixture(n_components=1)

    gmm.fit(dataSet)

    likelihoods = gmm.score_samples(dataSet)

    outlier_threshold = np.mean(likelihoods) - 3 * np.std(likelihoods)

    outliers = np.where(likelihoods < outlier_threshold)[0]
    clean_data = dataSet.drop(outliers)

    clean_data.to_csv('cleaned_data_soft_clustering.csv', index=False)

    print(
        f'Removed {len(outliers)} outliers from dataset. Cleaned dataset size: {clean_data.shape}')

    return clean_data


def DBSCANOutliersRemoval(dataSet):
    X = dataSet.iloc[:, :-1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.5, min_samples=5)

    dbscan.fit(X)

    labels = dbscan.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    print('Number of clusters:', n_clusters)
    print('Number of outliers:', n_outliers)

    clean_dataSet_DBSCAN = dataSet[labels != -1]

    clean_dataSet_DBSCAN.to_csv('clean_dataset_DBSCAN.csv', index=False)

    return clean_dataSet_DBSCAN


def ZscoreOutliersRemoval(dataSet):
    z_scores = (dataSet - dataSet.mean()) / dataSet.std()

    threshold = 1.8
    df_clean = dataSet[(z_scores < threshold).all(axis=1)]
    df_clean.to_csv('cleaned_data_z_score.csv', index=False)

    return df_clean


def isolationForestOutliersRemoval(dataSet):
    isolation_forest = IsolationForest()

    isolation_forest.fit(dataSet.iloc[:, :-1])

    outliers = isolation_forest.predict(dataSet.iloc[:, :-1])

    dataSet = dataSet[outliers != -1]

    dataSet.to_csv('cleaned_data_isolation_forest.csv', index=False)

    return dataSet
