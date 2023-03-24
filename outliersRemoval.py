import numpy as np
from sklearn.mixture import GaussianMixture


def softClusteringEMOutliersRemoval(dataSet):
    gmm = GaussianMixture(n_components=3)

    gmm.fit(dataSet)

    likelihoods = gmm.score_samples(dataSet)

    outlier_threshold = np.mean(likelihoods) - 3 * np.std(likelihoods)

    outliers = np.where(likelihoods < outlier_threshold)[0]
    clean_data = dataSet.drop(outliers)

    print(
        f'Removed {len(outliers)} outliers from dataset. Cleaned dataset size: {clean_data.shape}')

    dataSet.to_csv('cleanedDataSetSoftClusteringEM.csv', index=False)

    return clean_data
