from stampe import *
from datasetCleaning import *
from balancingOfClasses import *
from outliersRemoval import *
from training import *
from verificationFeaturesImportance import *
from userInterface import *

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator

# alternativa a HillClimbSearch
from bnlearn import structure_learning

#
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork, BayesianModel

from sklearn.feature_selection import SelectKBest, f_regression


def printBestFeatures(dataSet, differentialColumn):
    dataSetTemp = dataSet.copy()
    X = dataSetTemp.drop(differentialColumn, axis=1)
    y = dataSetTemp[differentialColumn]

    selector = SelectKBest(score_func=f_regression, k=5)
    X_new = selector.fit_transform(X, y)

    idxs_selected = selector.get_support(indices=True)
    selected_columns = X.columns[idxs_selected]

    print (selected_columns.tolist())

def selectBestFeatures(dataSet, differentialColumn):
    dataSetTemp = dataSet.copy()
    """
    #generation of best features

    X = dataSetTemp.drop(differentialColumn, axis=1)
    y = dataSetTemp[differentialColumn]

    selector = SelectKBest(score_func=f_regression, k=5)
    X_new = selector.fit_transform(X, y)

    idxs_selected = selector.get_support(indices=True)
    selected_columns = X.columns[idxs_selected]

    selected_columns = selected_columns.tolist() + [differentialColumn]
    """

    previouslyGeneratedBestFeatures =['energy', 'loudness', 'tempo', 'speechiness', 'instrumentalness', 'songIsLiked']
    selected_columns = previouslyGeneratedBestFeatures

    dataSet = dataSetTemp[selected_columns]

    return dataSet


def modelCreation(dataSet, differentialColumn):
    """
    #generation of the model
    hc_k2 = HillClimbSearch(dataSet)

    k2_model = hc_k2.estimate(
        max_iter=22
    )  

    import pickle

    with open("k2_model.pkl", "wb") as file:
        pickle.dump(k2_model, file)
    
    """
    #reading of the model from file
    k2_model = 0
    import pickle
    with open('k2_model.pkl', 'rb') as file:
        k2_model = pickle.load(file)
    
    
    return k2_model


def bNetCreation(k2_model, dataSet):
    bNet = BayesianNetwork(k2_model.edges())

    bNet.fit(dataSet)

    return bNet


def showGraphOfNodes(k2_model, bNet):
    G = nx.MultiDiGraph(k2_model.edges())
    G.add_edges_from(bNet.edges())
    pos = nx.spring_layout(G, iterations=100, k=2, threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=7,
        arrowstyle="->",
        edge_color="purple",
        connectionstyle="angle3,angleA=90,angleB=0",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.clf()
    #plt.show()
    """
    import os
    import datetime

    plt.savefig(
        os.path.join(
            os.path.abspath("pic/"),
            "bayesian_network "
            + str(datetime.datetime.now().strftime("%d-%m-%y %H-%M-%S")),
        )
    )
    """


def testQueries(data, differentialColumn):
    prYellow("Probability results for songIsLiked are so structured:")
    print(
        "+-------------------------------+---------------------------+\n",
        "|         feature name          |    feature probability    |\n",
        "+===============================+===========================+\n",
        "|  feature name(not liked song) | probability feature (val) |\n",
        "+-------------------------------+---------------------------+\n",
        "|   feature name(liked song)    | probability feature (val) |\n",
        "+-------------------------------+---------------------------+",
    )
    prPurple(
        "Probability value fluctuates between 0 (impossible event) to 1 (certain event)\n"
    )

    # Potential notLikedSong
    notLikedSong = data.query( #0
        show_progress=False,
        variables=[differentialColumn],
        evidence={
            #"trackIsexplicit": 0,
            "danceability": 93,
            "energy": 99,
            #"key": 64,
            "loudness": 63,
            "speechiness": 99,
            "acousticness": 78,
            #"instrumentalness": 84,
            #"liveness": 99,
            "valence": 99,
            #"tempo": 91,
        },
    )
    prRed("\nProbability for a potential not liked song:")
    print(notLikedSong)
    

    # Potential likedSong
    likedSong = data.query( #1
        show_progress=False,
        variables=[differentialColumn],
        evidence={
            #"trackIsexplicit": 0,
            "danceability": 83,
            "energy": 97,
            #"key": 42,
            "loudness": 51,
            "speechiness": 99,
            "acousticness": 80,
            #"instrumentalness": 91,
            #"liveness": 99,
            "valence": 96,
            #"tempo": 0,
        },
    )
    prGreen("\nProbability for a potentially liked song:")
    print(likedSong, "\n")


def saveData (dataSet):
    selected_cols = ['energy', 'speechiness', 'instrumentalness', "songIsLiked"]
    data_selected = dataSet[selected_cols]

    # save the selected columns to a new CSV file
    data_selected.to_csv('selected_data_file.csv', index=False)

def bayesianNetwork(dataSet, differentialColumn):
    #dataSet = selectBestFeatures(dataSet, differentialColumn)

    model = modelCreation(dataSet, differentialColumn)

    bNet = bNetCreation(model, dataSet)

    showGraphOfNodes(model, bNet)

    prYellow('\nMarkov blanket for "songIsLiked"')
    print(bNet.get_markov_blanket(differentialColumn), "\n")

    data = VariableElimination(bNet)

    #saveData (dataSet)

    print ("best features:")
    printBestFeatures(dataSet, differentialColumn)

    testQueries(data, differentialColumn)

    querySystem(data, differentialColumn)
