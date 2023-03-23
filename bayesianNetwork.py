from stampe import *
from datasetCleaning import *
from balancingOfClasses import *
from outliersRemoval import *
from training import *
from verificationFeaturesImportance import *

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork, BayesianModel


def modelCreation(dataSet, differentialColumn):
    """
    dataSet_int = np.array(dataSet, dtype=int)
    dataSet = pd.DataFrame(dataSet_int, columns=dataSet.columns)

    X_train = dataSet
    y_train = dataSet[differentialColumn]

    k2 = K2Score(X_train)
    hc_k2 = HillClimbSearch(X_train)

    k2_model = hc_k2.estimate(scoring_method=k2)

    """  
    k2_model = 0
    import pickle

    # Open the file in binary mode and read the object
    with open('k2_model.pkl', 'rb') as file:
        k2_model = pickle.load(file)
    #"""
    return k2_model

def bNetCreation(k2_model, dataSet):
    bNet = BayesianNetwork(k2_model.edges())
    
    bNet.fit(dataSet, estimator=MaximumLikelihoodEstimator)
    
    return bNet


def showGraphOfNodes(k2_model, bNet):
    G = nx.MultiDiGraph(k2_model.edges())
    G.add_edges_from(bNet.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", clip_on=True, horizontalalignment="center",
                            verticalalignment="bottom")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=7, arrowstyle="->", edge_color="purple",
                           connectionstyle="angle3,angleA=90,angleB=0", min_source_margin=1.2, min_target_margin=1.5,
                           edge_vmin=2, edge_vmax=2)

    plt.title("BAYESIAN NETWORK GRAPH")
    #plt.show()


def testQueries(data, differentialColumn):
    prYellow("Probability results for Class_att are so structured:")
    print(" +--------------------------------+---------------------------+\n",
          "|         feature name          |    feature probability    |\n",
          "+===============================+===========================+\n",
          "|  feature name(abnormal spine) | probability feature (val) |\n",
          "+-------------------------------+---------------------------+\n",
          "|  feature name(normal spine)   | probability feature (val) |\n",
          "+-------------------------------+---------------------------+")
    prPurple(
        "Probability value fluctuates between 0 (impossible event) to 1 (certain event)\n")

    # Potential normal spine subject
    normalSpine = data.query(
        show_progress=False, 
        variables=[differentialColumn],
        evidence={
            #'pelvic radius': 115,
            'sacrum angle': 4,
            'thoracic slope': 17,
            'degree spondylolisthesis': 2,
            'lumbar lordosis angle': 39,
        }
    )

    prGreen('\nProbability for a potentially normal spines:')
    print(normalSpine, '\n')

    # Potential abnormal spine subject
    abnormalSpine = data.query(
        show_progress=False, 
        variables=[differentialColumn],
        evidence={
            'pelvic radius': 113,
            'sacrum angle': -26,
            'thoracic slope': 8,
            'degree spondylolisthesis': 4,
            'lumbar lordosis angle': 24,
        }
    )

    prRed('\nProbability for a potential abnormal spine:')
    print(abnormalSpine)


def querySystem(dataSet, differentialColumn, data, bNet):
    prYellow("\n\n\t\t\t\t\tWelcome to our system!\n\n\t"
             "It allows you to predict whether, taken a subjects, they have a normal spine or not.\n\n")

    while True:
        i = 0
        try:
            prYellow(
                "Do you want to enter your data for a prediction? - Y/N? - (Typing 'n' close program)")
            result = str(input())
            if 'N' == result or result == 'n':
                exit(1)
            elif 'Y' == result or result == 'y':
                prYellow("Please insert: ")
                columns = ['lumbar lordosis angle', 'pelvic radius',
                           'sacrum angle', 'degree spondylolisthesis']
                print(columns)
                prRed("All values are obligatory to enter!")

                values = [None] * len(columns)
                while i < len(columns):
                    print("The minimum acceptable \"", columns[i], "\" value is:", dataSet[columns[i]].min(),
                          "The maximum is:", dataSet[columns[i]].max())
                    print("Insert ", columns[i], " value: ")

                    values[i] = int(input())
                    if values[i] < dataSet[columns[i]].min():
                        prRed("Error! You entered too small value!")
                    elif values[i] > dataSet[columns[i]].max():
                        prRed("Error! You entered too large value!")
                    else:
                        i += 1

                try:
                    i = 0
                    dataAvailable = {}
                    while i < len(columns):
                        dataAvailable[columns[i]] = values[i]
                        i = i + 1
                    UserInput = data.query(show_progress=False, variables=[differentialColumn],
                                           evidence=dataAvailable)
                    print(UserInput)

                except IndexError as e:
                    prRed("Error!")
                    print("You have insert:  ", values)
                    print(e.args)
            else:
                print("Wrong input. Write Y or N")
        except ValueError:
            print("Wrong input")


def bayesianNetwork(dataSet, differentialColumn):
    np.random.seed(42)  # set seed to always get the same byesian network

    model = modelCreation(dataSet, differentialColumn)

    bNet = bNetCreation(model, dataSet)

    showGraphOfNodes(model, bNet)

    prYellow("\nMarkov blanket for \"Class_att\"")
    print(bNet.get_markov_blanket(differentialColumn), "\n")

    data = VariableElimination(bNet) 

    testQueries(data, differentialColumn)

    querySystem(dataSet, differentialColumn, data, bNet)
