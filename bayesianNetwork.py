import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

from stampe import prGreen, prRed, prYellow
from userInterface import querySystem, outputExplain


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
    # reading of the already generated model from file
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
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
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
    plt.show()
    plt.clf()


def testQueries(data, differentialColumn):
    outputExplain ()

    # Potential notLikedSong
    notLikedSong = data.query(  # 0
        show_progress=False,
        variables=[differentialColumn],
        evidence={
            'trackIsexplicit': 0,
            'danceability': 91,
            'energy': 83,
            'key': 4,
            'loudness': 29,
            'speechiness': 99,
            'acousticness': 99,
            'instrumentalness': 100,
            'valence': 88,
            'tempo': 92
        },
    )
    prRed("\nProbability for a potential not liked song:")
    print(notLikedSong)

    # Potential likedSong
    likedSong = data.query(  # 1
        show_progress=False,
        variables=[differentialColumn],
        evidence={
            'trackIsexplicit': 1, 
            'danceability': 83, 
            'energy': 88, 
            'key': 98, 
            'loudness': 13,
            'speechiness': 99, 
            'acousticness': 98, 
            'instrumentalness': 99, 
            'valence': 90, 
            'tempo': 96
        },
    )
    prGreen("\nProbability for a potentially liked song:")
    print(likedSong, "\n")


def bayesianNetwork(dataSet, differentialColumn):
    model = modelCreation(dataSet, differentialColumn)

    bNet = bNetCreation(model, dataSet)

    showGraphOfNodes(model, bNet)

    prYellow('\nMarkov blanket for "songIsLiked"')
    print(bNet.get_markov_blanket(differentialColumn), "\n")

    data = VariableElimination(bNet)

    testQueries(data, differentialColumn)

    querySystem(data, differentialColumn)
