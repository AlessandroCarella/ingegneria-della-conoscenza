import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os
import sys
import datetime
import itertools


def histogramAndDensityPlot(dataSet, differentialColumn):
    # Drop "Class_att" feature
    #dataSet = dataSet.drop(differentialColumn, axis=1)

    # Histogram
    sns.histplot(data=dataSet)
    #plt.savefig (os.path.join((os.path.abspath ("")), "pic", "histogram " + str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".jpeg"))

    # Density plot
    sns.kdeplot(data=dataSet)
    #plt.savefig (os.path.join((os.path.abspath ("")), "pic", "density plot " + str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".jpeg"))

    setFeatures = ["pelvic incidence", "pelvic tilt", "lumbar lordosis angle", "sacral slope", "pelvic radius", "degree spondylolisthesis",
                   "pelvic slope", "direct tilt", "thoracic slope", "cervical tilt", "sacrum angle", "scoliosis slope", "Class_att"]

    combinations = list(itertools.combinations(setFeatures, 2))

    # Visualize the clusters
    for feature1, feature2 in combinations:
        plt.scatter(dataSet[feature1], dataSet[feature2])
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        #plt.savefig(os.path.join((os.path.abspath ("")), "pic", "scatter plot", "scatter plot" + " " + feature1 + "-" + feature2 + " " +str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".jpeg"))


def prPurple(prt):
    print("\033[95m{}\033[00m".format(prt))


def prRedMoreString(prt, prt2, prt3):
    print("\033[91m{}\033[00m".format(prt), prt2, prt3)


def prGreenMoreString(prt, prt2, prt3):
    print("\n\033[92m{}\033[00m".format(prt), prt2, prt3)


def prRed(prt):
    print("\033[91m{}\033[00m".format(prt))


def prGreen(prt):
    print("\033[92m{}\033[00m".format(prt))


def prYellow(prt):
    print("\033[93m{}\033[00m".format(prt))


def autopct(pct):
    # shows only values of labers that are greater than 1%
    return ('%.2f' % pct + "%") if pct > 1 else ''
