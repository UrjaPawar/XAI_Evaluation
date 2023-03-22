import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import math
from sklearn.linear_model import LogisticRegression
from scipy import stats

def plot_side_by_side(dataframe):
    '''
    :param dataframe: with columns as index of x axis and values in y axis
    :return: a plot figure that can be saved or shown
    '''

def plot_stacked(dataframe):
    '''

    :param dataframe: with columns as index of x axis and values in y axis
    :return: a plot figure that can be saved or shown
    '''

def plot_histograms(dataframe, path, ind):
    '''

    :param dataframe:
    :return: a plot figure that can be saved or shown
    '''

    fig, axs = plt.subplots(2, int(len(dataframe.columns) / 2) + 1, figsize=(30,18))
    fig.suptitle('Histogtams')
    _ind = 0
    for j in range(2):
        for i in range(int(len(dataframe.columns) / 2) + 1):
            if _ind < len(dataframe.columns):
                axs[j][i].hist(dataframe[dataframe.columns[_ind]])
                _ind += 1
                axs[j][i].tick_params(axis='x',labelsize=20)
                axs[j][i].tick_params(axis='y', labelsize=20)
                axs[j][i].set_title(dataframe.columns[_ind-1])
    plt.savefig(path+"/histogram"+str(ind)+".png")




def plot_scatter(dataframe):
    '''

    :param dataframe: with columns as index of x axis and values in y axis
    :return: a plot figure that can be saved or shown
    '''

def pie_plot_dashboard(dataframe):
    '''

    :param dataframe: with columns as index of x axis and values in y axis
    :return: a plot figure that can be saved or shown
    '''