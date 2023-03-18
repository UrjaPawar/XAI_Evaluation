from med_dataset import Data
from sklearn.neighbors import NearestNeighbors,KernelDensity
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

class Density:
    def __init__(self, data: Data):
        self.data = data
        self.data_kde = None
        self.clusterer = self.k_means_clusterer(4)
        self.cluster_precedence = self.data.cluster_order


    def get_density_score(self, data_samples):
        self.data_kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(self.data.train_df[self.data.features])
        return self.data_kde.score_samples(data_samples)

    def get_clusters(self, ind:int, sample:pd.Series):
        clusters = []
        for feat in self.cluster_precedence[ind]:
            if feat in self.data.continuous:
                mini = min(self.data.df[feat])
                maxi = max(self.data.df[feat])
                divides = int((maxi - mini) / 3)
                if len(clusters) != 0:
                    originals = clusters.copy()
                    clusters = []
                    for cluster in originals:
                        clusters.append(
                            cluster.loc[(self.data.df[feat] <= mini + divides) & (self.data.df[feat] >= mini)])
                        clusters.append(
                            cluster.loc[(self.data.df[feat] <= maxi - divides) & (self.data.df[feat] >= mini + divides)])
                        clusters.append(
                            cluster.loc[(self.data.df[feat] <= maxi) & (self.data.df[feat] >= maxi - divides)])
                else:
                    clusters.append(
                        self.data.df.loc[(self.data.df[feat] <= mini + divides) & (self.data.df[feat] >= mini)])
                    clusters.append(
                        self.data.df.loc[(self.data.df[feat] <= maxi - divides) & (self.data.df[feat] >= mini + divides)])
                    clusters.append(
                        self.data.df.loc[(self.data.df[feat] <= maxi) & (self.data.df[feat] >= maxi - divides)])
            else:
                cats = set(self.data.df[feat])

                if len(clusters) != 0:
                    originals = clusters.copy()
                    clusters = []
                    for ind, cluster in enumerate(originals):
                        for cat in cats:
                            clusters.append(cluster.loc[self.data.df[feat] == cat])
                else:
                    for cat in cats:
                        clusters.append(self.data.df.loc[self.data.df[feat] == cat])
        return clusters

    def k_means_clusterer(self, n):
        return KMeans(n_clusters=n, random_state=2).fit(self.data.train_df[self.data.features])

    def get_cluster(self, samples):
        return self.clusterer.predict(samples)
    # def get_kde(self, sample):



# the scale of deflection is same in image or NLP but here we are interested in not standardised deflections but actual nea values that can tell more about a feature

