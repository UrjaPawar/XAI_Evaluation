from neighborhood import Neighborhood
import numpy as np
class Nece_Suff:
    def __init__(self, neighborhood_obj):
        self.neighborhood_obj: Neighborhood = neighborhood_obj

    def remove_values_from_list(self, the_list, val):
        return [value for value in the_list if value != val]

    def necessity(self, sample, output, model, train_df, neighborhood_json, use_metric='None'):
        scores = []
        for feat in self.neighborhood_obj.feats:
            neighbors = self.neighborhood_obj.generate_neighbourhood([feat], sample, self.neighborhood_obj.feats,
                                                                     no_of_neighbours=neighborhood_json["no_of_neighbours"]*4,
                                                                     probability=neighborhood_json["probability"],
                                                                     bound=neighborhood_json["bound"],
                                                                     use_range=neighborhood_json["use_range"],
                                                                     truly_random=neighborhood_json["truly_random"])
            preds = model.predict(neighbors)
            # check that the output is same as "output"
            size = neighborhood_json["no_of_neighbours"]
            neighbors = neighbors.iloc[preds == output]
            # filter out ngihbours with same prediction
            if use_metric == 'MB':
                dists = self.neighborhood_obj.calculateMahalanobis(neighbors[self.neighborhood_obj.feats],
                                                                   np.array(sample).reshape(1, -1),
                                                                   np.cov(train_df[self.neighborhood_obj.feats].values))
                inds = np.argsort(dists[:, 0])
                neighbors = neighbors.iloc[inds]
            elif use_metric == "Euc":
                dists = [self.neighborhood_obj.calculatel2(neighbors[self.neighborhood_obj.feats].iloc[i], sample) for i in range(len(neighbors))]
                inds = np.argsort(dists)
                neighbors = neighbors.iloc[inds]
            neighbors = neighbors.iloc[:size]
            if feat not in self.neighborhood_obj.continuous:
                select = self.neighborhood_obj.feature_range[feat]
                if sample[feat] in select:
                    select = self.remove_values_from_list(select,sample[feat])
                neighbors[feat] = np.random.choice(select, len(neighbors))
            elif self.neighborhood_obj.precisions[feat] == 0:
                select = list(range(int(self.neighborhood_obj.mini[feat]), int(self.neighborhood_obj.maxi[feat])))
                if sample[feat] in select:
                    select = self.remove_values_from_list(select,sample[feat])
                neighbors[feat] = np.random.choice(select, len(neighbors))
            else:
                select = list(np.random.uniform(self.neighborhood_obj.mini[feat], self.neighborhood_obj.maxi[feat], 2 * len(neighbors)))
                select = [round(r, self.neighborhood_obj.precisions[feat]) for r in select]
                if sample[feat] in select:
                    select = self.remove_values_from_list(select,sample[feat])
                select = select[:len(neighbors)]
                neighbors[feat] = select
            check = list(neighbors[feat]==sample[feat])
            # if sum([int(ch) for ch in check])==0:
            #     print("check successful")
            # else: print("check unsuccessful",feat)
            if len(neighbors)==0:
                print(feat, sample)
                score = -2
            else:
                preds = model.predict(neighbors)
            # neighbors['Colestrol'].hist().show()
                score = sum((preds != output).astype(int)) / len(neighbors)
            scores.append(round(score, 2))
        return scores


    def sufficiency(self, sample, output, model, train_df, neighborhood_json, use_metric="MB"):
        scores = []
        for feat in self.neighborhood_obj.feats:
            neighbors = self.neighborhood_obj.generate_neighbourhood([], sample, self.neighborhood_obj.feats,
                                                                     no_of_neighbours=neighborhood_json["no_of_neighbours"]*4,
                                                                     probability=neighborhood_json["probability"],
                                                                     bound=neighborhood_json["bound"],
                                                                     use_range=neighborhood_json["use_range"],
                                                                     truly_random=neighborhood_json["truly_random"])
            neighbors = neighbors[neighbors[feat] != sample[feat]]
            preds = model.predict(neighbors)
            size = neighborhood_json["no_of_neighbours"]
            # filter out neighbours with different preds
            neighbors = neighbors.iloc[preds != output]
            if len(neighbors)==0:
                print(feat, sample)
                score = -2
            else:
                if use_metric=="MB":
                    dists = self.neighborhood_obj.calculateMahalanobis(neighbors[self.neighborhood_obj.feats], np.array(sample).reshape(1,-1), np.cov(train_df[self.neighborhood_obj.feats].values))
                    inds = np.argsort(dists[:, 0])
                    neighbors = neighbors.iloc[inds]
                elif use_metric == "Euc":
                    dists = [self.neighborhood_obj.calculatel2(neighbors[self.neighborhood_obj.feats].iloc[i], sample) for i
                             in range(len(neighbors))]
                    inds = np.argsort(dists)
                    neighbors = neighbors.iloc[inds]
                neighbors = neighbors.iloc[:size]
                if len(neighbors) > 0:
                    neighbors[feat] = [sample[feat]] * len(neighbors)
                    preds = model.predict(neighbors)
                    score = sum((preds == output).astype(int)) / len(neighbors)
                else:
                    score = 0
            scores.append(round(score, 2))
        return scores