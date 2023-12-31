import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd
from evaluation import Evaluation, measure_kendall_correlation
from shap_lime_cf import XAI
from neighborhood import Neighborhood
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
import random
minmax = MinMaxScaler()

cf_neighborhood_configs = ["dist_range_outside", "dist_outside", "prob_dist_outside",
                           "outside", "range_outside", "prob_outside", "prob_range_outside"]
nece_neighborhood_configs = ["dist_range_inside", "dist_inside", "prob_dist_inside",
                             "inside", "range_inside", "prob_inside", "prob_range_inside"]
fi_neighborhood_configs = ["prob", "prob_range", "prob_dist", "dist_range", "prob_dist_range", "generic", "range",
                           "dist","Normal"]

neighborhood_configs = {
    "prob": {"probability": True, "distance": False, "range": False, "restricted": False, "outside": True},
    "prob_range": {"probability": True, "distance": False, "range": True, "restricted": False, "outside": True},
    "prob_dist": {"probability": True, "distance": True, "range": False, "restricted": False, "outside": True},
    "dist_range": {"probability": False, "distance": True, "range": True, "restricted": False, "outside": True},
    "prob_dist_range": {"probability": True, "distance": True, "range": True, "restricted": False, "outside": True},
    "generic": {"probability": False, "distance": False, "range": False, "restricted": False, "outside": True},
    "range": {"probability": False, "distance": False, "range": True, "restricted": False, "outside": True},
    "dist": {"probability": False, "distance": True, "range": False, "restricted": False, "outside": True},
    "dist_range_outside": {"probability": False, "distance": True, "range": True, "restricted": True, "outside": True},
    "dist_outside": {"probability": False, "distance": True, "range": False, "restricted": True, "outside": True},
    "prob_dist_outside": {"probability": True, "distance": True, "range": False, "restricted": True, "outside": True},
    "outside": {"probability": False, "distance": False, "range": False, "restricted": True, "outside": True},
    "range_outside": {"probability": False, "distance": False, "range": True, "restricted": True, "outside": True},
    "prob_outside": {"probability": True, "distance": False, "range": False, "restricted": True, "outside": True},
    "prob_range_outside": {"probability": True, "distance": False, "range": True, "restricted": True, "outside": True},
    "dist_range_inside": {"probability": False, "distance": True, "range": True, "restricted": True, "outside": False},
    "dist_inside": {"probability": False, "distance": True, "range": False, "restricted": True, "outside": False},
    "prob_dist_inside": {"probability": True, "distance": True, "range": False, "restricted": True, "outside": False},
    "inside": {"probability": False, "distance": False, "range": False, "restricted": True, "outside": False},
    "range_inside": {"probability": False, "distance": False, "range": True, "restricted": True, "outside": False},
    "prob_inside": {"probability": True, "distance": False, "range": False, "restricted": True, "outside": False},
    "prob_range_inside": {"probability": True, "distance": False, "range": True, "restricted": True, "outside": False}
    }

proportions = [20,30,40,50,60,70]
class Framework:
    def __init__(self, eval_obj):
        self.eval_obj: Evaluation = eval_obj

    def get_special_nbr(self, sample, missing_feats):
        e = self.eval_obj
        neighbors1 = e.context.generate_neighbourhood([], sample, self.eval_obj.features,
                                                     800,
                                                     False, True, False, True)
        neighbors2 = e.context.generate_neighbourhood([], sample, self.eval_obj.features,
                                                     800,
                                                     False, True, True, True)
        for feat in missing_feats:
            neighbors1[feat] = neighbors2[feat]
        dists = e.context.calculateMahalanobis(neighbors1[e.context.feats],
                                               np.array(sample[e.features]).reshape(1, -1),
                                               np.cov(e.traindf[e.features].values))[:, 0]

        inds = np.argsort(dists)
        dists = dists[inds]
        filtered_nbrs = neighbors1.iloc[inds]
        filtered_nbrs = filtered_nbrs[dists > 0]
        dists = dists[dists > 0]
        filtered_nbrs = filtered_nbrs.iloc[:200]

        weights = 1 / dists[:200]
        weights /= sum(weights)
        return filtered_nbrs, weights

    def get_nbr(self, sample, config, fixed_feats=[], distance="Euc"):
        json = neighborhood_configs[config]
        nbr_json = {"no_of_neighbours": 1000, "probability": json["probability"], "bound": True,
                    "use_range": json["range"], "truly_random": True}
        e = self.eval_obj
        pred = e.clf.predict([sample[e.features]])[0]
        neighbors = e.context.generate_neighbourhood(fixed_feats, sample, self.eval_obj.features,
                                                     nbr_json["no_of_neighbours"],
                                                     nbr_json["probability"], True, nbr_json["use_range"], True)
        filtered_nbrs = neighbors
        if json["restricted"] == True:
            preds = np.array(self.eval_obj.clf.predict(neighbors))

            if json["outside"] == True:
                filtered_nbrs = neighbors[preds != pred]
            else:
                filtered_nbrs = neighbors[preds == pred]
        filtered_nbrs = filtered_nbrs.drop_duplicates()
        if distance=="MB":
            dists = e.context.calculateMahalanobis(filtered_nbrs[e.context.feats],
                                               np.array(sample[e.features]).reshape(1, -1),
                                               np.cov(e.traindf[e.features].values))[:, 0]
        else:
            dists = e.context.calculatel2(filtered_nbrs[e.features],
                                              np.array(sample).reshape(1, -1))[:,0]

        if json["distance"]:
            inds = np.argsort(dists)
            # inds = np.argsort(dists[:, 0])
            dists = dists[inds]
            filtered_nbrs = filtered_nbrs.iloc[inds]

        filtered_nbrs = filtered_nbrs[dists > 0]
        dists = dists[dists > 0]

        filtered_nbrs = filtered_nbrs.iloc[:200]

        weights = 1 / dists[:200]
        weights /= sum(weights)
        return filtered_nbrs, weights

    def get_fi(self, sample, nbr_config, method, dist,missing_feats = None):
        if nbr_config != "Normal" and nbr_config!="special":
            json = neighborhood_configs[nbr_config]
        if method == "SHAP":
            if nbr_config == "Normal":
                return self.eval_obj.xai.get_shap_vals(sample)
            if nbr_config == "special":
                nbrhood, weights = self.get_special_nbr(sample, missing_feats)
            else:
                if json["distance"] == False:
                    dist = "Euc"
                nbrhood, weights = self.get_nbr(sample, nbr_config,distance=dist)
            if len(nbrhood) == 0:
                return []
            xai_obj = XAI(self.eval_obj.clf, self.eval_obj.data, self.eval_obj.train_inds, "SVM", None, nbrhood)
            return xai_obj.get_shap_vals(sample)


        elif method == "LIME_weights":
            if nbr_config == "Normal":
                distances = np.linalg.norm(e.traindf[e.features].values - sample.values, axis=1)
                distances[distances==0] = 0.01

                weights = 1 / distances
                weights /= sum(weights)
                try:
                 return self.eval_obj.xai.get_lime(sample, original=True, nbrhood_=None, weights=weights)
                except:
                    print(1)

            if nbr_config == "special":
                nbrhood, weights = self.get_special_nbr(sample, missing_feats)
            else:
                if json["distance"] == False:
                    dist = "Euc"
                nbrhood, weights = self.get_nbr(sample, nbr_config, distance=dist)
            if len(nbrhood) == 0:
                return []
            xai_obj = XAI(self.eval_obj.clf, self.eval_obj.data, self.eval_obj.train_inds, "SVM", None, nbrhood)
            try:
                return xai_obj.get_lime(sample, False, nbrhood, weights)
            except Exception as er:
                print(er)
                return []

        elif method == "CF" and nbr_config!="Normal":
            if nbr_config == "special":
                nbrhood, weights = self.get_special_nbr(sample, missing_feats)
            else:
                nbrhood, weights = self.get_nbr(sample, nbr_config,distance=dist)
            if len(nbrhood) == 0:
                return []
            score = []
            for f in e.features:
                score.append(len(nbrhood[nbrhood[f] != sample[f]]))
            scores = [a / sum(score) for a in score]
            return scores
        else: return []

    def get_generic_scores(self, features_to_freeze, sample, method, config_, dist):
        json = neighborhood_configs[config_]
        if json["distance"] == False:
            dist = "Euc"
        if method == "SHAP":
            nbrhood, weights = self.get_nbr(sample, config_, features_to_freeze,distance = dist)
            xai_obj = XAI(self.eval_obj.clf, self.eval_obj.data, self.eval_obj.train_inds, "SVM", None, nbrhood)
            return xai_obj.get_shap_vals(sample)

        elif method == "LIME_weights":
            nbrhood, weights = self.get_nbr(sample, config_, features_to_freeze,distance=dist)
            xai_obj = XAI(self.eval_obj.clf, self.eval_obj.data, self.eval_obj.train_inds, "SVM", None, nbrhood)
            return xai_obj.get_lime(sample, False, nbrhood, weights)
        elif method == "CF":
            score = []
            nbrhood, weights = self.get_nbr(sample, config_, features_to_freeze,distance=dist)
            for f in e.features:
                score.append(len(nbrhood[nbrhood[f] != sample[f]]))
            scores = [a / sum(score) for a in score]
            return scores

    def check_feature_ranking(self, feature_importance_present, feature_importance_missing):
        all_feature_importance = np.concatenate([feature_importance_present, feature_importance_missing])
        ranked_features = rankdata(all_feature_importance, method='dense')
        mean_rank_present = np.mean(ranked_features[:len(feature_importance_present)])
        mean_rank_missing = np.mean(ranked_features[len(feature_importance_present):])
        return mean_rank_present > mean_rank_missing

    def missing_explananda(self, samples, neighborhood_config, method, dist, imputation_method, svm_method):
        results = []
        index_df = []
        feature_importance = []
        for prop in tqdm(proportions):
            total = 0
            cgss_ct, wgss_ct, cgsts_ct, wgsts_ct = 0, 0, 0, 0

            trial_sets = []
            no_trials = 100
            if dataname=="diab_without_insulin" and prop==20:
                no_trials = 5
            for trial in tqdm(range(no_trials)):
                missing_features = random.sample(list(range(len(self.eval_obj.features))),
                                                    int(round(prop / 100 * len(self.eval_obj.features), 0)))
                loop_count=0
                while missing_features in trial_sets and loop_count<150:
                    missing_features = random.sample(list(range(len(self.eval_obj.features))),
                                                     int(round(prop / 100 * len(self.eval_obj.features), 0)))
                    loop_count+=1
                if missing_features in trial_sets:continue
                trial_sets.append(missing_features)

                present_features = [k for k in feat_inds if k not in missing_features]

                for index, row in tqdm(samples.iterrows()):
                    index_dict = {}
                    cgs, wgs, cgst, wgst,  = -1, -1, -1,-1
                    row_with_missing = row.copy()
                    for feature in self.eval_obj.features[missing_features]:
                        row_with_missing[feature] = mean_values[feature]
                    fi_scores = self.get_fi(row_with_missing, neighborhood_config, method, dist,self.eval_obj.features[missing_features])
                    if len(fi_scores) != 0:
                        index_dict["scores"] = fi_scores
                        index_dict["missing"] = missing_features
                        index_dict["trial"] = trial
                        index_dict["index"] = index
                        index_dict["proportion"] = prop
                        index_dict["data_impute"] = imputation_method
                        index_dict["svm"] = svm_method

                        fi_scores = np.array([abs(score) for score in fi_scores])
                        original_pred = self.eval_obj.clf.predict(row[self.eval_obj.features].values.reshape(1, -1))
                        missing_pred = self.eval_obj.clf.predict(
                            row_with_missing[self.eval_obj.features].values.reshape(1, -1))
                        cgs, wgs, cgst, wgst = 0, 0, 0, 0
                        if self.check_feature_ranking(fi_scores[present_features],fi_scores[missing_features]):
                            if original_pred == missing_pred:
                                cgss_ct += 1
                                cgs = 1
                            else:
                                wgss_ct += 1
                                wgs = 1
                        else:
                            if original_pred != missing_pred:
                                cgsts_ct += 1
                                cgst = 1
                            else:
                                wgsts_ct += 1
                                wgst = 1
                        total+=1
                    index_df.append({
                        "total": total,
                        'missing_percentage': prop,
                        'correct_guesses_stop': cgs,
                        'wrong_guesses_stop': wgs,
                        'correct_guesses_start': cgst,
                        'wrong_guesses_start': wgst,
                        "index": index,
                        "trial": trial,
                        "data_impute": imputation_method,
                        "svm": svm_method
                    })
                    feature_importance.append(index_dict)
            results.append({
                'total': total,
                'missing_percentage': prop,
                'correct_guesses_stop': cgss_ct,
                'wrong_guesses_stop': wgss_ct,
                'correct_guesses_start': cgsts_ct,
                'wrong_guesses_start': wgsts_ct,
                "data_impute": imputation_method,
                "svm": svm_method
            })
        res_df = pd.DataFrame(results)
        index_df = pd.DataFrame(index_df)
        feature_importance = pd.DataFrame(feature_importance)
        res_df.to_csv(path + f'/early_stopping_{method}_{neighborhood_config}.csv', index=False)
        index_df.to_csv(path + f'/index_{method}_{neighborhood_config}.csv', index=False)
        feature_importance.to_csv(path + f'/fi_{method}_{neighborhood_config}.csv', index=False)

datasets =["cerv", "heart_db","diab_without_insulin"]
fi_scores_original = []
folders = {"diab_without_insulin":"diab",
           "heart_db":"heart",
           "cerv":"cerv"}
for dataname in datasets:
    input_data = {"data": dataname, "classifier": "SVM", "fold": "fold1"}

    svm_type = "rf"
    path = "journal_reb_2_mean_es/" + svm_type+"/" +folders[dataname]
    e = Evaluation(input_data,svm_type)

    feat_inds = list(range(len(e.features)))
    train_data = e.traindf
    data_imputation = "mean" #median, mean
    mean_values = train_data.mean()
    if data_imputation=="mean":
        for f in e.features:
            if f in e.data.continuous:
                mean_values[f] = mean_values[f].round(e.data.dec_precisions[f])
            if f in e.data.categorical:
                categories = np.array(e.data.feature_range[f])
                nearest_category_indices = np.argmin(np.abs(mean_values[f] - categories), axis=0)
                mean_values[f] = categories[nearest_category_indices]
    elif data_imputation=="median":
        mean_values = train_data.median()
    f = Framework(e)
    samples = e.testdf[e.features].iloc[:70]
    original_path = path
    # for config in tqdm(cf_neighborhood_configs+fi_neighborhood_configs+nece_neighborhood_configs+["special"]):
    #     if "dist" in config:
    #         path = original_path
    #         for method in tqdm(["SHAP"]):
    #             f.missing_explananda(samples, config, method, "MB",data_imputation,svm_type)
    #     else:
    #         path = original_path
    #         for method in tqdm(["SHAP"]):
    #             f.missing_explananda(samples, config, method, "Euc", data_imputation,svm_type)


