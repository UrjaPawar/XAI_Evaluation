import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from med_dataset import Data
from neighborhood import Neighborhood
from suff_nece import Nece_Suff
from shap_lime_cf import XAI
from density_cluster import Density
from sklearn.metrics import recall_score,accuracy_score
from tqdm import tqdm
from plotting import plot_histograms
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import statistics
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


def measure_kendall_correlation(ranking1, ranking2):
    kendal = stats.kendalltau(ranking1, ranking2)
    return kendal.correlation


class Evaluation:
    def __init__(self, input_data, svm_type):
        self.input_data = input_data
        self.svm_type = svm_type
        self.data = self.get_data(False)
        self.inds_path = "analysis_outputs/" + input_data["data"] + "/" + input_data["fold"]
        self.test_inds = joblib.load(self.inds_path + "/test")
        self.train_inds = joblib.load(self.inds_path + "/train")
        if input_data["data"] == "cerv":
            self.train_inds = self.train_inds[self.train_inds <= 1252]
            self.test_inds = self.test_inds[self.test_inds <= 1252]

        self.clf = self.get_clf()
        print("recall: ", self.get_recall())
        self.context = Neighborhood(self.data)
        self.traindf = self.data.df.iloc[self.train_inds]
        self.testdf = self.data.df.iloc[self.test_inds]
        self.density = Density(self.data, self.train_inds)
        self.nece_suff_obj = Nece_Suff(self.context)
        self.features = np.array(self.data.features)
        self.densities = self.density.get_density_score(self.testdf[self.features])
        self.top_10_high_density = np.argsort(self.densities)[-10:]
        self.top_10_low_density = np.argsort(self.densities)[:10]

        self.xai = XAI(self.clf, self.data, self.train_inds, input_data["classifier"], self.density)
        self.means = {}
        for feat in self.features:
            if feat in self.data.continuous:
                self.means[feat] = round(np.mean(self.traindf[feat]), self.data.dec_precisions[feat])
            else:
                self.means[feat] = statistics.mode(self.traindf[feat])
        # self.expln_use_range = input_data["explanandum_context"] == "medical"
        # neighborhood_json_none = {"no_of_neighbours": 500, "probability": False, "bound": True, "use_range": False,
        #                           "truly_random": True}
        # neighborhood_json_prob = {"no_of_neighbours": 500, "probability": True, "bound": True, "use_range": False,
        #                           "truly_random": True}
        #
        # self.nbr_dict = {
        #     "none": neighborhood_json_none,
        #     "prob": neighborhood_json_prob,
        # }
        # self.nbr_json = self.nbr_dict[input_data["nbr_json"]]
        # self.nbr_json = self.input_data["nbr_json"]

    def get_data(self, hot_encode):
        the_data = None
        if self.input_data["data"] == "heart_db":
            the_data = Data("Heart DB", hot_encode)
        elif self.input_data["data"] == "cerv":
            the_data = Data("Cervical DB", hot_encode)
        elif self.input_data["data"] == "diab_insulin" or self.input_data["data"] == "diab_without_insulin":
            the_data = Data("Diabetes DB", hot_encode, pick_without_insulin=True)
        elif self.input_data["data"] == "iris":
            the_data = Data("Iris", hot_encode)
        elif self.input_data["data"] == "wine":
            the_data = Data("Wine", hot_encode)
        return the_data

    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    def knn_grid_search(self,X_train, y_train, param_grid, cv=5):
        # Initialize KNN classifier
        knn = KNeighborsClassifier()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(knn, param_grid, cv=cv)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Best parameters and best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        return best_params, best_model

    def mlp_grid_search(self,X_train, y_train, param_grid, cv=5):

        mlp = MLPClassifier()  # Increase max_iter for convergence

        # Initialize GridSearchCV
        grid_search = GridSearchCV(mlp, param_grid, cv=cv)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Best parameters and best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        return best_params, best_model





    def get_clf(self):
        from sklearn import svm
        if self.input_data["classifier"] == "MLP":
            clf = MLPClassifier(random_state=1, max_iter=300)
            clf.fit(self.data.df[self.data.features].iloc[self.train_inds],
                    self.data.df[self.data.target].iloc[self.train_inds])
        elif self.input_data["classifier"] == "SVM":
            if self.svm_type=="linear":
                clf = svm.SVC(kernel='linear',C=1, probability=True,random_state=1)
            elif self.svm_type=="rbf":
                clf = svm.SVC(kernel='rbf',C=1000, probability=True, random_state=1)
            elif self.svm_type=="rf":
                clf =RandomForestClassifier(max_depth=5, random_state=0)
            elif self.svm_type=="knn":
                # clf = MLPClassifier(random_state=1, max_iter=300)
                # clf = LogisticRegression(max_iter=1000)
                clf = KNeighborsClassifier(n_neighbors=10)
                if self.input_data["data"] == "heart_db":
                    clf = KNeighborsClassifier(n_neighbors=7,algorithm="kd_tree",leaf_size=10,weights="distance",metric="manhattan")
                # #     # Example usage
                #     param_grid = {
                #         'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                #         'max_iter': [100,200,300,400,500,600,700,800,900],
                #         'alpha': [0.0001, 0.001, 0.01],
                #         'learning_rate': ['constant', 'adaptive'],
                #     }
                # #
                # #     # Assuming X_train and y_train are already defined
                #     best_params, clf = self.mlp_grid_search(self.data.df[self.data.features], self.data.df[self.data.target], param_grid)
                #     print("Best Parameters:", best_params)


            else:
                clf = svm.SVC(kernel='poly', C=100, probability=True, random_state=1)
            clf.fit(self.data.df[self.data.features].iloc[self.train_inds],
                    self.data.df[self.data.target].iloc[self.train_inds])

        else:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(self.data.df[self.data.features].iloc[self.train_inds],
                    self.data.df[self.data.target].iloc[self.train_inds])
        return clf

    def get_recall(self):
        return recall_score(self.clf.predict(self.data.df[self.data.features].iloc[self.test_inds][self.data.features]),
                            self.data.df.iloc[self.test_inds][self.data.target],average="micro")

    def ex3(self, sample_for_ex3, n, k, scores, output, clf):
        total = 0
        points = 0
        for i in range(200):
            features_to_change_from = np.random.choice(len(self.features), n, replace=False)
            temp = sample_for_ex3.copy()
            for feat in self.features[features_to_change_from]:
                temp[feat] = self.means[feat]
            if clf.predict([temp])[0] != output:
                scores = np.array(scores)
                inds_sorted = np.argsort(scores[features_to_change_from])
                features_revert = self.features[features_to_change_from[inds_sorted[-k:]]]
                for feature_revert in features_revert:
                    temp[feature_revert] = sample_for_ex3[feature_revert]
                if clf.predict([temp])[0] == output:
                    points += 1
                total += 1
        if total != 0:
            return points / total
        else:
            return 0
# "diab_without_insulin"
datasets = ["cerv","heart_db"]
ticks = ["SHAP", "K-LIME", "DICE-CF", "SUFF", "NECE"]
clfs = [ "MLP", "Log-Reg","SVM"]
expl_context = "medical"
nbr_json = "none"


# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV

# for data_name in datasets:
#     input_data = {"data": data_name, "classifier": "SVM", "fold": "fold1", "explanandum_context": expl_context,
#                   "nbr_json": nbr_json}
#     print(input_data)
#     eval = Evaluation(input_data)
#     param_grid = {
#         'C': [0.1, 1, 10, 100],
#         'gamma': [1, 0.1, 0.01, 0.001],
#         'kernel': ['rbf', 'linear']
#     }
#     svm = SVC()
#     grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=2, n_jobs=-1)
#     grid_search.fit(eval.traindf[eval.features], eval.traindf[eval.data.target])
#     best_params = grid_search.best_params_
#     print(best_params)

# {'C': 1, 'gamma': 1, 'kernel': 'linear'}
# {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# {'C': 0.1, 'gamma': 1, 'kernel': 'linear'}
