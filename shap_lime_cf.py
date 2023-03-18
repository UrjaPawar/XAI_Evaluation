from med_dataset import Data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular


class SHAP_LIME:
    def __init__(self, clf, data: Data, train_inds, model_name, custom_neighborhood=None):
        self.model = clf
        self.data = data
        self.train_df = data.df[data.features].iloc[train_inds]
        self.custom_neighborhood = custom_neighborhood[data.features]
        self.feats = data.features
        self.model_name = model_name
        self.shap_explainer = self.get_shap_explainer()
        self.lime_explainer = self.get_lime()



    def get_shap_explainer(self):
        if self.model_name == "log_clf":
            # return shap.Explainer(self.model, self.train_df[self.feats].iloc[15:25])
            return shap.Explainer(self.model, self.custom_neighborhood)
        elif self.model_name == "rf_clf":
            return shap.TreeExplainer(self.model)
        if self.model_name == "MLP" or self.model_name == "SVM":
            # masker = shap.maskers.Independent(data = self.train_df[self.feats])
            masker = shap.maskers.Independent(data=self.custom_neighborhood)
            # gradient and deep explainers might be requiring Image type inputs
            # return shap.KernelExplainer(self.model.predict, data=self.train_df[self.feats], masker=masker)
            return shap.KernelExplainer(self.model.predict, data=self.custom_neighborhood, masker=masker)

    def get_shap_vals(self, sample):
        if self.model_name=="log_clf":
            shap_vals = self.shap_explainer(np.array(sample).reshape(1,-1))[0]
            return shap_vals.values
        elif self.model_name == "rf_clf":
            shap_vals = self.shap_explainer(np.array(sample).reshape(1, -1))
            return shap_vals.values[:,:,1] # for class 1
        elif self.model_name == "MLP" or self.model_name == "SVM":
            shap_vals = self.shap_explainer.shap_values(np.array(sample).reshape(1, -1))
            # return shap_vals.values[:, :][0] # for MLP - 2D output as
            return shap_vals[0]
            # both classes show different shap values, we need both

    def get_lime(self):
        return lime.lime_tabular.LimeTabularExplainer(self.custom_neighborhood[self.feats].values[:, :],
                                                      feature_names=self.feats,
                                                      class_names=self.data.target,categorical_features=self.data.categorical)

    def get_lime_values(self, sample):
        # if self.model_name == "MLP":
        #     lime_vals = self.lime_explainer.explain_instance(np.array(sample), self.model,
        #                                                      num_features=len(self.feats))
        # else:
        lime_vals = self.lime_explainer.explain_instance(np.array(sample), self.model.predict_proba,
                                                             num_features=len(self.feats),num_samples=200)
        scores = []
        for expln in lime_vals.as_list():
            for f in self.feats:
                if f in expln[0]:
                    scores.append(abs(expln[1]))
        return scores
