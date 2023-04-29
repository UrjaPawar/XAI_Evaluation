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
from tqdm import tqdm
from plotting import plot_histograms
import statistics
import warnings

warnings.filterwarnings("ignore")

datasets = ["diab_without_insulin","cerv", "heart_db"]
ticks = ["SHAP", "K-LIME", "DICE-CF", "SUFF", "NECE"]
clfs = [ "MLP", "Log-Reg","SVM"]
# expl_contexts = ["medical", "random"]
# nbr_jsons = ["none", "prob"]
expl_context = "medical"
nbr_json = "none"
input_data = {"data": "heart_db", "fold": "fold1", "explanandum_context": expl_context,
                      "nbr_json": nbr_json}

# TODO check with use range =True explanandums
# TODO Finalise neighborhood for each thing - nece suff neigbhborhood none with mb,
# TODO density wise analysis of correlation and explananda


ex1s_shap = []
ex1s_dice = []
ex1s_lime = []
ex1s_nece = []
ex1s_suff = []
ex2s_shap = []
ex2s_dice = []
ex2s_lime = []
ex2s_nece = []
ex2s_suff = []
datas = []
models = []
ks = []
for data_name in datasets:
    for top_k in tqdm([1, 2, 3, 4, 5]):
        for clf_name in ["MLP", "SVM", "Log-Reg"]:
            dump_path = expl_context + "_eval/" + nbr_json + "/" + \
                        data_name + "/" + clf_name + "/"
            ex_scores = joblib.load(dump_path + "ex_score_list_" + str(top_k))
            exs = np.array(ex_scores)
            ex1 = exs[:, 0]
            ex2 = exs[:, 1]
            ex1s_shap.extend(ex1[:,0])
            ex1s_lime.extend(ex1[:,1])
            ex1s_dice.extend(ex1[:,2])
            ex1s_suff.extend(ex1[:,3])
            ex1s_nece.extend(ex1[:,4])
            ex2s_shap.extend(ex2[:,0])
            ex2s_lime.extend(ex2[:,1])
            ex2s_dice.extend(ex2[:,2])
            ex2s_suff.extend(ex2[:,3])
            ex2s_nece.extend(ex2[:,4])
            datas.extend([data_name for _ in range(len(exs))])
            models.extend([clf_name for _ in range(len(exs))])
            ks.extend([top_k for _ in range(len(exs))])
ex1s_shap = np.array(ex1s_shap)
ex1s_lime = np.array(ex1s_lime)
ex1s_dice = np.array(ex1s_dice)
ex1s_suff = np.array(ex1s_suff)
ex1s_nece = np.array(ex1s_nece)
ex2s_shap = np.array(ex2s_shap)
ex2s_lime = np.array(ex2s_lime)
ex2s_dice = np.array(ex2s_dice)
ex2s_suff = np.array(ex2s_suff)
ex2s_nece = np.array(ex2s_nece)
datas = np.array(datas)
models = np.array(models)
ks = np.array(ks)
data_colors = {"diab_without_insulin": "blue", "cerv":"red", "heart_db":"green"}
data_names = {"diab_without_insulin": "Diabetes", "cerv":"Cervical Cancer", "heart_db":"Heart Disease"}
models_colors = {"MLP": "blue", "SVM":"red", "Log-Reg":"green"}
xai_colors = {"SHAP":"blue", "K-LIME":"purple", "DICE-CF":"green", "SUFF":"red", "NECE":"pink"}
xai_scores_1 = {"SHAP":ex1s_shap, "K-LIME":ex1s_lime, "DICE-CF":ex1s_dice, "SUFF":ex1s_suff, "NECE":ex1s_nece}
xai_scores_2 = {"SHAP":ex2s_shap, "K-LIME":ex2s_lime, "DICE-CF":ex2s_dice, "SUFF":ex2s_suff, "NECE":ex2s_nece}

# colors = [data_colors[d] for d in datas]
colors = [models_colors[model] for model in models]
kss = [i*200 for i in ks]
# fig,ax = plt.subplots(2,3,figsize=(25,18))
# plt.scatter(models, sizes, s=kss, c=colors, alpha=0.2)
i = 0
j = 0
data_name = "heart_db"
model_name = "MLP"

res = []
for expl_context in ["random", "medical"]:
    list_res = []
    for data_name in datasets:
        ex1s_none = []
        ex2s_none = []
        for top_k in tqdm([1, 2, 3, 4, 5]):
            ex1 = []
            ex2 = []
            for clf_name in ["MLP", "SVM", "Log-Reg"]:
                dump_path = expl_context + "_eval/" + nbr_json + "/" + \
                            data_name + "/" + clf_name + "/"
                suff_nece_corrs = joblib.load(dump_path + "suff_nece_corr_" + str(top_k))
                ex_scores = joblib.load(dump_path + "ex_score_list_" + str(top_k))
                exs = np.array(ex_scores)
                ex1.append(np.mean(exs[:, 0], axis=0))
                ex2.append(np.mean(exs[:, 1], axis=0))
            ex1s_none.append(ex1)
            ex2s_none.append(ex2)
        dump_path = expl_context+ "_eval/"
        df1s = []
        df2s = []
        for top_k in [1,2,3,4,5]:
            df_1 = pd.DataFrame(data=ex1s_none[top_k - 1], columns=ticks)
            df_1.index = clfs
            df_2 = pd.DataFrame(data=ex2s_none[top_k - 1], columns=ticks)
            df_2.index = clfs
            df1s.append(df_1)
            df2s.append(df_2)

        for model_name in clfs:
            for xai in ticks:
                print(data_name,"; ",model_name,": ", xai, ": ", np.mean([d[xai].loc[model_name] for d in df2s]))
                list_res.append(np.mean([d[xai].loc[model_name] for d in df2s]))
    res.append(list_res)


print("ok")
