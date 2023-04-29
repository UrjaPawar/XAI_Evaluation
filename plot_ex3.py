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
clfs = ["MLP", "Log-Reg","SVM"]
expl_context = "medical"
nbr_json = "none"
input_data = {"data": "heart_db", "fold": "fold1", "explanandum_context": expl_context,
                      "nbr_json": nbr_json}
ex3s_1 = []
ex3s_2 = []
ex3s_3 = []

datas = []
models = []
for data_name in datasets:
    ex3s_none_1 = []
    ex3s_none_2 = []
    ex3s_none_3 = []
    for clf_name in ["MLP", "SVM", "Log-Reg"]:
        ex3_ = []
        for top_k in tqdm([1, 2, 3, 4, 5]):
            dump_path = expl_context + "_eval/" + nbr_json + "/" + \
                        data_name + "/" + clf_name + "/"
            suff_nece_corrs = joblib.load(dump_path + "suff_nece_corr_" + str(top_k))
            ex_scores = joblib.load(dump_path + "ex_score_list_" + str(top_k))
            exs = np.array(ex_scores)
            ex3 = exs[:,2]
            ex3_shap = [np.array(li[0]) for li in ex3]
            ex3_lime = [np.array(li[1]) for li in ex3]
            ex3_dice = [np.array(li[2]) for li in ex3]
            ex3_suff = [np.array(li[3]) for li in ex3]
            ex3_nece = [np.array(li[4]) for li in ex3]
            ex3_.append(np.array([ex3_shap,ex3_lime,ex3_dice,ex3_suff,ex3_nece]))
            # datas.extend([data_name for _ in range(len(exs))])
            # models.extend([clf_name for _ in range(len(exs))])
            # ks.extend([top_k for _ in range(len(exs))])
        ex3s_none_1.append([[ex3_[i][j][:,0] for j in range(5)] for i in range(5)])
        ex3s_none_2.append([[ex3_[i][j][:,1] for j in range(5)] for i in range(5)])
        ex3s_none_3.append([[ex3_[i][j][:,2] for j in range(5)] for i in range(5)])
    ex3s_1.append(ex3s_none_1)
    ex3s_2.append(ex3s_none_2)
    ex3s_3.append(ex3s_none_3)

datas = np.array(datas)
models = np.array(models)
data_colors = {"diab_without_insulin": "blue", "cervicle":"red", "heart_db":"green"}
data_names = {"diab_without_insulin": "Diabetes", "cerv":"Cervical Cancer", "heart_db":"Heart Disease"}
models_colors = {"MLP": "blue", "SVM":"red", "Log-Reg":"green"}
xai_colors = {"SHAP":"blue", "K-LIME":"purple", "DICE-CF":"green", "SUFF":"red", "NECE":"pink"}

colors = [models_colors[model] for model in models]
repl_name = ["replacing_top_1", "replacing_top_2", "replacing_top_3"]
subset_name = ["subset_of_1_feature", "subset_of_2_features", "subset_of_3_features", "subset_of_4_features", "subset_of_5_features"]
#
# for top_k in range(1, 6):
#     for d,data_name in enumerate(datasets):
#         for m,model_name in enumerate(clfs):
#             ex_shap = []
#             ex_lime = []
#             ex_dice = []
#             ex_suff = []
#             ex_nece = []
#             ks = []
#             for n, ex_score in enumerate([ex3s_1, ex3s_2, ex3s_3]):
#                 ks.extend([n+1]*len(ex_score[d][m][top_k-1][0]))
#                 ex_shap.extend(ex_score[d][m][top_k-1][0])
#                 ex_lime.extend(ex_score[d][m][top_k-1][1])
#                 ex_dice.extend(ex_score[d][m][top_k-1][2])
#                 ex_suff.extend(ex_score[d][m][top_k-1][3])
#                 ex_nece.extend(ex_score[d][m][top_k-1][4])
#
#             xai_scores_3 = {"SHAP": ex_shap, "K-LIME": ex_lime, "DICE-CF": ex_dice, "SUFF": ex_suff,
#                             "NECE": ex_nece}
#             for xai_method in ticks:
#                 percent_scores = [i * 100 for i in xai_scores_3[xai_method]]
#                 plt.scatter(ks, percent_scores, s=70, c=xai_colors[xai_method], alpha=0.6)
#
#             plt.tick_params(axis='both', which='major', labelsize=12)
#             plt.xticks([1,2,3,4,5])
#             plt.xlabel('top-K (Replaced Features)', fontsize=12)
#             plt.ylabel('Evaluation Score (%)', fontsize=15)
#             plt.title("Explanandum 3 on "+data_names[data_name] + " with " + model_name + " classifier ", fontsize=15)
#             plt.savefig(expl_context + "_eval/" + data_name+"_"+model_name+"_"+expl_context+"_"+ subset_name[top_k-1]+"_ex3_scatter.png")
#             plt.clf()
#
#
# for n,ex_score in enumerate([ex3s_1, ex3s_2, ex3s_3]):
#     for d,data_name in enumerate(datasets):
#         for m,model_name in enumerate(clfs):
#             ex_s = ex_score[d][m]
#             ex_shap = []
#             ex_lime = []
#             ex_dice = []
#             ex_suff = []
#             ex_nece = []
#             ks = []
#             for top_k in range(1, 6):
#                 ks.extend([top_k]*len(ex_score[d][m][top_k-1][0]))
#                 ex_shap.extend(ex_score[d][m][top_k-1][0])
#                 ex_lime.extend(ex_score[d][m][top_k-1][1])
#                 ex_dice.extend(ex_score[d][m][top_k-1][2])
#                 ex_suff.extend(ex_score[d][m][top_k-1][3])
#                 ex_nece.extend(ex_score[d][m][top_k-1][4])
#
#             xai_scores_3 = {"SHAP": ex_shap, "K-LIME": ex_lime, "DICE-CF": ex_dice, "SUFF": ex_suff,
#                             "NECE": ex_nece}
#             for xai_method in ticks:
#                 percent_scores = [i * 100 for i in xai_scores_3[xai_method]]
#                 plt.scatter(ks, percent_scores, s=70, c=xai_colors[xai_method], alpha=0.6)
#
#             plt.tick_params(axis='both', which='major', labelsize=12)
#             plt.xticks([1,2,3,4,5])
#             plt.xlabel('No. of Features in Subset', fontsize=12)
#             plt.ylabel('Evaluation Score (%)', fontsize=15)
#             plt.title("Explanandum 3 on "+data_names[data_name] + " with " + model_name + " classifier ", fontsize=15)
#             plt.savefig(expl_context + "_eval/" + data_name+"_"+model_name+"_"+expl_context+"_"+ repl_name[n]+"_ex3_scatter.png")
#             plt.clf()

for n,ex_score in enumerate([ex3s_1, ex3s_2, ex3s_3]):
    for d,data_name in enumerate(datasets):
        ex3s_none = []
        for m, clf_name in enumerate(["MLP", "SVM", "Log-Reg"]):
            ex3 = []
            for top_k in tqdm([4, 5]):
                ex_scores = ex_score[d][m][top_k-1]
                exs = np.array(ex_scores)
                ex3.append(np.mean(exs, axis=1))
            res = np.mean(ex3,axis=0)
            for t,tick in enumerate(ticks):
                print(data_name + " : " + repl_name[n] + " : " + clf_name + ": " + str(res[t]) + " ", tick)

# for n,ex_score in enumerate([ex3s_1, ex3s_2, ex3s_3]):
#     for d,data_name in enumerate(datasets):
#         ex3s_none = []
#         for top_k in tqdm([1, 2, 3, 4, 5]):
#             ex3 = []
#             for m,clf_name in enumerate(["MLP", "SVM", "Log-Reg"]):
#                 ex_scores = ex_score[d][m][top_k-1]
#                 exs = np.array(ex_scores)
#                 ex3.append(np.mean(exs, axis=1))
#             print(data_name + " : " + repl_name[n] + " : " + clf_name + "_" + "subset: " + str(top_k) + "_" )

        #     ex3s_none.append(ex3)
        # dump_path = expl_context+ "_eval/"
        # for combo in [[1,2], [3,4], [4,5]]:
        #     ax_ind1 = 0
        #     fig1, axes1 = plt.subplots(1, 2, figsize=(40, 18))
        #     legend_ax1 = fig1.add_subplot(111, frameon=False)
        #     for top_k in combo:
        #         df_1 = pd.DataFrame(data=ex3s_none[top_k - 1], columns=ticks)
        #         df_1.index = clfs
        #         ax1 = axes1[ax_ind1]
        #         ax_ind1 += 1
        #         x1 = np.array(list(range(len(df_1.index))))
        #         y1 = df_1.columns
        #         width = 0.1
        #         patterns = ["/", "o", "-", "/", "*"]
        #         for j in range(len(y1)):
        #             if j in [0,2,4]:
        #                 ax1.bar(x1-(width*len(y1)/2) + j*width, df_1[y1[j]], width, label=y1[j], color="black", edgecolor="white", hatch=patterns[j])
        #             else:
        #                 ax1.bar(x1-(width*len(y1)/2) + j*width, df_1[y1[j]], width, label=y1[j], color="white", edgecolor="black", hatch=patterns[j])
        #
        #         ax1.set(xticks=x1, xticklabels=clfs)
        #         ax1.tick_params(axis='both', which='major', labelsize=28)
        #         ax1.set_xlabel('Classifiers', fontsize=30)
        #         ax1.set_ylabel('Evaluation Score', fontsize=30)
        #         ax1.set_title("Explanandum 3, in subset of " + str(top_k) + " features", fontsize=33)
        #
        #     legend_ax1.axis("off")
        #     plt.savefig(dump_path +data_name+"_"+repl_name[n]+"_"+expl_context+"_"+"ex3_top_" + str(combo[0]) + "_" + str(combo[1]) + ".png")
        #
