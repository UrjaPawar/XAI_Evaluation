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
ticks = ["SHAP", "K-LIME", "DICE-CF", "SUFF", "NECE"]
data_name = "heart_db"
model_name = "MLP"
#
#
for data_name in datasets:
    for model_name in clfs:
        filter = np.where((datas == data_name) & (models == model_name))
        for xai_method in ticks:
            percent_scores = [i * 100 for i in xai_scores_1[xai_method][filter]]
            plt.scatter(ks[filter], percent_scores, s=70, c=xai_colors[xai_method], alpha=0.6)

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks([1,2,3,4,5])
        plt.xlabel('top-K', fontsize=12)
        plt.ylabel('Evaluation Score (%)', fontsize=15)
        plt.title("Explanandum 1 on "+data_names[data_name] + " with " + model_name + " classifier ", fontsize=15)
        # plt.legend(ticks)
        # dump_path = expl_context + "_eval/" + nbr_json + "/" + \
        #             data_name + "/" + model_name + "/"
        # plt.savefig(dump_path + data_name+"_"+expl_context+"_"+"ex1_scatter.png")
        plt.savefig(expl_context + "_eval/" + data_name+"_"+model_name+"_"+expl_context+"_"+"ex1_scatter.png")
        plt.clf()
        for xai_method in ticks:
            percent_scores = [i * 100 for i in xai_scores_2[xai_method][filter]]
            plt.scatter(ks[filter], percent_scores, s=70, c=xai_colors[xai_method], alpha=0.6)

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks([1,2,3,4,5])
        plt.xlabel('top-K', fontsize=12)
        plt.ylabel('Evaluation Score (%)', fontsize=15)
        plt.title("Explanandum 2 on "+data_names[data_name] + " with " + model_name + " classifier ", fontsize=15)
        # plt.legend(ticks)
        # dump_path = expl_context + "_eval/" + nbr_json + "/" + \
        #             data_name + "/" + model_name + "/"
        # plt.savefig(dump_path + data_name+"_"+expl_context+"_"+"ex2_scatter.png")
        plt.savefig(expl_context + "_eval/"  + data_name + "_" +model_name+"_"+ expl_context + "_" + "ex2_scatter.png")
        plt.clf()


ticks = ["SHAP", "K-LIME", "DICE-CF", "SUFF", "NECE"]
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
    # dump_path = expl_context + "_eval/" + nbr_json + "/" + \
    #             data_name + "/"
    dump_path = expl_context+ "_eval/"
    for combo in [[1,2], [2,3], [4,5]]:
        ax_ind1 = 0
        fig1, axes1 = plt.subplots(1, 2, figsize=(40, 18))
        legend_ax1 = fig1.add_subplot(111, frameon=False)

        for top_k in combo:
            df_1 = pd.DataFrame(data=ex1s_none[top_k - 1], columns=ticks)
            df_1.index = clfs
            ax1 = axes1[ax_ind1]
            ax_ind1 += 1
            x1 = np.array(list(range(len(df_1.index))))
            y1 = df_1.columns
            width = 0.1
            patterns = ["/", "o", "-", "/", "*"]
            for j in range(len(y1)):
                if j in [0,2,4]:
                    ax1.bar(x1-(width*len(y1)/2) + j*width, df_1[y1[j]], width, label=y1[j], color="black", edgecolor="white", hatch=patterns[j])
                else:
                    ax1.bar(x1-(width*len(y1)/2) + j*width, df_1[y1[j]], width, label=y1[j], color="white", edgecolor="black", hatch=patterns[j])

            ax1.set(xticks=x1, xticklabels=clfs)
            ax1.tick_params(axis='both', which='major', labelsize=28)
            ax1.set_xlabel('Classifiers', fontsize=30)
            ax1.set_ylabel('Evaluation Score', fontsize=30)
            ax1.set_title("Explanandum 1, with top " + str(top_k) + " features", fontsize=33)

        legend_ax1.axis("off")
        # legend_ax.legend(*axes[0].get_legend_handles_labels(), loc='center', ncol=1, bbox_to_anchor=(1.1,0.8), fancybox = True,
        #                  shadow=True, handlelength=3, fontsize=20)
        plt.savefig(dump_path +data_name+"_"+expl_context+"_"+"ex1_top_" + str(combo[0]) + "_" + str(combo[1]) + ".png")

    for combo in [[1,2], [2,3], [4,5]]:
        ax_ind2 = 0
        fig2, axes2 = plt.subplots(1, 2, figsize=(40, 18))
        legend_ax2 = fig2.add_subplot(111, frameon=False)
        for top_k in combo:
            df_2 = pd.DataFrame(data=ex2s_none[top_k - 1], columns=ticks)
            df_2.index = clfs
            ax2 = axes2[ax_ind2]
            ax_ind2 += 1
            x2 = np.array(list(range(len(df_2.index))))
            y2 = df_2.columns
            width = 0.1
            patterns = ["/", "o", "-", "/", "*"]
            for j in range(len(y2)):
                if j in [0,2,4]:
                    ax2.bar(x2-(width*len(y2)/2) + j*width, df_2[y2[j]], width, label=y2[j], color="black", edgecolor="white", hatch=patterns[j])
                else:
                    ax2.bar(x2-(width*len(y2)/2) + j*width, df_2[y2[j]], width, label=y2[j], color="white", edgecolor="black", hatch=patterns[j])
            ax2.set(xticks=x2, xticklabels=clfs)
            ax2.tick_params(axis='both', which='major', labelsize=28)
            ax2.set_xlabel('Classifiers', fontsize=30)
            ax2.set_ylabel('Evaluation Score', fontsize=30)
            ax2.set_title("Explanandum 2, with top " + str(top_k) + " features", fontsize=33)

        legend_ax1.axis("off")
        # legend_ax.legend(*axes[0].get_legend_handles_labels(), loc='center', ncol=1, bbox_to_anchor=(1.1,0.8), fancybox = True,
        #                  shadow=True, handlelength=3, fontsize=20)
        plt.savefig( dump_path + data_name+"_"+expl_context+"_"+"ex2_top_" + str(combo[0]) + "_" + str(combo[1]) + ".png")
# plt.tight_layout()
# plt.savefig(dump_path + "top_k_1_2" + str(top_k) + ".png")
#     ax = fig.add_subplot(111)
#     bars = df_1.plot.bar(color=["#3E5641", "#6B3124", "#D36135", "#282B28", "#83BCA9"],figsize=(15,12),legend=False,ax=ax)
#     # ax = df_1.plot.bar(color = ["#3E5641", "#6B3124", "#D36135", "#282B28", "#83BCA9"],figsize=(21,12))
#     figleg.legend(bars,ticks)
#     figleg.show()
#     ax.tick_params(axis="both", labelsize=24)
#     ax.set_title("Explanandum 1, with top " + str(top_k) + " features", fontsize=35)
#     plt.xticks(rotation=45, linespacing=0.1)
#     # plt.legend(loc="upper right", prop={"size": 18})
#
#
#     plt.clf()
#
# plt.rcParams["figure.figsize"] = (30,20)

# df_2 = pd.DataFrame(data=ex1s_none[1], columns=ticks)
# df_3 = pd.DataFrame(data=ex1s_none[2], columns=ticks)
# df_4 = pd.DataFrame(data=ex1s_none[3], columns=ticks)
print("ok")
