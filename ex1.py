import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from med_dataset import Data
from neighborhood import Neighborhood
from suff_nece import Nece_Suff
from shap_lime_cf import SHAP_LIME
from sklearn.metrics import accuracy_score, recall_score, precision_score
from density_cluster import Density
from tqdm import tqdm
from plotting import plot_histograms
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import statistics
import warnings
import dice_ml

warnings.filterwarnings("ignore")
def get_data(hot_encode):
    data = None
    if input_data["data"] == "heart_db":
        data = Data("Heart DB", hot_encode)
    elif input_data["data"] == "cerv":
        data = Data("Cervical DB", hot_encode)
    elif input_data["data"] == "diab_insulin" or input_data["data"] == "diab_without_insulin":
        data = Data("Diabetes DB", hot_encode, pick_without_insulin=True)
    return data

input_data = {"data": "heart_db", "classifier": "SVM", "fold": "fold1"}
path = "analysis_outputs/" + input_data["data"] + "/" + input_data["fold"]

test_inds = joblib.load(path + "/test")
train_inds = joblib.load(path + "/train")

data: Data = get_data(False)

if input_data["classifier"]=="MLP":
    clf = MLPClassifier(random_state=1,max_iter=300)
    clf.fit(data.df[data.features].iloc[train_inds], data.df[data.target].iloc[train_inds])
elif input_data["classifier"]=="SVM":
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(data.df[data.features].iloc[train_inds], data.df[data.target].iloc[train_inds])
else:
    clf = joblib.load(path+"/"+input_data["classifier"])



def get_accuracy():
    return accuracy_score(clf.predict(data.df[data.features].iloc[test_inds][data.features]),
                          data.df.iloc[test_inds][data.target])

def get_precision():
    return precision_score(clf.predict(data.df[data.features].iloc[test_inds][data.features]),
                           data.df.iloc[test_inds][data.target])

def get_recall():
    return recall_score(clf.predict(data.df[data.features].iloc[test_inds][data.features]),
                        data.df.iloc[test_inds][data.target])

# nbrhood of suff and nece, toy d
context = Neighborhood(data)



def measure_kendall_correlation(ranking1, ranking2):
    kendal = stats.kendalltau(ranking1, ranking2)
    return kendal.correlation


def generate_csv(scores, clf, indices, histname):
    class_0 = list(clf.predict_proba(data.df[data.features].iloc[indices])[:, 0])
    class_1 = list(clf.predict_proba(data.df[data.features].iloc[indices])[:, 1])
    truths = list(data.df[data.target].iloc[indices])
    density = Density(data)
    clusters = list(density.get_cluster(data.df[data.features].iloc[indices]))
    densities = list(density.get_density_score(data.df[data.features].iloc[indices]))
    entries = np.array(
        [indices, class_0, class_1, truths, densities, clusters, scores[0], scores[1], scores[2], scores[3]])
    df = pd.DataFrame(
        columns=["indices", "class_0", "class_1", "truth", "density", "cluster", "kendal_shap_suff", "kendal_lime_suff",
                 "kendal_shap_nece", "kendal_lime_nece"],
        data=entries.transpose())
    df.to_csv(path + "/results_mb_big/results_corr" + histname + ".csv", index=False)


traindf = data.df.iloc[train_inds]
testdf = data.df.iloc[test_inds]
density = Density(data)
clusters = list(density.get_cluster(traindf[data.features]))
nece_suff = Nece_Suff(context)



densities = density.get_density_score(testdf[data.features])
top_10_high_density = np.argsort(densities)[-10:]
top_10_low_density = np.argsort(densities)[:10]
shap_org = SHAP_LIME(clf, data, train_inds, input_data["classifier"], custom_neighborhood=traindf)

features = np.array(data.features)

# featurestoy_data_continuous =
# toy_data_categorical =
# toy_data_binary =
def ex3(sample_for_ex3, n, k, scores, output, clf):
    total = 0
    points = 0
    for i in range(200):
        features_to_change_from = np.random.choice(len(features), n, replace=False)
        temp = sample_for_ex3.copy()
        for feat in features[features_to_change_from]:
            if feat in data.continuous:
                temp[feat] = round(np.mean(traindf[feat]), data.dec_precisions[feat])
            else:
                temp[feat] = statistics.mode(traindf[feat])
        if clf.predict([temp])[0]!=output:
            scores = np.array(scores)
            inds_sorted = np.argsort(scores[features_to_change_from])
            features_revert = features[inds_sorted[-k:]]
            for feature_revert in features_revert:
                temp[feature_revert] = sample_for_ex3[feature_revert]
            if clf.predict([temp])[0]!=output:
                points+=1
            total+=1
    if total!=0:
        return points/total
    else: return -1
    # at random stage selection excluding n features

# less dense more dense
def ex2(sample_for_ex2, k, scores, output, clf):
    # changing rest of the features and keeping top k features constant could also be tried
    inds_sorted = np.argsort(scores)
    # mean_sample = sample_for_ex2.copy()
    nbr_ex1 = context.generate_neighbourhood(features[inds_sorted[-k:]], sample_for_ex2, data.features, 200, False, True,
                                             False, True)
    outputs = clf.predict(nbr_ex1)
    return len(outputs[outputs == output]) / len(outputs)
    # for feat in features[inds_sorted[:k]]:
    #     if feat in data.continuous:
    #         mean_sample[feat] = round(np.mean(traindf[feat]),data.dec_precisions[feat])
    #     else:
    #         mean_sample[feat] = statistics.mode(traindf[feat])
    # original = clf.predict_proba([sample_for_ex2])[0][output]
    # new_op = clf.predict([mean_sample])[0]
    # if new_op == output:
    #     return 1
    # else: return 0

def ex1(sample_for_ex1, k, scores, output, clf):
    inds_sorted = np.argsort(scores)
    inds_to_fix = inds_sorted[:-k]
    nbr_ex1 = context.generate_neighbourhood(features[inds_to_fix], sample_for_ex1, data.features, 200, False, True,
                                                 False, True)
    outputs = clf.predict(nbr_ex1)
    return len(outputs[outputs != output]) / len(outputs)
    # else:
    #     nbr_ex1 = context.generate_neighbourhood(features[inds_to_fix], sample_for_ex1, data.features, 200, False, True,
    #                                              True, True)
    # put this across as any permutation of change happening in top k features
    # for feat in features[inds_sorted[-k:]]:
    #     nbr_ex1 = nbr_ex1[nbr_ex1[feat]!=sample_for_ex1[feat]]

ex1s_none = []
ex2s_none = []
ex3s_none = []
ex1s_prob = []
ex2s_prob = []
ex3s_prob = []
for nbr_json in ["none","prob"]:
    for top_k in tqdm([1, 2, 3, 4, 5, 6]):
        ex1 = []
        ex2 = []
        ex3 = []
        for clf_name in ["MLP", "SVM", "log_clf"]:
            dump_path = input_data['data'] + "/" + nbr_json + "/" + clf_name + "/"
            suff_nece_corrs = joblib.load(dump_path+"suff_nece_corr_"+str(top_k))
            ex_scores = joblib.load(dump_path+"ex_score_list_"+str(top_k))
            exs = np.array(ex_scores)
            ex1.append(np.mean(exs[:, 0], axis=0))
            ex2.append(np.mean(exs[:, 1], axis=0))
            # ex3.append(np.mean(exs[:, 2], axis=0))
        if nbr_json=="none":
            ex1s_none.append(ex1)
            ex2s_none.append(ex2)
            ex3s_none.append(ex3)
        else:
            ex1s_prob.append(ex1)
            ex2s_prob.append(ex2)
            ex3s_prob.append(ex3)

ticks = ["shap_old", "lime_old", "suff_mb_false", "nece_mb_false"]
clfs = ["MLP", "SVM", "log_clf"]

for top_k in [1,2,3,4,5,6]:
    dump_path = input_data['data'] + "/" + "prob" + "/"
    df_1 = pd.DataFrame(data=ex2s_prob[top_k-1], columns=ticks)
    df_1.index = clfs
    ax = df_1.plot.bar()
    ax.set_title("Explanandum 2, with top "+str(top_k)+" features")
    plt.savefig(dump_path+"ex2_top_"+str(top_k)+".png")
    plt.clf()
# df_2 = pd.DataFrame(data=ex1s_none[1], columns=ticks)
# df_3 = pd.DataFrame(data=ex1s_none[2], columns=ticks)
# df_4 = pd.DataFrame(data=ex1s_none[3], columns=ticks)
print("ok")


# TODO check with use range =True explanandums
# TODO Finalise neighborhood for each thing - nece suff neigbhborhood none with mb,
# TODO density wise analysis of correlation and explananda
neighborhood_json_none = {"no_of_neighbours": 500, "probability": False, "bound": True,"use_range": False, "truly_random": True}
neighborhood_json_prob_range = {"no_of_neighbours": 500, "probability": True, "bound": True,"use_range": True, "truly_random": True}
neighborhood_json_prob = {"no_of_neighbours": 500, "probability": True, "bound": True,"use_range": False, "truly_random": True}
neighborhood_json_range = {"no_of_neighbours": 500, "probability": False, "bound": True,"use_range": True, "truly_random": True}

# nbr_json = "none"
nbr_dict = {
    "none":neighborhood_json_none,
    "prob":neighborhood_json_prob,
    "range":neighborhood_json_range,
    "prob_range":neighborhood_json_prob_range
}
# shap_nece = []
# shap_suff = []
# shap_mean = []
# shap_nece2 = []
# shap_suff2 = []
# shap_mean2 = []

less_correlated = [44, 6, 32, 35]
high_correlated = [4, 13, 56, 40]
for top_k in tqdm([5,6]):
    for nbr_json in ["none","prob"]:
        ex_score_list = []
        suff_nece_corr = []
        dump_path = input_data['data'] + "/" + nbr_json + "/" + input_data["classifier"] + "/"
        for ind in tqdm(range(len(testdf))):

            original_sample = testdf[data.features].iloc[ind].copy()
            output = clf.predict([original_sample])[0]
            nbr = context.generate_neighbourhood([], original_sample, data.features, 2000,False, True, False, True)
            dists = context.calculatel2(nbr[features],np.array(original_sample).reshape(1, -1))
            inds = np.argsort(dists[:,0])
            nbr = nbr.iloc[inds]
            nbr = nbr.iloc[:500]
            # outputs = clf.predict(nbr)
            # nbr = nbr[outputs!=output]

            # shap_lime = SHAP_LIME(clf, data, train_inds, input_data["classifier"], custom_neighborhood=nbr)
            lime_old = get_lime(original_sample,True,nbr)
            # lime_new = get_lime(original_sample,False,nbr)
            #
            # shap_new = shap_lime.get_shap_vals(original_sample)
            shap_old = shap_org.get_shap_vals(original_sample)
            #
            # shap_old = [abs(score) for score in shap_old]
            # shap_new = [abs(score) for score in shap_new]
            #
            # lime_old =  [abs(score) for score in lime_old]
            # lime_new =  [abs(score) for score in lime_new]

            suff_mb_false = nece_suff.sufficiency(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
                                                nbr_dict[nbr_json], use_metric="MB")
            nece_mb_false = nece_suff.necessity(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
                                              nbr_dict[nbr_json], use_metric="MB")
            # suff_euc_false = nece_suff.sufficiency(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                     neighborhood_json, use_metric="Euc")
            # nece_euc_false = nece_suff.necessity(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                   neighborhood_json, use_metric="Euc")
            # suff_euc_True = nece_suff.sufficiency(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                     neighborhood_json_prob, use_metric="Euc")
            # nece_euc_True = nece_suff.necessity(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                   neighborhood_json_prob, use_metric="Euc")
            # suff_true = nece_suff.sufficiency(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                     neighborhood_json_prob, use_metric="None")
            # nece_true = nece_suff.necessity(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                   neighborhood_json_prob, use_metric="None")
            # suff_false = nece_suff.sufficiency(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                     neighborhood_json, use_metric="None")
            # nece_false = nece_suff.necessity(original_sample, clf.predict([original_sample]), clf, traindf[data.features],
            #                                   neighborhood_json, use_metric="None")
            suff_nece_corr.append(measure_kendall_correlation(suff_mb_false, nece_mb_false))
            ex1_scores = []
            ex2_scores = []
            ex3_scores = []
            # for score in [suff_mb_false,suff_euc_false,suff_euc_True,suff_false,suff_true,nece_mb_false,nece_euc_false,
            #               nece_euc_True,nece_false,nece_true]: ex1_scores.append(ex1(original_sample, 5, False, score, output, clf))

            for score in [shap_old, lime_old, suff_mb_false, nece_mb_false]:
                ex1_scores.append(ex1(original_sample, top_k, score, output, clf))
                ex2_scores.append(ex2(original_sample, top_k, score, output, clf))
                ex3_scores.append(ex3(original_sample, top_k, score, output, clf))
            ex_score_list.append([ex1_scores, ex2_scores, ex3_scores])
        dump_path = input_data['data']+"/"+nbr_json+"/"+input_data["classifier"]+"/"
        joblib.dump(suff_nece_corr, dump_path+"suff_nece_corr_"+str(top_k))
        joblib.dump(ex_score_list, dump_path+"ex_score_list_"+str(top_k))

# done - log_clf, topk=1, none, diab
# done - log_clf, topk=1, prob, diab

# plt.bar(list(range(len(ex_score_list[0]))), np.mean(ex_score_list, axis=0))
# plt.xticks(rotation=45)
# plt.show()
# nece euc false for exp 1 with True
    # if ind in less_correlated:
    #     fig, ax = plt.subplots(2, 2)
    #     fig.suptitle("low correlated"+str(output))
    #     ax[0][0].bar(list(data.features), shap1[0], linewidth=2)
    #     ax[0][0].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     ax[0][0].set_title("Shap with new nbr")
    #     # ax[0][1].bar(list(data.features), shap2, linewidth=2)
    #     # ax[0][1].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     # ax[0][1].set_title("Shap with old nbr")
    #     ax[1][0].bar(list(data.features), suff_scores, linewidth=2)
    #     ax[1][0].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     ax[1][0].set_title("sufficiency")
    #     ax[1][1].bar(list(data.features), nece_scores, linewidth=2)
    #     ax[1][1].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     ax[1][1].set_title("necessity")
    #     plt.show()
    # elif ind in high_correlated:
    #     fig, ax = plt.subplots(2, 2)
    #     fig.suptitle("high correlated"+str(output))
    #     ax[0][0].bar(list(data.features), shap1[0], linewidth=2)
    #     ax[0][0].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     ax[0][0].set_title("Shap with new nbr")
    #     # ax[0][1].bar(list(data.features), shap2, linewidth=2)
    #     # ax[0][1].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     # ax[0][1].set_title("Shap with old nbr")
    #     ax[1][0].bar(list(data.features), suff_scores, linewidth=2)
    #     ax[1][0].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     ax[1][0].set_title("sufficiency")
    #     ax[1][1].bar(list(data.features), nece_scores, linewidth=2)
    #     ax[1][1].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
    #     ax[1][1].set_title("necessity")
    #     plt.show()
    #
    # suff_scores = suff_scores[-n:]
    # nece_scores = nece_scores[-n:]
    # shap_nece.append(measure_kendall_correlation(nece_scores, shap_scores1))
    # shap_suff.append(measure_kendall_correlation(suff_scores, shap_scores1))
    # mean_scores = [np.max([val1,val2]) for (val1,val2) in zip(suff_scores, nece_scores)]
    # shap_mean.append(measure_kendall_correlation(mean_scores, shap_scores1))
    # shap_nece2.append(measure_kendall_correlation(nece_scores, shap_scores2))
    # shap_suff2.append(measure_kendall_correlation(suff_scores, shap_scores2))
    # mean_scores = [np.max([val1,val2]) for (val1,val2) in zip(suff_scores, nece_scores)]
#     # shap_mean2.append(measure_kendall_correlation(mean_scores, shap_scores2))
# print("shap_suff: ", measure_kendall_correlation(np.mean(suffs, axis=0), np.mean(shaps, axis=0)))

# shap_nece, shap_suff, shap_mean, shap_nece2, shap_suff2, shap_mean2 = np.array(shap_nece), \
#                                                                       np.array(shap_suff), np.array(shap_mean), np.array(shap_nece2),\
#                                                                       np.array(shap_suff2), np.array(shap_mean2)
# print(" means (nece and suff)", np.mean(shap_nece[shap_nece > -1]), np.mean(shap_suff[shap_suff > -1]))
# print("shap_suff means (with and without nbr)",np.mean(shap_suff[shap_suff>-1]),np.mean(shap_suff2[shap_suff2>-1]))
# # print("shap_mean means (with and without nbr)",np.mean(shap_mean),np.mean(shap_mean2))
# print("shap_nece count < 0.25 (with and without nbr)",len(shap_nece[shap_nece<0.25]),len(shap_nece2[shap_nece2<0.25]))
# print("shap_suff count < 0.25 (with and without nbr)", len(shap_suff[shap_suff<0.25]),len(shap_suff2[shap_suff2<0.25]))
# print("shap_nece points < 0.25 (with and without nbr)",np.argwhere(shap_nece<0.25)[:,0],np.argwhere(shap_nece2<0.25)[:,0])
# print("shap_suff points < 0.25 (with and without nbr)", np.argwhere(shap_suff<0.25)[:,0],np.argwhere(shap_suff2<0.25)[:,0])
# print("count < 0.25 (nece and suff)", len(shap_nece[shap_nece < 0.25]), len(shap_suff[shap_suff < 0.25]))
# print("points < 0.25 (nece and suff)", np.argwhere(shap_nece < 0.25)[:, 0], np.argwhere(shap_suff < 0.25)[:, 0])
#     notice.append(ind)
#     density_low.append(density.get_density_score([original_sample]))
# elif shap_suff[-1]<=0.7:
#     density_mid.append(density.get_density_score([original_sample]))
# else:
#     density_high.append(density.get_density_score([original_sample]))


# fig,ax = plt.subplots(1,2)
# plt.savefig(path+'/suff3.png')
# ax[0].hist(shap_suff,linewidth=2)
# ax[1].hist(shap_nece,linewidth=2)
# ax[1][1].hist(density_mid,linewidth=2)
# ax[1][1].set_xticklabels(list(data.features), rotation=45, rotation_mode="anchor")
# plt.savefig(path+'/nece3.png')
# plt.show()
print("oj")

# Hyperparameters = no. of neighbors, size of neighbors
