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

pd.options.plotting.backend = "plotly"

input_data = { "data": "heart_db_old", "classifier": "MLP", "fold": "fold1", "mahalonobis_size": 2}



def validate_input():
    data_list = ["heart_db_old", "cerv", "diab_insulin","diab_without_insulin"]
    model_list = ["rf_clf", "log_clf", "SVM"]
    fold_list = ["fold"+str(i) for i in range(1,6)]
    if input_data["data"] not in data_list:
        print("Dataset not implemented, please choose data from ",", ".join(data_list))
    if input_data["classifier"] not in model_list:
        print("Model not implemented, please choose data from ",", ".join(model_list))
    if input_data["fold"] not in fold_list:
        print("invalid fold number, please choose among ",", ".join(fold_list))


# context_choice = input(
#         "Press 1 if you customised your context, 2 if you are using MB distance, or 3 is you are using euclidean distance")
from sklearn import svm
path = "analysis_outputs/"+input_data["data"]+"/"+input_data["fold"]
# clf = joblib.load(path+"/"+input_data["classifier"])
test_inds = joblib.load(path+"/test")
train_inds = joblib.load(path+"/train")
clf = svm.SVC(kernel='linear', probability=True)
# clf = MLPClassifier(random_state=1,max_iter=300)

def get_accuracy():
    return accuracy_score(clf.predict(data.df[data.features].iloc[test_inds][data.features]), data.df.iloc[test_inds][data.target])

def get_precision():
    return precision_score(clf.predict(data.df[data.features].iloc[test_inds][data.features]), data.df.iloc[test_inds][data.target])

def get_recall():
    return recall_score(clf.predict(data.df[data.features].iloc[test_inds][data.features]), data.df.iloc[test_inds][data.target])

def get_data(hot_encode):
    data = None
    if input_data["data"] == "heart_db_old":
        data = Data("Heart DB", hot_encode)
    elif input_data["data"] == "cerv":
        data = Data("Cervical DB", hot_encode)
    elif input_data["data"] == "diab_insulin" or input_data["data"] == "diab_without_insulin":
        data = Data("Diabetes DB", hot_encode, pick_without_insulin=True)
    return data

data:Data = get_data(False)

clf.fit(data.df[data.features].iloc[train_inds],data.df[data.target].iloc[train_inds])
# print(get_recall())

context = Neighborhood(data)

def visualise_context():
    dfs = context.generate_neighbourhood(None, data.df.iloc[0][data.features], data.features, no_of_neighbours=800,
                                         probability=False, bound=True, use_range=True, truly_random=True)
    fig, axs = plt.subplots(2, int(len(data.features)/2)+1)
    fig.suptitle('Vertically stacked subplots')
    feat_ind=0
    for j in range(2):
        for i in range(int(len(data.features)/2)+1):
            if feat_ind<len(data.features):
                axs[j][i].hist(dfs[data.features[feat_ind]])
                feat_ind+=1
    plt.show()




def compared_to_original(sample):
    means = data.df[data.features].iloc[train_inds].mean()
    df = data.df[data.features].iloc[train_inds]
    df = df.reset_index(drop=True)
    # for feat in data.features:
    #     if feat in data.binary:
    #         means[feat] = int(sample[feat]==0)
    #     elif feat in data.continuous:
    #         if means[feat]==sample[feat]: means[feat] = round(means[feat], data.dec_precisions[feat])
    #         else: means[feat] = round(means[feat]+10, data.dec_precisions[feat])
    #     else:
    #         means[feat] = data.feature_range[feat][np.argmin([abs(means[feat]-a) for a in data.feature_range[feat]])]
    neighbors = pd.DataFrame(
                np.repeat([means.values], 800, axis=0), columns=data.features)
    for feat in data.categorical:
        neighbors[feat] = np.random.choice(data.feature_range[feat], len(neighbors))
    for feat in data.continuous:
        neighbors[feat] = np.random.choice(data.df[feat],len(neighbors))
    for feat in data.binary:
        neighbors[feat] = np.random.choice([0,1], len(neighbors))

    num_features_to_vary = np.random.choice(list(range(len(data.features)-2)),800)
    for k,n in enumerate(num_features_to_vary):
        selected_features = np.random.choice(data.features, n, replace=True)
        for feat in selected_features:
            neighbors.at[k, feat] = sample[feat]
    FI = []
    output = clf.predict([sample])[0]
    for feat in data.features:
        sample_probs = clf.predict_proba(neighbors[neighbors[feat]==sample[feat]])[:,output]
        ref_probs = clf.predict_proba(neighbors[neighbors[feat]!=sample[feat]])[:,output]
        FI.append(round(abs(np.mean(sample_probs)-np.mean(ref_probs)),3))
    return FI

def euclidean_neighbors(sample):
    neighbors = pd.DataFrame(
        np.repeat([sample.values], 1000, axis=0), columns=data.features)
    for feat in data.categorical:
        neighbors[feat] = np.random.choice(data.feature_range[feat], len(neighbors))
    for feat in data.continuous:
        neighbors[feat] = np.random.choice(data.df[feat], len(neighbors))
    for feat in data.binary:
        neighbors[feat] = np.random.choice([0, 1], len(neighbors))
    num_features_to_vary = np.random.choice(list(range(len(data.features) - 2)), 1000)
    for k,n in enumerate(num_features_to_vary):
        selected_features = np.random.choice(data.features, n, replace=True)
        for feat in selected_features:
            neighbors.at[k, feat] = sample[feat]
    dists = [context.calculatel2(neighbors[data.features].iloc[i], sample) for i
             in range(len(neighbors))]
    inds = np.argsort(dists)
    neighbors = neighbors.iloc[inds]
    neighbors = neighbors[:200]
    outputs = clf.predict(neighbors)
    class_0 = neighbors.iloc[outputs==0]
    class_1 = neighbors.iloc[outputs==1]
    fi = []
    mini_length_for_cov = min([len(class_1),len(class_0)])
    for feat in data.features:
        fis = abs(np.mean(class_0[feat])-np.mean(class_1[feat]))
        fi.append(fis)
    return fi


def measure_kendall_correlation(ranking1, ranking2):
    kendal = stats.kendalltau(ranking1, ranking2)
    return kendal.correlation

def generate_csv(scores, clf, indices, histname):
    class_0 = list(clf.predict_proba(data.df[data.features].iloc[indices])[:,0])
    class_1 = list(clf.predict_proba(data.df[data.features].iloc[indices])[:,1])
    truths = list(data.df[data.target].iloc[indices])
    density = Density(data)
    clusters = list(density.get_cluster(data.df[data.features].iloc[indices]))
    densities = list(density.get_density_score(data.df[data.features].iloc[indices]))
    entries = np.array([indices, class_0, class_1,truths,densities,clusters,scores[0],scores[1],scores[2],scores[3]])
    df = pd.DataFrame(columns = ["indices","class_0","class_1","truth","density","cluster","kendal_shap_suff","kendal_lime_suff", "kendal_shap_nece","kendal_lime_nece"],
                      data = entries.transpose())
    df.to_csv(path+"/results_mb_big/results_corr"+histname+".csv",index=False)

traindf = data.df.iloc[train_inds]
'''plot_histograms(traindf, path, 0)
with open(path+"/means.txt", "w") as f:
    for feat in data.features:
        if feat in data.continuous:
            m = round(np.mean(traindf[feat]), data.dec_precisions[feat])
            f.write(feat+": "+str(m)+"\n")
        else:
            for cat in data.feature_range[feat]:
                f.write(feat+", category: "+ str(cat) + ": "+str(len(traindf[traindf[feat]==cat])))
                f.write("\n")'''
wf = ['Age', 'Sex', 'Typical_Angina', 'Atypical_Angina',
      'Resting_Blood_Pressure', 'Fasting_Blood_Sugar',
      'Rest_ECG','Colestrol','Asymptomatic_Angina',
      'Non_Anginal_Pain','Slope','ST_Depression',
      'Exercised_Induced_Angina','mhr_exceeded','Major_Vessels','Thalessemia']
mean_vals = {'Age':54, 'Sex':1, 'Typical_Angina':0, 'Atypical_Angina':0,
      'Resting_Blood_Pressure':131, 'Fasting_Blood_Sugar':0,
      'Rest_ECG':0,'Colestrol':247,'Non_Anginal_Pain':0,'Slope':1,'ST_Depression':1.0,'Asymptomatic_Angina':1,
      'Exercised_Induced_Angina':0,'mhr_exceeded':0,'Major_Vessels':0,'Thalessemia':3}
testdf = data.df.iloc[test_inds]
# for ind in range(len(test_inds)):
#     density = Density(data)
#     testdf = data.df.iloc[test_inds]
#     original_sample = testdf[data.features].iloc[ind].copy()
#     # print("density of sample: ", density.get_density_score([original_sample]))
#     sample = original_sample.copy()
#     for feat in wf[4:]:
#         sample[feat] = mean_vals[feat]
#     # print("densty of sample with mean values : ", density.get_density_score([sample]))
#     # print("classification probabilities with all mean values are: ",clf.predict_proba([sample]))
#     classes = clf.predict([sample])[0]
#     for feat in wf[4:]:
#         sample[feat] = original_sample[feat]
#         newclass = clf.predict([sample])[0]
#         if newclass!=classes:
#             print(feat, ind)

        # print("densty of sample with mean values: ", density.get_density_score([sample]))
        # print("classification probabilities with actual feature: ", feat, "are: ", clf.predict_proba([sample]))
nece_suff = Nece_Suff(context)
# shap_lime = SHAP_LIME(clf, data, train_inds, input_data["classifier"], custom_neighborhood = traindf[traindf[data.target]==0])
# shap1 = shap_lime.get_shap_vals([original_sample])
#
# plt.clf()
# plt.bar(list(data.features),shap1,linewidth=2)
# plt.xticks(rotation=45)
# plt.savefig(path+'/shap_0.png')
#
# shap_lime = SHAP_LIME(clf, data, train_inds, input_data["classifier"], custom_neighborhood = traindf[traindf[data.target]==1])
# shap2 = shap_lime.get_shap_vals([original_sample])
# plt.clf()
# plt.bar(list(data.features),shap2,linewidth=2)
# plt.xticks(rotation=45)
# plt.savefig(path+'/shap_1.png')
shap_lime_org = SHAP_LIME(clf, data, train_inds, input_data["classifier"], custom_neighborhood = traindf)
neighborhood_json = {"no_of_neighbours":500,
                     "probability":False,
                     "bound":True,
                     "use_range":True, "truly_random":True}
shap_nece = []
shap_suff = []
shap_mean = []
shap_nece2 = []
shap_suff2 = []
shap_mean2 = []
shaps = []
limes = []
necess = []
suffs = []
density = Density(data)
less_correlated = [44,6,32,35]
high_correlated = [4,13,56,40]
for ind in range(len(testdf)):
    original_sample = testdf[data.features].iloc[ind].copy()
    output = clf.predict([original_sample])[0]
    nbr = context.generate_neighbourhood(None, original_sample,data.features, neighborhood_json["no_of_neighbours"],
                                         neighborhood_json["probability"], neighborhood_json["bound"],
                                                                           neighborhood_json["use_range"],
                                                                           neighborhood_json["truly_random"])
    #
    # outputs = clf.predict(nbr)
    # nbr = nbr[outputs!=output]

    # shap_lime = SHAP_LIME(clf, data, train_inds, input_data["classifier"], custom_neighborhood=nbr)
    lime_log = LogisticRegression()
    lime_log.fit(nbr,clf.predict(nbr))
    lime_fi = lime_log.coef_
    # shap1 = shap_lime.get_shap_vals(original_sample)
    shap2 = shap_lime_org.get_shap_vals(original_sample)
    # shap_scores1 = [abs(score) for score in shap1]
    lime_scores = [abs(score) for score in lime_fi]
    limes.append(lime_scores)
    shap_scores2 = [abs(score) for score in shap2]
    shaps.append(shap_scores2)
    suff_scores = nece_suff.sufficiency(original_sample, clf.predict([original_sample]), clf, traindf[data.features], neighborhood_json,use_metric="MB")
    suffs.append(suff_scores)
    nece_scores = nece_suff.necessity(original_sample, clf.predict([original_sample]), clf, traindf[data.features], neighborhood_json,use_metric="MB")
    necess.append(nece_scores)
    # shap_scores1 = shap_scores1[-n:]
    # shap_scores2 = shap_scores2[-n:]

    # plt.bar(list(data.features), shap1, linewidth=2)
    # # plt.xticks(rotation=45)
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
print("shap_suff: ",measure_kendall_correlation(np.mean(suffs,axis=0), np.mean(shaps,axis=0)))

# shap_nece, shap_suff, shap_mean, shap_nece2, shap_suff2, shap_mean2 = np.array(shap_nece), \
#                                                                       np.array(shap_suff), np.array(shap_mean), np.array(shap_nece2),\
#                                                                       np.array(shap_suff2), np.array(shap_mean2)
print(" means (nece and suff)",np.mean(shap_nece[shap_nece>-1]),np.mean(shap_suff[shap_suff>-1]))
# print("shap_suff means (with and without nbr)",np.mean(shap_suff[shap_suff>-1]),np.mean(shap_suff2[shap_suff2>-1]))
# # print("shap_mean means (with and without nbr)",np.mean(shap_mean),np.mean(shap_mean2))
# print("shap_nece count < 0.25 (with and without nbr)",len(shap_nece[shap_nece<0.25]),len(shap_nece2[shap_nece2<0.25]))
# print("shap_suff count < 0.25 (with and without nbr)", len(shap_suff[shap_suff<0.25]),len(shap_suff2[shap_suff2<0.25]))
# print("shap_nece points < 0.25 (with and without nbr)",np.argwhere(shap_nece<0.25)[:,0],np.argwhere(shap_nece2<0.25)[:,0])
# print("shap_suff points < 0.25 (with and without nbr)", np.argwhere(shap_suff<0.25)[:,0],np.argwhere(shap_suff2<0.25)[:,0])
print("count < 0.25 (nece and suff)",len(shap_nece[shap_nece<0.25]),len(shap_suff[shap_suff<0.25]))
print("points < 0.25 (nece and suff)",np.argwhere(shap_nece<0.25)[:,0],np.argwhere(shap_suff<0.25)[:,0])
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
# log clf - 4,56 , nece -4 40 57 (high)
# log clf low 35 43 44 nece - 16 17 32 35 43 48
# less corrrelated points [35, 43, 44, 8 15,  16,17,26] nece 15,26
# high correlated points 4,13,56 , nece -40,41,57
# can you make them SHAP's neighborhood behave like suff/nece?
# can u use this information for classifier selection
# [90,189]
# you need to change feature range
# no_of_neighbours = [200,500,800]
# probability = [True, False]
# use_range = [True, False]
# truly_random = [True, False]
# neighborhood_jsons = []
# for size in no_of_neighbours:
#     for prob in probability:
#         for use_rang in use_range:
#             for randomness in truly_random:
#                 neighborhood_json = {"no_of_neighbours":size, "probability":prob, "bound":True, "use_range":use_rang, "truly_random":randomness}
#                 neighborhood_jsons.append(neighborhood_json)
# hist_name = 0
# for json in neighborhood_jsons:
#     indices = []
#     scores = []
#     kendals_shap_nece = []
#     kendals_shap_suff = []
#     kendals_lime_nece = []
#     kendals_lime_suff = []
#     kendals_cf_nece = []
#     kendals_cf_suff = []
#     for i in tqdm(test_inds, desc="Indices"):
#         indices.append(i)
#         sample = data.df[data.features].iloc[i]
#         # custom_neighbors = context.generate_neighbourhood(None, sample, data.features, 800,
#         #                                                   probability=True, bound=False, use_range=True, truly_random=True)
#         # custom_fi, percent_original, major_class = context.get_importance(clf, sample,
#         #                                                                   neighbors=custom_neighbors, weighted=True)
#         # fi_scores = []
#         # for feat_scores in custom_fi:
#         #     scores = [abs(score) for score in feat_scores if score>=-1]
#         #     fi_scores.append(np.mean(scores))
#         suff_scores = nece_suff.sufficiency(sample, clf.predict([data.df[data.features].iloc[i]]), clf,
#                                             data.df.iloc[train_inds],
#                                             json, use_metric="MB")
#         nece_scores = nece_suff.necessity(sample, clf.predict([data.df[data.features].iloc[i]]), clf,
#                                           data.df.iloc[train_inds],
#                                           json, use_metric="MB")
#
#         shap_scores = shap_lime.get_shap_vals(data.df[data.features].iloc[i])
#         shap_scores = [abs(score) for score in shap_scores]
#         lime_scores = shap_lime.get_lime_values(sample)
#         lime_scores = [abs(score) for score in lime_scores]
#         kendals_shap_suff.append(measure_kendall_correlation(suff_scores, shap_scores))
#         kendals_shap_nece.append(measure_kendall_correlation(nece_scores, shap_scores))
#         kendals_lime_nece.append(measure_kendall_correlation(nece_scores, lime_scores))
#         kendals_lime_suff.append(measure_kendall_correlation(suff_scores, lime_scores))
#     scores = [kendals_shap_suff,kendals_lime_suff,kendals_shap_nece,kendals_lime_nece]
#     generate_csv(scores, clf, indices, str(hist_name))
#     hist_name+=1

#following is the code for reading correlation csvs - BEG 1
# for ind in range(10):
# ind = 0
# df = pd.read_csv(path+"/results_corrs_csvs/results_corr"+str(ind)+".csv")
# print("ok")
# plot_histograms(df[df["kendal_shap_suff"]<0],path+"/results_corrs_csvs", ind)
# df[df["kendal_shap_suff"]<0]['indices']
#following is the code for reading correlation csvs - END 1

# for CF imp you need a different file
# for density and cluster (both types) you need a different methodology
# cf_imp =
# confidence = get_confidence()
# def generate_plots()
# run the program
# validate_input()

    # kendals_same.append(measure_kendall_correlation(shap_scores,fi_shap))
    # kendals_diff.append(measure_kendall_correlation(shap_scores,fi_scores))

    # generic_neighborhood = context.generate_neighbourhood(None, sample, data.features, 800,
    #                                                   probability=False, bound=True, use_range=False, truly_random=True)
    # custom_fi_new, percent_original, major_class = context.get_importance_gen(clf, sample,
    #                                                                   neighbors=generic_neighborhood, weighted=True)
    # fi_shap = compard_to_original(sample)
    # fi_lime = euclidean_neighbors(sample)


# def compared_to_reference():
#
# def local_compared_to_generic():
#
# def local_compared_to_reference():
#
'''
results:
heart_db_old db:
    with random neighbors + replacement with sample vs custom FI
    log_clf = 58 vs 32
    rf_clf = 62 vs 66
    svm: 0.4 vs 0.2
    mlp: 48 vs 32
diab:
    rf_clf = 77 vs 0.49
    log_clf = 62 vs 50
    svm = 0.55 vs 0.61
    mlp: 60 vs 0.41
cer:
    svm: 0.38 , 0.43
    MLP: 0.06
    
    condition on density = 25, for kendal_shap_suff, shap_nece, lime_nece, lime_suff
    suff and shap has difference w.r.t class probabilities 
'''

# if looking at a workflow from a start point, all the pending features to be taken -> show global
# we enter age =1, female =1, <- neighborhood size decreases, we can show a preliminary feature importance by showing decrease/increase of age value and female vs male
#