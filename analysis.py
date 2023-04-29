from fsets import f_sets
import numpy as np
import pandas as pd
from conf import Conf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from local_gen import Local
import random
from scipy.spatial import distance
from plot import bar_plot,bee_swarm


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

data_name = "Heart"
model_name = "RF"
obj = Conf(data_name, model_name, "all", False)

features = obj.data.features

# def save_models(path):
#     from sklearn.model_selection import KFold
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.ensemble import RandomForestClassifier
#     import joblib
#
#     kfold = KFold(5,shuffle=True,random_state=1)
#     fold=1
#     for train, test in kfold.split(obj.data.df):
#         current = path+"/fold"+str(fold)
#         mkdir_p(current)
#         log_clf = LogisticRegression(max_iter=1000)
#         log_clf.fit(obj.data.df[obj.feats].iloc[train], obj.data.df[obj.data.target].iloc[train])
#         joblib.dump(log_clf, current+'/log_clf')
#         rf_clf = RandomForestClassifier(bootstrap=True, class_weight=None,
#                                          criterion='gini', max_depth=3, max_features=2,
#                                          max_leaf_nodes=None, max_samples=None,
#                                          min_impurity_decrease=0.0,
#                                          min_samples_leaf=5, min_samples_split=2,
#                                          min_weight_fraction_leaf=0.0, n_estimators=100,
#                                          n_jobs=-1, oob_score=False, random_state=42, verbose=0,
#                                          warm_start=False)
#         rf_clf.fit(obj.data.df[obj.feats].iloc[train], obj.data.df[obj.data.target].iloc[train])
#         joblib.dump(rf_clf, current + '/rf_clf')
#         joblib.dump(train, current + "/train")
#         joblib.dump(test, current + "/test")
#         fold+=1
#
# path = "analysis_outputs/heart_db_old"
# mkdir_p(path)
# save_models(path)



def get_norm(arr):
  std_scaler = StandardScaler()
  return std_scaler.fit_transform(np.array(arr).reshape(-1, 1))

def get_norm_(arr):
  min_max_scaler = MinMaxScaler()
  return min_max_scaler.fit_transform(np.array(arr).reshape(-1, 1))

def is_pos_def(A):
    if np.allclose(A, A.T,0.1,0.1):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

from scipy.spatial.distance import cdist
def calculateMahalanobis(y=None, data=None, cov=None):
    x = cdist(y, data, 'mahalanobis', VI=cov)
    return x
    # X = np.vstack([y, yy])
    # V = np.cov(X.T)
    # p = np.linalg.inv(V)
    # D = np.sqrt(np.sum(np.dot(e, p) * e, axis=1))

def get_continuous_samples(low, high, precision, size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if precision == 0:
        result = np.random.randint(low, high + 1, size).tolist()
        result = [float(r) for r in result]
    else:
        result = np.random.uniform(low, high + (10 ** -precision), size)
        result = [round(r, precision) for r in result]
    return result


def generate_neighbourhood(feature_to_freeze, sample, no_of_neighbours=200):
    features_to_change = obj.feats.copy()
    if feature_to_freeze:
        features_to_change.remove(feature_to_freeze)
    random_instances = get_neighbourhood_samples(
        feature_to_freeze,
        obj.ranges, 2, no_of_neighbours, sample)
    neighbors = pd.DataFrame(
        np.repeat(sample.values, no_of_neighbours, axis=0), columns=sample.columns)
    # Loop to change one feature at a time, then two features, and so on.
    for num_features_to_vary in range(1, len(features_to_change) + 1):
        selected_features = np.random.choice(features_to_change, (no_of_neighbours, 1), replace=True)
        for k in range(no_of_neighbours):
            neighbors.at[k, selected_features[k][0]] = random_instances.at[k, selected_features[k][0]]
    return neighbors

def get_neighbourhood_samples(fixed_features_values, feature_range, sampling_random_seed, sampling_size, original_sample):
    precisions = obj.dec_precisions
    categorical_features_frequencies = {}

    for feature in obj.data.categorical:
        categorical_features_frequencies[feature] = len(obj.data.train_df[feature].value_counts())

    if sampling_random_seed is not None:
        random.seed(sampling_random_seed)

    samples = []
    for feature in obj.feats:
        if fixed_features_values and feature in fixed_features_values:
            sample = [original_sample[feature]] * sampling_size
        elif feature in obj.data.continuous:
            # low = feature_range[feature][0]
            # high = feature_range[feature][1]
            low = original_sample[feature] - 5*(obj.changes_allowed[feature])
            high = original_sample[feature] + 5*(obj.changes_allowed[feature])
            low = low if low.values[0]>feature_range[feature][0] else feature_range[feature][0]
            sample = obj.get_continuous_samples(
                low, high, precisions[feature], size=sampling_size,
                seed=sampling_random_seed)
        else:
            if sampling_random_seed is not None:
                random.seed(sampling_random_seed)
            sample = random.choices(feature_range[feature], k=sampling_size)

        samples.append(sample)
    samples = pd.DataFrame(dict(zip(obj.feats, samples)))
    return samples

def get_clusters(ind):
    clusters = []
    for feat in obj.cluster_precedence[ind]:
        if feat in obj.data.continuous:
            mini = min(obj.data.df[feat])
            maxi = max(obj.data.df[feat])
            divides = int((maxi-mini)/3)
            if len(clusters)!=0:
                originals = clusters.copy()
                clusters = []
                for cluster in originals:
                    clusters.append(cluster.loc[(obj.data.df[feat] <= mini+divides) & (obj.data.df[feat] >= mini)])
                    clusters.append(cluster.loc[(obj.data.df[feat] <= maxi-divides) & (obj.data.df[feat] >= mini + divides)])
                    clusters.append(cluster.loc[(obj.data.df[feat] <= maxi) & (obj.data.df[feat] >= maxi-divides)])
            else:
                clusters.append(obj.data.df.loc[(obj.data.df[feat] <= mini + divides) & (obj.data.df[feat] >= mini)])
                clusters.append(obj.data.df.loc[(obj.data.df[feat] <= maxi - divides) & (obj.data.df[feat] >= mini + divides)])
                clusters.append(obj.data.df.loc[(obj.data.df[feat] <= maxi) & (obj.data.df[feat] >= maxi - divides)])
        else:
            cats = set(obj.data.df[feat])

            if len(clusters)!=0:
                originals = clusters.copy()
                clusters = []
                for ind,cluster in enumerate(originals):
                    for cat in cats:
                        clusters.append(cluster.loc[obj.data.df[feat] == cat])
            else:
                for cat in cats:
                    clusters.append(obj.data.df.loc[obj.data.df[feat] ==cat])
    return clusters



# clusters = get_clusters(1)

# cluster analysis
# for j in range(len(clusters)):
#     plt.clf()
#     plt.xlabel("Size of neighborhood")
#     plt.ylabel("Percentage of same classification")
#     percents = []
#     for i in range(len(clusters[j])):
#         instance = clusters[j][obj.feats].iloc[i]
#         instance_df = pd.DataFrame(columns=features)
#         instance_df = instance_df.append(instance)
#         output = obj.get_probability(instance_df)[0]
#         output_ind = np.argmax(output)
#         local = Local()
#         neighbors = generate_neighbourhood(None, instance_df, 800)
#         dists = calculateMahalanobis(neighbors[obj.feats], instance_df, np.cov(obj.data.df[obj.feats].values))
#         d = np.array(dists[:, 0])
#         inds = np.argsort(dists[:, 0])
#         neighbors = neighbors.iloc[inds]
#         percents=[]
#         sizes = [50,100,150,200,250,300,400,500,600]
#         for size in sizes:
#             neighbors_ = neighbors[:size]
#             outputs = obj.get_probability(neighbors_)
#             outputs = np.array(outputs[:, output_ind])
#             x = len(outputs[outputs > output[output_ind]])
#             if x!=0:
#                 integral = sum(outputs[outputs > output[output_ind]])/len(outputs)
#                 percents.append(integral)
#             else:
#                 percents.append(x)
#         plt.plot(sizes, percents)
#
#     file_path = "cluster_neighbors_new/"
#     mkdir_p(file_path)
#     plt.savefig(file_path+"/conf"+str(j)+'.png')


# for i in range(len(obj.data.test_df)):
#     instance = obj.data.test_df[obj.feats].iloc[i]
#     instance_df = pd.DataFrame(columns=features)
#     instance_df = instance_df.append(instance)
#     output = obj.get_output(instance_df)
#     dists = calculateMahalanobis(obj.data.df[obj.feats], instance_df, np.cov(obj.data.train_df[obj.feats].values))
#     inds = np.argsort(dists[:, 0])
#     neighbors = obj.data.df[obj.feats].iloc[inds]
#     percents = []
#     sizes = [10,20,30,50,80,100,120,150,200]
#     for size in sizes:
#         neighbors_ = neighbors[:size]
#         outputs = obj.get_output(neighbors_)
#         percent = len(outputs[outputs==output[0]])/len(neighbors_)
#         percents.append(percent)
#     plt.plot(sizes, percents)
#
#     file_path = "cluster_neighbor/"
#     mkdir_p(file_path)
#     plt.savefig(file_path+"/conf"+str(i)+'.png')


# for i in range(len(obj.data.test_df)):
#     instance = obj.data.test_df[obj.feats].iloc[i]
#     instance_df = pd.DataFrame(columns=features)
#     instance_df = instance_df.append(instance)
#     output = obj.get_output(instance_df)
#     neighbors = generate_neighbourhood(None, instance_df, 1000)
#     dists = calculateMahalanobis(neighbors[obj.feats], instance_df, np.cov(obj.data.train_df[obj.feats].values))
#     d = np.array(dists[:, 0])
#     percents = []
#     # sizes = [100,200,300,400,500,600,700,800,900,1000]
#     sizes = [50, 60, 70, 80, 90, 100,120]
#     for size in sizes:
#         neighbors_ = neighbors[d<size]
#         outputs = obj.get_output(neighbors_)
#         percent = len(outputs[outputs==output[0]])/len(neighbors_)
#         percents.append(percent)
#     plt.plot(sizes, percents)
#
#     file_path = "cluster_neighbor/"
#     mkdir_p(file_path)
#     plt.savefig(file_path+"/c"+str(i)+'.png')

# def get_shap_lime_df(path):
#     import joblib
#     import shap
#     import lime
#     df = None
#     for fold in range(5):
#         fold_path = "fold"+str(fold+1)
#         log_clf = joblib.load(path+fold_path+"/log_clf")
#         rf_clf = joblib.load(path+fold_path+"/rf_clf")
#         train_inds = joblib.load(path+fold_path+"/train")
#         # test_inds = joblib.load(path+fold_path+"/test")
#         log_coefs = log_clf.coef_[0].reshape(1,-1)
#         # display(pd.DataFrame(columns=list(x.columns), index=["Pred","Coef"],data = np.array([a[0],b[0]])))
#         rf_fi = rf_clf.feature_importances_
#         df = obj.data.df
#         shaps_log = []
#         limes_log = []
#         shaps_rf = []
#         limes_rf = []
#         top_preds_arr = []
#         # actual_output_log = log_clf.predict(df[obj.feats])
#         # actual_output_rf = rf_clf.predict(df[obj.feats])
#         for i in range(len(df)):
#             instance = df[obj.feats].iloc[i]
#             instance_df = pd.DataFrame(columns=features)
#             instance_df = instance_df.append(instance)
#
#             # logistic
#             log_shap = shap.Explainer(log_clf, df[obj.feats].iloc[train_inds])
#             shaps_log.append(list(log_shap(instance_df).values[0]))
#             log_lime = lime.lime_tabular.LimeTabularExplainer(df[obj.feats].iloc[train_inds].values[:, :], feature_names=obj.feats,
#                                                           class_names=obj.data.target)
#             lime_vals = log_lime.explain_instance(np.array(instance_df)[0], log_clf.predict_proba,
#                                                  num_features=len(obj.feats))
#             scores=[]
#             # for cerv
#             for expln in lime_vals.as_list():
#                 scores.append(abs(expln[1]))
#             # for heart_db_old
#             # for expln in lime_vals.as_list():
#             #     for f in obj.feats:
#             #         if f in expln[0]:
#             #             scores.append(abs(expln[1]))
#             limes_log.append(scores)
#             top_preds = log_coefs * instance_df.iloc[0].values.reshape(1, -1)
#             top_preds_arr.append(list(top_preds[0]))
#
#             # rf
#             rf_shap = shap.TreeExplainer(rf_clf)
#             shaps_rf.append(list(rf_shap(np.array(instance_df).reshape(1,-1)).values[:,:,1][0]))
#             rf_lime = lime.lime_tabular.LimeTabularExplainer(df[obj.feats].iloc[train_inds].values[:, :], feature_names=obj.feats,
#                                                           class_names=obj.data.target)
#             lime_vals = rf_lime.explain_instance(np.array(instance_df)[0], rf_clf.predict_proba,
#                                                  num_features=len(obj.feats))
#             scores = []
#             # for cerv
#             for expln in lime_vals.as_list():
#                 scores.append(abs(expln[1]))
#             # for heart_db_old
#             # for expln in lime_vals.as_list():
#             #     for f in obj.feats:
#             #         if f in expln[0]:
#             #             scores.append(abs(expln[1]))
#             limes_rf.append(scores)
#             bar_plot([shaps_log[-1], shaps_rf[-1], limes_rf[-1], limes_log[-1], top_preds_arr[-1]], ["SUFF", "NECE", "SHAP", "LIME", "Conf"],
#                  obj.feats, (20, 10), path+fold_path+"/shap_lime")
#     print("ok")

# path = "analysis_outputs/cerv/"
# # mkdir_p(path)
# get_shap_lime_df(path)


def necessity(sample, output, model, train_df):
    scores = []
    for feat in obj.feats:
        neighbors = generate_neighbourhood(feat,sample,800)
        preds = model.predict(neighbors)
        # check that the output is same as "output"
        neighbors = neighbors.iloc[preds==output]
        # filter out ngihbours with same prediction
        dists = calculateMahalanobis(neighbors[obj.feats], sample, np.cov(train_df[obj.feats].values))
        inds = np.argsort(dists[:, 0])
        neighbors = neighbors.iloc[inds]
        neighbors = neighbors.iloc[:100]
        if feat not in obj.data.continuous:
            select = list(obj.ranges[feat])
            if sample[feat].values[0] in select:
                select.remove(sample[feat].values[0])
            neighbors[feat] = np.random.choice(select,len(neighbors))
        elif obj.dec_precisions[feat]==0:
            select = list(range(int(obj.ranges[feat][0]), int(obj.ranges[feat][1])))
            if sample[feat].values[0] in select:
                select.remove(sample[feat].values[0])
            neighbors[feat] = np.random.choice(select, len(neighbors))
        else:
            select = list(np.random.uniform(obj.ranges[feat][0], obj.ranges[feat][1], 2*len(neighbors)))
            select = [round(r, obj.dec_precisions[feat]) for r in select]
            if sample[feat].values[0] in select:
                select.remove(sample[feat].values[0])
            select = select[:len(neighbors)]
            neighbors[feat] = select

        preds = model.predict(neighbors)
        score = sum((preds!=output).astype(int))/len(neighbors)
        scores.append(round(score,2))
    return scores

def sufficiency(sample, output, model, train_df):
    scores = []
    for feat in obj.feats:
        neighbors = generate_neighbourhood(None,sample,800)
        neighbors = neighbors[neighbors[feat]!=sample[feat].values[0]]
        preds = model.predict(neighbors)
        # filter out neighbours with different preds
        neighbors = neighbors.iloc[preds != output]
        dists = calculateMahalanobis(neighbors[obj.feats], sample, np.cov(train_df[obj.feats].values))
        inds = np.argsort(dists[:, 0])
        neighbors = neighbors.iloc[inds]
        neighbors = neighbors.iloc[:100]
        if len(neighbors)>0:
            neighbors[feat] = [sample[feat].iloc[0]] * len(neighbors)

            preds = model.predict(neighbors)
            score = sum((preds==output).astype(int))/len(neighbors)
        else: score = 0
        scores.append(round(score,2))
    return scores

from tqdm import tqdm
def get_suff_nece(path):
    import joblib
    df = None
    for fold in tqdm(range(5)):
        fold_path = "fold"+str(fold+1)
        log_clf = joblib.load(path+fold_path+"/log_clf")
        rf_clf = joblib.load(path+fold_path+"/rf_clf")
        train_inds = joblib.load(path+fold_path+"/train")
        # test_inds = joblib.load(path+fold_path+"/test")
        df = obj.data.df
        nece_log = []
        suff_log = []
        nece_rf = []
        suff_rf = []
        for i in tqdm(range(len(df.iloc[191:]))):
            instance = df[obj.feats].iloc[i+191]
            instance_df = pd.DataFrame(columns=features)
            instance_df = instance_df.append(instance)

            # logistic
            nece_log.append(necessity(instance_df,log_clf.predict(instance_df)[0], log_clf, df.iloc[train_inds]))
            suff_log.append(sufficiency(instance_df, log_clf.predict(instance_df)[0], log_clf, df.iloc[train_inds]))
            # rf
            nece_rf.append(necessity(instance_df, rf_clf.predict(instance_df)[0], rf_clf, df.iloc[train_inds]))
            suff_rf.append(sufficiency(instance_df, rf_clf.predict(instance_df)[0], rf_clf, df.iloc[train_inds]))
            bar_plot([ suff_log[-1],nece_log[-1],suff_rf[-1],nece_rf[-1]], ["SUFF_log", "NECE_log", "SUFF_rf", "NECE_rf"],
                 obj.feats, (20, 10), path+fold_path+"/suff_nece")
    print("ok")

# get_suff_nece("analysis_outputs/heart_db_old/")

def get_confidence(path):
    import joblib
    df = None
    for fold in tqdm(range(5)):
        fold_path = "fold"+str(fold+1)
        log_clf = joblib.load(path+fold_path+"/log_clf")
        rf_clf = joblib.load(path+fold_path+"/rf_clf")
        train_inds = joblib.load(path+fold_path+"/train")
        # test_inds = joblib.load(path+fold_path+"/test")
        df = obj.data.df
        cred_log = []
        cred_fi_log_favor = []
        cred_fi_log_against = []
        cred_rf = []
        cred_fi_rf = []
        log_output = log_clf.predict(df[obj.feats])
        for i in tqdm(range(len(df))):
            instance = df[obj.feats].iloc[i]
            instance_df = pd.DataFrame(columns=features)
            instance_df = instance_df.append(instance)

            # logistic
            neighbors = generate_neighbourhood(None, instance_df, 800)
            dists = calculateMahalanobis(neighbors[obj.feats], instance_df, np.cov(df.iloc[train_inds][obj.feats].values))
            neighbors = neighbors.iloc[dists<1000]
            outputs = log_clf.predict(neighbors)
            percent = len(outputs[outputs==log_output[i]])/len(neighbors)
            cred_log.append(percent)
            favor = neighbors[outputs==log_output[i]]
            against = neighbors[outputs!=log_output[i]]
            fi_favor_arr = []
            fi_against_arr = []
            for feat in features:
                filtered_favor = favor[favor[feat] != instance[feat]]
                same_in_favor = len(favor)-len(filtered_favor)
                filtered_against = against[against[feat]!=instance[feat]]
                same_in_against = len(against) - len(filtered_against)
                if feat in obj.data.continuous:
                    distances_favor = abs(filtered_favor[feat] - instance[feat])
                    distances_against = abs(filtered_favor[feat] - instance[feat])
                    inv_favor = list(1/distances_favor)
                    inv_favor.extend([1 for i in range(same_in_favor)])
                    inv_against = list(1/distances_against)
                    inv_against.extend([0 for i in range(same_in_against)])
                    fi_favor = np.mean(inv_favor)
                    fi_against = np.mean(inv_against)
                else:
                    if len(favor)!=0:
                        fi_favor = same_in_favor/len(favor)
                    else: fi_favor = 0
                    if len(against)!=0:
                        fi_against = len(filtered_against)/len(against)
                    else: fi_against = 0
                fi_favor_arr.append(fi_favor)
                fi_against_arr.append(fi_against)
            cred_fi_log_against.append(fi_against_arr)
            cred_fi_log_favor.append(fi_favor_arr)
            bar_plot([cred_fi_log_against[-1],cred_fi_log_favor[-1]], ["Log_favor_FI", "Log_Against_FI"],
                 obj.feats, (20, 10), path+fold_path+"/log_fi")
        joblib.dump(cred_log,path+fold_path+"/cred_scores_log")
        # joblib.dump(cred_rf, path + fold_path + "/cred_scores_rf")
    print("ok")

# get_confidence("analysis_outputs/heart_db_old/")
from matplotlib.pyplot import figure
figure(figsize=(20, 15), dpi=120)
def plot_results(inds,path,j):
    shap_lime = pd.read_csv(path+"shap_lime.csv")
    shap_df = pd.DataFrame(columns=[["inds"]+obj.feats+["metrics"]],data=shap_lime.values)
    shap_first = pd.DataFrame(columns=shap_df.columns, data=np.array([shap_lime.columns]))
    shap_df = pd.concat([shap_first,shap_df])
    coefs = joblib.load(path+"/log_clf")
    suff_nece = pd.read_csv(path + "suff_nece.csv")
    suff_nece_df = pd.DataFrame(columns=[["inds"]+obj.feats+["metrics"]],data=suff_nece.values)
    the_first = pd.DataFrame(columns=suff_nece_df.columns, data=np.array([suff_nece.columns]))
    suff_nece_df = pd.concat([the_first,suff_nece_df])
    fi = pd.read_csv(path+"log_fi.csv")
    fi_df = pd.DataFrame(columns=[["inds"]+obj.feats+["metrics"]],data=fi.values)
    the_first = pd.DataFrame(columns=fi_df.columns, data=np.array([fi.columns]))
    fi_df = pd.concat([the_first,fi_df])
    for feat in obj.feats:
        if len(suff_nece_df.iloc[0][feat].split("."))==3:
            suff_nece_df.iloc[0][feat] = (".").join(suff_nece_df.iloc[0][feat].split('.')[:2])

    for feat in obj.feats:
        shap_df[feat]= shap_df[feat].astype(float)
        # suff_nece_df[feat] = suff_nece_df[feat].astype(float)
            # fi_df[feat] = fi_df[feat].astype(float)
    # for j in range(len(clusters)):
    plt.clf()
    plt.xlabel("Features")
    plt.ylabel("Scores")
        # inds = list(clusters[j].index)
    k=0
    lw = 4
    ms = 15
    fts = obj.feats.copy()
    fts.remove("ST_Depression")
    # plt.plot(obj.feats, coefs.coef_[0], label='Coefs',linewidth=lw, markersize=ms)
    # plt.plot(obj.feats, shap_df[obj.feats].iloc[k * 5], label="SHAP",linewidth=lw, markersize=ms)
    # plt.plot(obj.feats, shap_df[obj.feats].iloc[k * 5 + 3], label="LIME",linewidth=lw, markersize=ms)
    # plt.plot(obj.feats, suff_nece_df[obj.feats].iloc[k * 4], label="SUFF",linewidth=lw, markersize=ms)
    # plt.plot(obj.feats, suff_nece_df[obj.feats].iloc[k * 4 + 1], label="NECE",linewidth=lw, markersize=ms)
    # plt.plot(obj.feats, shap_df[obj.feats].iloc[k * 5 + 4], label="Preds",linewidth=lw, markersize=ms)
    # plt.plot(obj.feats, fi_df[obj.feats].iloc[k * 2], label="FI_FAVOR",linewidth=lw, markersize=ms)
    plt.bar(fts, fi_df[fts].iloc[k * 2 + 1], label="FI_AGAINST")
    # for k in inds:
    #     if 5*k+4 < len(shap_df):
    #         plt.plot(obj.feats,shap_df[obj.feats].iloc[k*5+4])
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(path+"plots_kmeans/obs" + str(k) + '.png')

# shap_inds = 5*k
# lime_inds = 5*k+3
# suff_inds = 4*k
# nece_inds = 4*k+1
# fi_favor = 2*k
# fi_against = 2*k+1
import joblib
from collections import defaultdict
path = "analysis_outputs/heart_db/fold1/"
train_inds = joblib.load(path+"/train")
test_inds = joblib.load(path+"/test")

from sklearn.cluster import KMeans
# clusters = get_clusters(1)
kmeans = KMeans(n_clusters=4, random_state=2).fit(obj.data.df[obj.feats].iloc[train_inds])
clusters = kmeans.predict(obj.data.df[obj.feats])
c=4
# for c in list(range(4)):
inds = np.argwhere(clusters==c)
inds = inds[:,0]
plot_results(inds,path,c)





        # bar_plot([suff, nece, shapp, limee, confs], ["SUFF", "NECE", "SHAP", "LIME", "Conf"], features, (20, 10),
        #              "combined")

            # for i in range(len(obj.data.test_df)):
#     instance = obj.data.test_df[obj.feats].iloc[i]
#     instance_df = pd.DataFrame(columns=features)
#     instance_df = instance_df.append(instance)
#     output = obj.get_output(instance_df)
#     local = Local()
#     suff = get_norm_(obj.sufficiency(instance_df))[:,0] #(n,features)
#     nece = get_norm_(obj.necessity(instance_df))[:,0]
#     shapp = get_norm_(obj.get_shap_vals(np.array(instance_df.iloc[0].values)))[:,0]
#     limee = get_norm_(obj.get_lime_values(instance_df.iloc[0].values))[:,0]
#     cf_fi = None
#     neighbors = generate_neighbourhood(None, instance_df, 800)
#     dists = calculateMahalanobis(neighbors[obj.feats], instance_df, np.cov(obj.data.train_df[obj.feats].values))
#     inds = np.argsort(dists[:, 0])
#     neighbors = neighbors.iloc[inds]
#     percents = []
#     sizes = [50,100,150,200,250,300,400,500,600]
#     for size in sizes:
#         neighbors_ = neighbors[:size]
#         outputs = obj.get_output(neighbors_)
#         percent = len(outputs[outputs==output[0]])/len(neighbors_)
#         percents.append(percent)
    # neighbors_ = neighbors[:200]
    # confs=[]
    # for feat in features:
    #     filtered = neighbors_[neighbors_[feat]!=instance[feat]]
    #     n_outputs = obj.get_output(filtered)
    #     diffs = len([o for o in n_outputs if o!=output[0]])
    #     diffs = diffs/len(filtered)
    #     confs.append(diffs)
    # confs = get_norm_(confs)[:,0]
    # plt.clf()
    # plt.plot(sizes,percents)
    # plt.xlabel("Size of neighborhood")
    # plt.ylabel("Percentage of same classification")
    # plt.savefig("arb/variancerf"+str(i)+'.png')
    # plt.clf()
    # bar_plot([suff, nece, shapp, limee,confs ], ["SUFF", "NECE", "SHAP", "LIME", "Conf"], features, (20, 10), "combined")
# path = "analysis_outputs/heart_db_old"
# mkdir_p(path)
# save_models(path)

    # confs = []
    # for feat in features:
    #     filtered = neighbors[neighbors[feat]!=instance[feat]]
    #     n_outputs = obj.get_output(filtered)
    #     diffs = len([o for o in n_outputs if o!=output[0]])
    #     diffs = diffs/len(filtered)
    #     confs.append(diffs)
    # indss = np.argsort(confs)
    # plt.clf()
    # feat_name = []
    # for feature in obj.feats:
    #     feat_name.append(obj.data.nick_name_dict[feature])
    # plt.bar(np.array(feat_name)[indss], np.array(confs)[indss])
    # plt.savefig('arb/foo'+str(i)+'.png')
    # plt.style.use("seaborn-dark")
    # # df = pd.DataFrame(columns=labels, data=list_of_scores, index=legends)
    # new_scores = []
    # legends = ["SUFF","NECE","SHAP","Prob"]
    # inds = np.argsort(shapp)
    # new_scores.append(list(shapp)+[legends[0]])
    # for i, scores in enumerate([nece, suff, np.array(confs)]):
    #     new_scores.append(list(scores[inds]) + [legends[i+1]])
    # features = np.array(features)
    # features=features[inds]
    # there is some problem with the sorting
    # you have to implement some form of distance metric to not equate age=1 change with age = 2 change and so on

    # df = pd.DataFrame(columns=list(features) + ["metric"], data=new_scores)
    # df.index = df["metric"]
    # df = df.drop("metric", axis=1)
    # df = df.T


# for i in range(len(neighbors)):
#     suff = obj.sufficiency(neighbors.iloc[i:i+1])
#     nece = obj.necessity(neighbors.iloc[i:i+1])
#     shapp = obj.get_shap_vals(np.array(neighbors.iloc[i].values))
#     limee = obj.get_lime_values(neighbors.iloc[i].values)
#     change = []
#     # for feat in features:
#     #     ini_prob = obj.get_probability(neighbors.iloc[i:i+1])
#     #     prob_after_removal = obj.get_output_clfs(feat, neighbors.iloc[i:i+1])
#     #     b = [np.argmax(p) for p in ini_prob]
#     #     a = [np.argmax(p) for p in prob_after_removal]
#     #     change.append(np.log(ini_prob[0][b[0]]) - np.log(prob_after_removal[0][a[0]]))
#     # change = get_norm(change)
#     # prob = change.squeeze()
#     probs = obj.probability_change(neighbors.iloc[i:i+1])
#     bar_plot([suff,nece,shapp,limee,probs],["SUFF","NECE","SHAP","LIME","Prob"],features,(20,10),trial)
#
# for feature in features:
#     bee_swarm(feature, features, "combined")







