import json 
import pickle
import numpy as np 

from sklearn.cluster import KMeans 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA 

################################################## classical clustering method ##################################################
def cluster(cluster_model_path, data_json):
    with open(cluster_model_path, 'rb') as file:
        cluster_models_dict = pickle.load(file)
    
    with open(data_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    # Get data from json file
    data = []
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                data.append(feature)

    # Convert data to NumPy array
    data = np.array(data)
        
    pca = cluster_models_dict['pca']
    cluster_model = cluster_models_dict['clustering_model']
    
    pca_vectors = pca.fit_transform(data)
    cluster_ids = cluster_model.fit_predict(pca_vectors)

    # Add cluster ID for each feature
    clustered_data = {}
    feature_index = 0
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                cluster_id = cluster_ids[feature_index] + 1
                clustered_data.setdefault(ppt_name, {}).setdefault(feature_file_name, int(cluster_id))
                feature_index += 1
                    
    # # Save clustering results as JSON files
    # with open('clustered_train.json', 'w', encoding='utf-8') as output_file:
    #     json.dump(clustered_data, output_file, ensure_ascii=False, indent=4)

    return clustered_data



def kmeans_cluster(file_json, num_pca_components, num_clusters):
    with open(file_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Get data from json file
    data = []
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                data.append(feature)

    # Convert data to NumPy array
    data = np.array(data)

    # Apply PCA 
    pca = PCA(n_components=num_pca_components)
    reduced_data = pca.fit_transform(data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(reduced_data)
    
    # Add cluster ID for each image feature
    clustered_data = {}
    feature_index = 0
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                cluster_id = cluster_ids[feature_index] + 1 # Add 1 to start from 1 
                clustered_data.setdefault(ppt_name, {}).setdefault(feature_file_name, int(cluster_id))
                feature_index += 1

    return clustered_data


def hierarchical_cluster(file_json, num_pca_components, num_clusters):
    with open(file_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
         
    # Get data from json file
    data = []
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                data.append(feature)

    # Convert data to NumPy array
    data = np.array(data)

    # Apply PCA 
    pca = PCA(n_components=num_pca_components)
    reduced_data = pca.fit_transform(data)
    
    # Perform hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters, 
                                           metric='euclidean', 
                                           linkage='complete') # average linkage도 시도해볼 것
    cluster_ids = hierarchical.fit_predict(reduced_data)

    # Add cluster ID for each image feature
    clustered_data = {}
    feature_index = 0
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                cluster_id = cluster_ids[feature_index] + 1  # Add 1 to start from 1 
                clustered_data.setdefault(ppt_name, {}).setdefault(feature_file_name, int(cluster_id))
                feature_index += 1

    return clustered_data


def GaussianMixture_cluster(file_json, num_pca_components, num_clusters):
    with open(file_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
         
    # Get data from json file
    data = []
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                data.append(feature)

    # Convert data to NumPy array
    data = np.array(data)

    # Apply PCA 
    pca = PCA(n_components=num_pca_components)
    reduced_data = pca.fit_transform(data)
    
    # Perform Gaussian Mixture Model clustering
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    cluster_ids = gmm.fit_predict(reduced_data)

    # Add cluster ID for each image feature
    clustered_data = {}
    feature_index = 0
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                cluster_id = cluster_ids[feature_index] + 1  # Add 1 to start from 1 
                clustered_data.setdefault(ppt_name, {}).setdefault(feature_file_name, int(cluster_id))
                feature_index += 1

    return clustered_data
    
    
def dbscan_cluster(file_json, num_pca_components, eps, min_samples):
    with open(file_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Get data from json file
    data = []
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                data.append(feature)

    # Convert data to NumPy array
    data = np.array(data)

    # Apply PCA 
    pca = PCA(n_components=num_pca_components)
    reduced_data = pca.fit_transform(data)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = dbscan.fit_predict(reduced_data)

    # Add cluster ID for each image feature
    clustered_data = {}
    feature_index = 0
    for ppt_name, slide_data in json_data.items():
        for feature_file_name, feature_list in slide_data.items():
            for feature in feature_list:
                cluster_id = cluster_ids[feature_index] + 1  # Add 1 to start from 1 
                clustered_data.setdefault(ppt_name, {}).setdefault(feature_file_name, int(cluster_id))
                feature_index += 1

    return clustered_data


################################################## New clustering method ##################################################
