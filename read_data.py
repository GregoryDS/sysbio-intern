#!python

"""
Reading in the data.
"""
import numpy as np
import pandas as pd
# from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from components import data_handling

import time

rndm_state = 7
np.random.seed(rndm_state)

if __name__ == "__main__":

    path_to_feature_val = "./data/GE_PPI/GEO_HG_PPI.csv"
    path_to_feature_graph = "./data/GE_PPI/HPRD_PPI.csv"
    path_to_labels = "./data/GE_PPI/labels_GEO_HG.csv"

    DP = data_handling.DataPreprocessor(path_to_feature_values=path_to_feature_val, path_to_feature_graph=path_to_feature_graph,
                                        path_to_labels=path_to_labels)
    X = DP.get_feature_values_as_np_array()  # gene expression
    # A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # adjacency matrix of the PPI network, no use of it here
    y = DP.get_labels_as_np_array()  # labels
	
	# !!!
    # Making data lying in the interval [0, 8.35]
    X = X - np.min(X)
    
		
    print("GE data, X shape: ", X.shape)
    print("Labels, y shape: ", y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Need to know which patients got into train and test subsets
    _, _, patient_indexes_train, patient_indexes_test = train_test_split(X, DP.labels.columns.values.tolist(), test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Data frame with test patients and corresponding ground truth labels
    patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})

    
    print("X_train max", np.max(X_train))
    print("X_train min", np.min(X_train))
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train, shape: ", y_train.shape)
    print("y_test, shape: ", y_test.shape)

    
