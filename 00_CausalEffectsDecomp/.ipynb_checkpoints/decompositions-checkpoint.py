from causal_forest import ci_crf
from helpers import auto_dummy
import pandas as pd
import numpy as np

def fairness_cookbook(data, X, Z, W, Y, x0, x1, method = "causal_forest", nboot1 = 1, nboot2 = 100, crf_n_estimators = 100, crf_criterion = "mse", crf_min_samples_leaf = 5, crf_max_features = "sqrt", crf_honest = True):
    """
    Main function to decompose the causal effects.
    
    :data:(dataframe)
    :X:(array) scalar giving the name of the protected attribute. Must be one of the entries of data.columns
    :Z:(array) vector giving the names of all mediators. Must be one of the entries of data.columns
    :W:(array) vector giving the names of all confounders. Must be one of the entries of data.columns
    :Y:(array) scalar giving the name of the outcome. Must be one of the entries of data.columns
    :x0:(string) scalar values giving the two levels of the binary protected attribute.
    :x1:(string) scalar values giving the two levels of the binary protected attribute.
    :method:("causal_forest" for causal forest from EconML.grf  or "medDML" for mediation analysis with double-machine learning) Only support "causal_forest" for now. 
    :nboot1:(integer) scalar determining the number of outter bootstrap repetitions, that is, how many times the fitting procedure is repeated. 
    :nboot2:(integer) scalar determining the number of inner bootstrap repetitions, that is, how many bootstrap samples are taken after the potential outcomes are obtained from the estimation procedure. 
        
    see EconML documentaion for details of the parameters below
    :crf_n_estimators:(integer) Number of trees.
    :crf_criterion:(string) "mse" or "het". The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error in a linear moment estimation tree and “het” for heterogeneity score.
    :crf_min_samples_leaf:(integer) The minimum number of samples required to be at a leaf node. 
    :crf_max_features:(int, float, {“auto”, “sqrt”, “log2”}, or None, default 'sqrt' to keep consistency with R grf library) – The number of features to consider when looking for the best split.
    :crf_honest:(logical) Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal sized subsets, the train and the val set. All samples in train are used to create the split structure and all samples in val are used to calculate the value of each node in the tree.
    :return1:(dataframe) This df inclused all types of causal effects. value=calculated measure, boot=row number/bootstrap id, measure=meas (name of the calculated measure)
    :return2:(dataframe) Aggregated summary of return1.
    """
    
    Z = None if ((len(Z)==0) | (Z is "")) else Z
    W = None if ((len(W)==0) | (W is "")) else W
    
    Z_dtypes = np.isin(data[Z].dtypes, np.array(["object","string","category"]))
    W_dtypes = np.isin(data[W].dtypes, np.array(["object","string","category"]))
    
    if Z_dtypes.sum() > 0:
        data, Z = auto_dummy(data = data, col = Z)
    if W_dtypes.sum() > 0:
        data, W = auto_dummy(data = data, col = W)
    
    idx = (data[X] == x1)
    
    res = pd.DataFrame({'boot':[],
             'value':[],
             'measure':[],
             'rep':[]})
    
    if method == "causal_forest":
        for r in range(nboot1):
            res_tmp = ci_crf(data=data, X=X, Z=Z, W=W, Y=Y, x0=x0, x1=x1, rep=r, nboot = nboot2, crf_n_estimators = crf_n_estimators, crf_criterion = crf_criterion, crf_min_samples_leaf = crf_min_samples_leaf, crf_max_features = crf_max_features, crf_honest = crf_honest)
            
            res = pd.concat([res,res_tmp])

    res_summary = res.groupby("measure").agg({'value':['mean','std']})
    return res, res_summary