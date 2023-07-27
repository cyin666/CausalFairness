from ci_helpers import msd_one, msd_two, msd_three, inh_str
import pandas as pd
import numpy as np
from econml.grf import CausalForest

#removed tune_params for now compared to the original version. Unlike the GRF library in R, the python's corresponding EconML's causal forest does not allow auto-tuning. I added several other parameters here for manual tuning.
def ci_crf(data, X, Z, W, Y, x0, x1, rep, nboot = 100, crf_criterion = "mse", crf_min_samples_leaf = 5, max_features = "sqrt", honest = True):
    """
    Use causal random forest to decompose the causal effects.
    
    :data:(dataframe)
    :X:(array) scalar giving the name of the protected attribute. Must be one of the entries of data.columns
    :Z:(array) vector giving the names of all mediators. Must be one of the entries of data.columns
    :W:(array) vector giving the names of all confounders. Must be one of the entries of data.columns
    :Y:(array) scalar giving the name of the outcome. Must be one of the entries of data.columns
    :x0:(string) scalar values giving the two levels of the binary protected attribute.
    :x1:(string) scalar values giving the two levels of the binary protected attribute.
    :rep:(integer) scalar index input from the outter bootstrap loop
    :nboot:(integer) scalar determining the number of inner bootstrap repetitions, that is, how many bootstrap samples are taken after the potential outcomes are obtained from the estimation procedure. 
    
    see EconML documentaion for details of the parameters below
    :crf_criterion:(string) "mse" or "het". The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error in a linear moment estimation tree and “het” for heterogeneity score.
    :crf_min_samples_leaf:(integer) The minimum number of samples required to be at a leaf node. 
    :max_features:(int, float, {“auto”, “sqrt”, “log2”}, or None, default 'sqrt' to keep consistency with R grf library) – The number of features to consider when looking for the best split.
    :honest:(logical) Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal sized subsets, the train and the val set. All samples in train are used to create the split structure and all samples in val are used to calculate the value of each node in the tree.
    
    :return:(dataframe) This df inclused all types of causal effects. value=calculated measure, boot=row number/bootstrap id, measure=meas (name of the calculated measure)
    """
    #load data by using outter bootstrap index
    nrow_df = data.shape[0]
    boot_samp = np.random.randint(0,nrow_df,nrow_df) if rep>1 else np.arange(0,nrow_df)  
    boot_data = data.iloc[boot_samp,:].reset_index(drop=True)
    
    #creat index of this inner bootstrap
    nrow_boot_df = boot_data.shape[0]
    boots = dict.fromkeys(np.arange(nboot))
    for key in boots:
        boots[key] = dict.fromkeys(["all","id0","id1"])
        ind = np.random.randin(0,nrow_boot_df,nrow_boot_df)
        
        idx0 = boot_data[X][ind] == x0
        ind0 = ind[idx0]
        ind1 = ind[~idx0]
        
        boots[key]["all"] = ind
        boots[key]["id0"] = ind0
        boots[key]["id1"] = ind1
    
    y = boot_data[Y].astype(float)
    tv = msd_two(y, "id1", -y, "id0", "tv", boots)    