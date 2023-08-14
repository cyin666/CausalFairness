from helpers import msd_one, msd_two, msd_three, inh_str
import pandas as pd
import numpy as np
from econml.grf import CausalForest

#removed tune_params for now compared to the original version. Unlike the GRF library in R, the python's corresponding EconML's causal forest does not allow auto-tuning. I added several other parameters here for manual tuning.
def ci_crf(data, X, Z, W, Y, x0, x1, rep, nboot = 100,crf_n_estimators = 100, crf_criterion = "het", crf_min_samples_leaf = 5, crf_max_features = "sqrt", crf_honest = True):
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
    :crf_n_estimators:(integer) Number of trees.
    :crf_criterion:(string) "mse" or "het". The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error in a linear moment estimation tree and “het” for heterogeneity score.
    :crf_min_samples_leaf:(integer) The minimum number of samples required to be at a leaf node. 
    :crf_max_features:(int, float, {“auto”, “sqrt”, “log2”}, or None, default 'sqrt' to keep consistency with R grf library) – The number of features to consider when looking for the best split.
    :crf_honest:(logical) Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal sized subsets, the train and the val set. All samples in train are used to create the split structure and all samples in val are used to calculate the value of each node in the tree.
    
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
        ind = np.random.randint(0,nrow_boot_df,nrow_boot_df)
        
        idx0 = boot_data[X][ind] == x0
        ind0 = ind[idx0]
        ind1 = ind[~idx0]
        
        boots[key]["all"] = ind
        boots[key]["id0"] = ind0
        boots[key]["id1"] = ind1
    
    y = boot_data[Y].astype(float)
    tv = msd_two(y, "id1", -y, "id0", "tv", boots) 
    
    if len(Z) == 0:
        te = inh_str(tv,"te")
        ett = inh_str(tv,"ett")
        expse_x1 = inh_str(tv,"expse_x1",set0=True)
        expse_x0 = inh_str(tv,"expse_x0",set0=True)
        ctfse = inh_str(tv,"ctfse",set0=True)
        crf_te = np.array([tv['value'][0]]*data.shape[0])
    else:
        crf_tmp = CausalForest(n_estimators = crf_n_estimators, criterion = crf_criterion, min_samples_leaf = crf_min_samples_leaf, max_features = crf_max_features, honest = crf_honest  )
        crf_tmp.fit(X = boot_data[Z].values, T = np.where(boot_data[X].values==x0,0,1), y = y.values)
        crf_te = crf_tmp.oob_predict(Xtrain =  boot_data[Z].values).ravel()
        
        te = msd_one(crf_te,"all","te",boots)
        ett = msd_one(crf_te,"id0","ett",boots)
        ctfse = msd_three(crf_te,"id0",-y,"id1",y,"id0","ctfse",boots)
        expse_x0 = inh_str(tv, "expse_x0", setna=True)
        expse_x1 = inh_str(tv, "expse_x1", setna=True)

    if len(W)==0:
        nde = inh_str(te,"nde")
        ctfde = inh_str(ett,"ctfde")
        ctfie = inh_str(ett,"ctfie",set0=True)
        nie = inh_str(te,"nie",set0=True)
    else:
        crf_tmp = CausalForest(n_estimators = crf_n_estimators, criterion = crf_criterion, min_samples_leaf = crf_min_samples_leaf, max_features = crf_max_features, honest = crf_honest  )
        crf_tmp.fit(X = boot_data[np.concatenate([Z,W])].values, T = np.where(boot_data[X].values==x0,0,1), y = y.values)
        crf_med = crf_tmp.oob_predict(Xtrain = boot_data[np.concatenate([Z,W])].values).ravel()
        
        nde = msd_one(crf_med,"all","nde",boots)
        ctfde = msd_one(crf_med,"id0","ctfde",boots)
        nie = msd_two(crf_med,"all",-crf_te,"all","nie",boots)
        ctfie = msd_two(crf_med,"id0",-crf_te,"id0","ctfie",boots)
        
    res = pd.concat([tv,te,expse_x1,expse_x0,ett,ctfse,nde,nie,ctfde,ctfie])
    res['rep'] = rep
    
    
    
    return res
        