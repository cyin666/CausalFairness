from causal_forest import ci_crf
from helpers import auto_dummy
import pandas as pd
import numpy as np

def fairness_cookbook(data, X, Z, W, Y, x0, x1, method = "causal_forest", nboot1 = 1, nboot2 = 100, crf_n_estimators = 100, crf_criterion = "mse", crf_min_samples_leaf = 5, crf_max_features = "sqrt", crf_honest = True):
    
    
    
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