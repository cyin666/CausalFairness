import pandas as pd
import numpy as np

def msd_one(x1,t1,meas,boots):
    """
    For each bootstrap index "boots", calculate the fairness measure "meas" from data "x" and the corresponding index name "t"
    
    :boots:(nested dictionary) boostrap indexes, including index of all samples, as well as the index of 0-level ("id0") and 1-level ("id1") 
    :x1:(1-d array) data input for calculation 
    :t1:(string) string indicating name of the index for x1
    :meas:(string) name for the output measure
    :return:(dataframe) with value=calculated measure, boot=row number/bootstrap id, measure=meas (name of the calculated measure)
    """
    meas_result = []
    
    for b in boots:
        boot_data1_tmp = x1[boots[b][t1]]
        
        meas_result_tmp = np.nanmean(boot_data1_tmp)
        meas_result.append(meas_result_tmp)
        
    meas_df = pd.DataFrame({'value' : meas_result,
                            'measure': meas
                          }).reset_index()
    meas_df.columns = ['boot','value','measure']
    return meas_df
        
    
def msd_two(x1,t1,x2,t2,meas,boots):
    """
    For each bootstrap index "boots", calculate the fairness measure "meas" from data "x" and the corresponding index name "t"
    
    :boots:(dictionary) boostrap indexes, including index of all samples, as well as the index of 0-level ("id0") and 1-level ("id1") 
    :x1:(1-d array) data input for calculation 
    :t1:(string) string indicating name of the index for x1
    :x2:(1-d array) data input for calculation 
    :t2:(string) string indicating name of the index for x2
    :meas:(string) name for the output measure
    :return:(dataframe) value=calculated measure, boot=row number/bootstrap id, measure=meas (name of the calculated measure)
     """
    meas_result = []
    
    for b in boots:
        boot_data1_tmp = x1[boots[b][t1]]
        boot_data2_tmp = x2[boots[b][t2]]
        
        meas_result_tmp = np.nanmean(boot_data1_tmp) + np.nanmean(boot_data2_tmp)
        meas_result.append(meas_result_tmp)
        
    meas_df = pd.DataFrame({'value' : meas_result,
                            'measure': meas
                          }).reset_index()
    meas_df.columns = ['boot','value','measure']
    return meas_df
    

    

def msd_three(x1,t1,x2,t2,x3,t3,meas,boots):
    """
    For each bootstrap index "boots", calculate the fairness measure "meas" from data "x" and the corresponding index name "t"
    
    :x1:(1-d array) data input for calculation 
    :t1:(string) string indicating name of the index for x1
    :x2:(1-d array) data input for calculation 
    :t2:(string) string indicating name of the index for x2
    :x3:(1-d array) data input for calculation 
    :t3:(string) string indicating name of the index for x3
    :meas:(string) name for the output measure
    :return:(dataframe) value=calculated measure, boot=row number/bootstrap id, measure=meas (name of the calculated measure)
    """
    meas_result = []
    
    for b in boots:
        boot_data1_tmp = x1[boots[b][t1]]
        boot_data2_tmp = x2[boots[b][t2]]
        boot_data3_tmp = x3[boots[b][t3]]
        
        meas_result_tmp = np.nanmean(boot_data1_tmp) + np.nanmean(boot_data2_tmp) + np.nanmean(boot_data3_tmp)
        meas_result.append(meas_result_tmp)
        
    meas_df = pd.DataFrame({'value' : meas_result,
                            'measure': meas
                          }).reset_index()
    meas_df.columns = ['boot','value','measure']
    return meas_df


def inh_str(x,meas,set0=False,setna=False):
    """
    Addjust the values in measure df x for later operations
    
    :x:(df) dataframe with the calcualted measures for each bootstrap sample
    :meas:(string) to rename the measure's name
    :set0:
    :setna:
    :return:(dataframe) value=calculated measure, boot=row number/bootstrap id, measure=meas (name of the calculated measure)
    """
    x = x.copy()
    x['measure'] = meas
    if set0:
        x['value'] = 0 
    if setna:
        x['value'] = np.nan
    
    return x

def auto_dummy(data, col):
    """
    Automatically change the categorical variables in the "col" columns of data into dummies
    
    :data:(dataframe) the entire dataset
    :col:(array) the columns to be screened and adjusted
    :return:(dataframe, array) the adjusted dataframe, and the adjusted column names of "col"
    """
    data_adj = data.copy()
    
    data_adj_col = data_adj[data_adj.columns[np.isin(data_adj.columns, col)]]
    data_adj_other = data_adj[data_adj.columns[~np.isin(data_adj.columns, col)]]
        
    col_cat = data_adj_col.dtypes.index[np.isin(data_adj_col.dtypes, np.array(["object","string","category"]))].values
    col_other = col[~(np.isin(col, col_cat))]
    
    data_adj_col_cat = data_adj_col[col_cat]

    data_adj_col_other = data_adj_col[col_other]
    
    data_adj_col_cat = pd.get_dummies(data = data_adj_col_cat,columns = col_cat)
    col_cat_adj = data_adj_col_cat.columns.values
    col_adj = np.concatenate([col_cat_adj,col_other])
    
    data_adj_col = pd.concat([data_adj_col_cat,data_adj_col_other],axis=1)
    data_adj = pd.concat([data_adj_col,data_adj_other],axis=1)    
    
    return data_adj, col_adj 
    