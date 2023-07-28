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
  