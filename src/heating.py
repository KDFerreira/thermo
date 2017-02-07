#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 23:26:58 2017

@author: kevinferreira
"""
def heating(path, alpha, neg_RC_inv = -0.000179180576858):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from math import pow, exp
    from statsmodels.tsa.stattools import adfuller
    import seaborn as sns; sns.set()
    
    df = pd.read_csv(path)
    
    result_m = list([])
    result_d = list([])
    result_h = list([])
    result_time = list([])
    result_Tout = list([])
    result_slope = list([])
    result_int = list([])
    result_SSE = list([])
    result_MSE = list([])
    result_j = list([])
    result_num = list([])
    
    #alpha = 5 #used for time scale. time unit is minute.
    #neg_RC_inv = -0.000179180576858 # <------------ change this
    
    df['month'] = pd.to_datetime(df['Time']).map(lambda x: x.month)
    df['day'] = pd.to_datetime(df['Time']).map(lambda x: x.day)
    df['hour'] = pd.to_datetime(df['Time']).map(lambda x: x.hour)
    df['t'] = pd.to_datetime(df['Time']).map(lambda x: x.hour*60*60+x.minute*60)
    
    #==============================================================================
    # # NEED TO TRANSFORM TO GET Y FOR LINEAR REGRESSION
    #==============================================================================
    df['y'] = df['T_ctrl']-df['T_out']
    
    df_heating = df[(df['T_ctrl']<df['T_stp_heat']) & (df['auxHeat1'] == 300) & (df['fan'] == 300) & (df['T_ctrl']>df['T_out'])]
    
    re_index = df_heating.set_index(['month','day','hour','T_out']).sort_index()
    i = list(re_index.index)
    # get rid of duplicates
    i = list(set(i))
    i.sort()
    
    #c = 0
    for j in range(len(i)):
        #print 'j= {0}'.format(j)
        temp = re_index.ix[i[j]]
        num = temp.Time.count()
        if num > 3:
            #print 'c= {0}'.format(c)
            #print 'num ={0}'.format(num)
            if temp['t'].max()-temp['t'].min() == 300*(num-1):
                #c=c+1
                #then this all belongs to one session
                #start line fitting
                y = np.array(temp['y'])
                li = list(range(num))
                x = np.array([np.ones(num),np.array([exp(neg_RC_inv*alpha*t) for t in li])]).T
                xx_inv = np.linalg.inv(np.dot(x.T,x))
                b = np.dot(np.dot(xx_inv,x.T),y)
                
                result_int.append(b[0])
                result_slope.append(b[1])
                
                result_m.append(i[j][0])
                result_d.append(i[j][1])
                result_h.append(i[j][2])
                result_Tout.append(i[j][3])
                result_time.append(temp.Time[0])
                result_j.append(j)
                
                SSE = 0
                for k in range (num):
                    SSE = SSE + pow(y[k]-(b[0]+b[1]*x[k][1]),2)
                MSE = SSE/num
                
                result_SSE.append(SSE)
                result_MSE.append(MSE)
                result_num.append(num)
                
    #print c
    
    result = { 'month': result_m,
        'day':result_d,
        'hour':result_h,
        'Time':result_time,
        'T_out':result_Tout,
        'slope':result_slope,
        'intercept':result_int,
        'SSE':result_SSE,
        'MSE':result_MSE,
        'j':result_j,
        'num':result_num}
    
    df_result = pd.DataFrame(result)
    
    df_result['Qh/C'] = np.array([-1*t/neg_RC_inv for t in list(df_result['intercept'])])
    
    plt.figure(2)
    plt.plot(pd.to_datetime(result['Time']),list(df_result['Qh/C']))
    plt.xlabel('Date')
    plt.xticks(rotation='vertical')
    plt.ylabel('Qh/C')
    plt.title('Series of Qh/C')
    #plt.xaxis
    plt.show()
    
    #==============================================================================
    # test for stationary -1/RC
    #==============================================================================
    
    adf_test = adfuller(list(df_result['Qh/C']))
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
    	print('\t%s: %.3f' % (key, value))
     
    print 'number of heating sessions = {0}'.format(len(df_result['Qh/C']))
    
    return df_result['Qh/C'].mean(), df_result['Qh/C'].var()

# test filtering of values and log transform
#df_result_filtered = df_result[(df_result['Qh/C'] <= mean+2*std) &  (df_result['Qh/C'] >= mean-2*std) &  (df_result['Qh/C'] > 0)]
#                               
#df_result_filtered_log = np.log(df_result_filtered['Qh/C']) 
#                               
#adf_test = adfuller(list(df_result_filtered_log))
#print('ADF Statistic: %f' % adf_test[0])
#print('p-value: %f' % adf_test[1])
#print('Critical Values:')
#for key, value in adf_test[4].items():
#	print('\t%s: %.3f' % (key, value))

def main():
    path = '../data/data.csv'
    alpha = 5 #used for time scale. time unit is 1 minute.
    Qh_C, variance = heating(path,alpha)
    print '\n\n Qh/C = {0}'.format(Qh_C)
    print '\n variance = {0}'.format(variance)
    return

if __name__  == '__main__':
    main()
 
