#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 18:17:08 2017

@author: kevinferreira
"""
def cooling(path, alpha):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    #from matplotlib import dates
    from math import pow
    from statsmodels.tsa.stattools import adfuller
    import seaborn as sns; sns.set()

    
    df = pd.read_csv(path)
    
    #initialize result lists
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
    result_Tctrl = list([])
    
    df['month'] = pd.to_datetime(df['Time']).map(lambda x: x.month)
    df['day'] = pd.to_datetime(df['Time']).map(lambda x: x.day)
    df['hour'] = pd.to_datetime(df['Time']).map(lambda x: x.hour)
    df['t'] = pd.to_datetime(df['Time']).map(lambda x: x.hour*60*60+x.minute*60)
    
    #==============================================================================
    # # NEED TO TRANSFORM TO GET Y FOR LINEAR REGRESSION
    #==============================================================================
    df['y'] = np.log(df['T_ctrl']-df['T_out'])
    
    df_cooling = df[(df['T_ctrl']>df['T_stp_heat']) & (df['auxHeat1'] == 0) & (df['fan'] == 0) & (df['T_ctrl']>df['T_out'])]

    
    re_index = df_cooling.set_index(['month','day','hour','T_out']).sort_index()
    i = list(re_index.index)
    # get rid of duplicates
    i = list(set(i))
    i.sort()
    
    #c = 0
    for j in range(len(i)):
        #print 'j= {0}'.format(j)
        temp = re_index.ix[i[j]]
        num = temp.Time.count()
        if num > 5:
            #print 'c= {0}'.format(c)
            #print 'num ={0}'.format(num)
            if temp.t[num-1]-temp.t[0] == 300*(num-1):
                #c=c+1
                #then this all belongs to one session
                #start line fitting
                y = np.array(temp['y'])
    #            x = np.array([np.ones(num),temp['t']-temp.t[0]]).T
                x = np.array([np.ones(num),np.dot(alpha,np.arange(num))]).T
                xx_inv = np.linalg.inv(np.dot(x.T,x))
                b = np.dot(np.dot(xx_inv,x.T),y)
                
                result_int.append(b[0])
                result_slope.append(b[1])
                
                result_m.append(i[j][0])
                result_d.append(i[j][1])
                result_h.append(i[j][2])
                result_Tout.append(i[j][3])
                result_time.append(temp.Time[0])
                result_Tctrl.append(temp.T_ctrl[0])
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
        'T_ctrl':result_Tctrl,
        'T_out':result_Tout,
        'slope':result_slope,
        'intercept':result_int,
        'SSE':result_SSE,
        'MSE':result_MSE,
        'j':result_j,
        'num':result_num}
    
    df_result = pd.DataFrame(result)
    df_result['deltaT'] = np.array(df_result['T_ctrl']-df_result['T_out'])
    
    plt.figure(1)
    plt.plot(pd.to_datetime(result['Time']),result['slope'])
    plt.xlabel('Date')
    plt.xticks(rotation='vertical')
    plt.ylabel('-1/RC')
    plt.title('Series of -1/RC')
    plt.show()
    
    plt.figure(10)
    plt.plot(pd.to_datetime(result['Time']),list(df_result['deltaT']))
    plt.xlabel('Date')
    plt.xticks(rotation='vertical')
    plt.ylabel('T_ctrl - T_out')
    #plt.title('Series of -1/RC')
    plt.show()
    
    print 'number of cooling sessions = {0}'.format(len(result['slope']))
    #plt.xaxis
    #plt.show()
    
    #==============================================================================
    # test for stationary -1/RC
    #==============================================================================
    
    adf_test = adfuller(result['slope'])
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
    	print('\t%s: %.3f' % (key, value))
    
    mean = df_result['slope'].mean()
    std = df_result['slope'].std()
    
    #==============================================================================
    # to reduce some of the noise I filter out samples beyond 2*std from mean
    #==============================================================================
    
    df_result_filtered = df_result[(df_result['slope'] <= mean+2*std) &  (df_result['slope'] >= mean-2*std)]
    #neg_RC_inv = df_result_filtered['slope'].mean()
    
    return df_result_filtered['slope'].mean(), df_result_filtered['slope'].var()
    
def main():
    path = '../data/data.csv'
    alpha = 5 #used for time scale. time unit is 1 minute.
    neg_RC_inv, variance = cooling(path,alpha)
    print '\n\n neg_RC_inv = {0}'.format(neg_RC_inv)
    print '\n variance = {0}'.format(variance)
    return

if __name__  == '__main__':
    main()                  