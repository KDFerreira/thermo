#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 09:25:25 2017

@author: kevinferreira
"""

def heatingPolyFit(path, alpha, neg_RC_inv = -0.000179180576858):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from math import pow, exp
    
#    path = '../data/data.csv'
#    alpha = 5
#    neg_RC_inv = -0.000179180576858
    
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
    
    
    df['month'] = pd.to_datetime(df['Time']).map(lambda x: x.month)
    df['day'] = pd.to_datetime(df['Time']).map(lambda x: x.day)
    df['hour'] = pd.to_datetime(df['Time']).map(lambda x: x.hour)
    df['t'] = pd.to_datetime(df['Time']).map(lambda x: x.hour*60*60+x.minute*60)
    
    #==============================================================================
    # # NEED TO TRANSFORM TO GET Y FOR LINEAR REGRESSION
    #==============================================================================
    df['y'] = df['T_ctrl']-df['T_out']
    
    df_heating = df[(df['T_ctrl']<df['T_stp_heat']) & (df['auxHeat1'] == 300) & (df['fan'] == 300) & (df['T_ctrl']>=df['T_out'])]
    
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
    
    
    
    
    
    #==============================================================================
    # start doing polynomial fitting
    #==============================================================================
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from numpy import newaxis
    import seaborn as sns; sns.set()
    
    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree),
                             LinearRegression(**kwargs))
    
    #need to make x with proper scale
    X = [0]
    for t in range (1,len(df_result['Time'])):
        temp = pd.to_datetime(df_result['Time'][t])-pd.to_datetime(df_result['Time'][t-1])
        X.append(X[t-1]+temp.seconds/(300/alpha))
    
    X = np.array(X)
    Y = np.array(df_result['Qh/C'])
    
    xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=0.33, random_state=42)
    
    xTrain = xTrain[:,newaxis]
    xTest = xTest[:,newaxis]
    
    #loop through various degrees of polynomials
    
    MSE_train = []
    MSE_test = []
    for d in range (10):
        model = PolynomialRegression(d)
        model.fit(xTrain,yTrain)
        yFit_train = model.predict(xTrain)
        yFit_test = model.predict(xTest)
        
        MSE_train.append(mean_squared_error(yFit_train, yTrain))
        MSE_test.append(mean_squared_error(yFit_test, yTest))
    
    plt.figure(3)
    plt_train = plt.scatter([range(10)],MSE_train)
    plt_test = plt.scatter([range(10)],MSE_test,c='r',marker='^')
    plt.xlabel('Model Degree')
    plt.ylabel('MSE')
    plt.title('Polynomial Model Degree Validation')
    plt.legend((plt_train,plt_test),('Training','Testing'),loc='best')
    plt.show()
    
    #==============================================================================
    # Fit Linear model
    #==============================================================================
    
    X_lin = X[:,newaxis]
    Y_lin = Y
    
    model = PolynomialRegression(1)
    model.fit(X_lin,Y_lin)
    
    #now that the model is fit we use it to get Qh/C at the start and end
    xFit = np.array([X[0],X[len(X)-1]])
    xFit = xFit[:,newaxis]
    yFit_lin = model.predict(xFit)
    
    return list(yFit_lin)

def main():
    path = '../data/data.csv'
    alpha = 5 #used for time scale. time unit is 1 minute.
    Qh_C_start, Qh_C_end = heatingPolyFit(path,alpha)
    print '\n\n Starting Qh/C = {0}'.format(Qh_C_start)
    print '\n Ending Qh/C = {0}'.format(Qh_C_end)
    return

if __name__  == '__main__':
    main()
