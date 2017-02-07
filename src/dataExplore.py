

def dataExplore(path):
    
    import pandas as pd
    import seaborn as sns; sns.set()

    df = pd.read_csv(path)
    
    df = df.drop(['T_stp_cool'],axis=1)
    
    #==============================================================================
    # one day's worth of data
    #==============================================================================
    
    test = df[['T_ctrl','T_stp_heat']].head(24*60/5)
    test.plot()

    #==============================================================================
    # plot entire time period
    #==============================================================================

    test3 = df[['T_ctrl','T_stp_heat','T_out']]
    
    test3.plot()
        
    #==============================================================================
    # check where there is missing data
    #==============================================================================
    
    df_null = df[df.isnull().any(axis=1)]
    numNullRows = df_null.Time.count()
                 
    #==============================================================================
    # analyze drop in T_stp_heat
    #==============================================================================

    pd.DataFrame.hist(pd.DataFrame(df['T_stp_heat']))
    
    df_stp_heat = df[(df['T_stp_heat']< 60)]
    numT_stp_heat = df_stp_heat.Time.count()
    
    return numNullRows, numT_stp_heat
    
def main():
    path = '../data/data.csv'

    numNullRows, numT_stp_heat = dataExplore(path)
    print '\n\n numNullRows = {0}'.format(numNullRows)
    print '\n numT_stp_heat = {0}'.format(numT_stp_heat)
    return

if __name__  == '__main__':
    main() 
                   

