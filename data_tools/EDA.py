
# The x parameter is the input column from a pandas dataframe ie. df['colmun']
# This method requires pandas, numpy, scipy stats, scipy.stats kurtosis, scipy.stats skew, matplotlib .
import matplotlib as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew

def EDA(x,df,column_name_string):
    #tables and print lines
    print('Kurtosis: ',kurtosis(x))
    print('Skewness: ',skew(x))
    one = stats.wilcoxon(x)
    two = x.describe()
    
    fig = plt.figure(1)
    #graphs
    gridsize = (2,2)
    plt.subplot2grid(gridsize,(0,0))
    #scatter plot
    plt.title('Scatter plot')
    plt.scatter(final['Key'], x)
    
    plt.subplot2grid(gridsize,(0,1))
    #histogram
    plt.title('Histogram')
    df[column_name_string].hist()

    plt.subplot2grid(gridsize,(1,0))
    #boxplot
    plt.title('Boxplot')
    df.boxplot([column_name_string],grid=False)
    
    plt.subplot2grid(gridsize,(1,1))
    #Probability Plot
    stats.probplot(x, plot=plt)  
    return one, two
