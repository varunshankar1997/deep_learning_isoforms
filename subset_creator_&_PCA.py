# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 11:10:00 2022

@author: Varun Shankar
"""

import os 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
 


isoform_df = pd.read_csv(r"D:\gtex_isoform_expression.tsv",sep='\t',nrows=1000)

#isoform_df.to_csv(r"D:\gtex_isoform_expression_subset.tsv", sep="\t")

#isoformdf2 = pd.read_csv(r"C:\Users\Varun Shankar\Downloads\subset.tsv",sep='\t')

#print(isoformdf2.shape)

df_numeric =  isoform_df.select_dtypes(['number'])


scaled_data = preprocessing.scale(df_numeric.T)

x= StandardScaler().fit_transform(scaled_data)

pca= PCA()

principalComponents = pca.fit_transform(x)

#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

#print(pca.explained_variance_ratio_)

# calculating percentage variations
per_var = np.round(pca.explained_variance_ratio_* 100, decimals = 5)
print(sum(pca.explained_variance_ratio_[0:100]))







labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height= per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.title('Scree Plot')
plt.show()

fig = px.scatter(principalComponents,x=0,y=1)
fig.show()
