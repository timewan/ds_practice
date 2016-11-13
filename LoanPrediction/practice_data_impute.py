# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:31:54 2016

@author: TimeWanes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\TimeWanes\\Documents\\Python\\Data Science\\LoanPrediction\\train_u6.csv')



# Data Munging

df['Self_Employed'].fillna('No',inplace=True)

table1 = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
    return table1.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['Self_Employed'].fillna('No',inplace=True)
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

# Assume NaN for Dependents is 0
df['Dependents'].fillna('0', inplace=True)
df['Credit_History'].fillna(1.0,inplace=True)
df['Gender'].fillna('Male',inplace=True)
df['Loan_Amount_Term'].fillna(360.0,inplace=True)

#MArried
df['Coapp'] = df['CoapplicantIncome'].gt(0)
def gage(x):
    if x['Coapp']:
        return 'Yes'
    else :
        return 'No'
df['Married'].fillna(df[df['Married'].isnull()].apply(gage, axis =1), inplace=True)            




# Create total income category
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

#df['LoanAmount_log'] = np.log(df['LoanAmount'])
#df['LoanAmount_log'].hist(bins=20)

df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20) 


### Data has been imputed 

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes 