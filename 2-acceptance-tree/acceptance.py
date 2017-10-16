import numpy as np
import pandas as pd
from math import log2

def applicant_data(filename):
    df = pd.read_csv(filename)
    df.drop('No.', 1, inplace=True)

    gpa_classes = []
    for index, row in df.iterrows():
        gpa = float(row['GPA'])
        if gpa >= 3.9:
            gpa_classes.append("GPA >= 3.9")
        elif gpa > 3.2:
            gpa_classes.append("3.9 > GPA > 3.2")
        else:
            gpa_classes.append("3.2 >= GPA")

    df['GPA Class'] = pd.Series(gpa_classes).values
    return df

def entropy(q):
    if q == 0 or q == 1:
        return 0
    return -(q*log2(q) + (1-q)*log2(1-q))

def information_gain(attr, df):
    gain = 0.0
    grouping = df.groupby('Class')['Class'].count()
    n = grouping['N']
    p = grouping['P']
    gain += entropy(n/(n+p))

    pt = df.pivot_table(index=attr, columns='Class', values='GPA', aggfunc='count', fill_value=0)

    for idx, row in pt.iterrows():
        ni = row['N']
        pi = row['P']
        weight = (ni+pi)/(n+p)
        gain -= weight*entropy(ni/(ni+pi))

    return gain

df = applicant_data('applicants.csv')
print(df)
print(information_gain('GPA Class', df))
print(information_gain('University', df))
#print(df.pivot_table(index='GPA Class', columns='Class', values='GPA', aggfunc='count').head())
#print(entropy(0.01))