import numpy as np
import pandas as pd
from math import log2
from treelib import Node, Tree

tree = Tree()

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
    n = len(df[df['Class'] == 'N'])
    p = len(df[df['Class'] == 'P'])
    gain += entropy(n/(n+p))

    for val in df[attr].unique():
        ni = len(df[(df[attr] == val) & (df['Class'] == 'N')])
        pi = len(df[(df[attr] == val) & (df['Class'] == 'P')])
        weight = (ni+pi)/(n+p)
        gain -= weight*entropy(ni/(ni+pi))

    return gain

def build_tree(df, parent, label):
    column_names = list(df)
    column_names.remove('Class')
    column_names.remove('GPA')
    new_label = parent + ' (' + label + ') - '
    printable = ' (' + label + ')'
    
    if len(column_names) == 0 or len(df) == 0:
        return
    
    if len(df[df['Class'] == 'N']) == 0:
        tree.create_node('P' + printable, new_label + 'P', parent=parent)
        return
    elif len(df[df['Class'] == 'P']) == 0:
        tree.create_node('N' + printable, new_label + 'N', parent=parent)
        return

    info_gains = [information_gain(attr, df) for attr in column_names]
    best_attr = column_names[info_gains.index(max(info_gains))]

    if parent == 'root':
        tree.create_node(best_attr, new_label + best_attr)
    else:
        tree.create_node(best_attr + printable, new_label + best_attr, parent=parent)

    index = 1
    for val in df[best_attr].unique():
        df_new = df[df[best_attr] == val].drop(best_attr, 1)
        build_tree(df_new, new_label + best_attr, val)
        index += 1
        

df = applicant_data('applicants.csv')

build_tree(df, 'root', '')
print(tree)
