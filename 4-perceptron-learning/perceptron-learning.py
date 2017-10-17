import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def sample_data(filename):
    df = pd.read_csv(filename)
    return df

def plot_line(w, iter):
    xvals = np.arange(-1,2,0.1)
    m = -w[1]/w[2]
    yvals = xvals * m - w[0]/w[2]
    plt.plot(xvals,yvals,color='blue',alpha=(iter/100))

def plot_line_2(w, iter):
    x = -w[0]/w[1]
    xvals = [x,x]
    yvals = [-1,1]
    plt.plot(xvals,yvals,color='blue',alpha=(iter/100))

def evaluate(row, w):
    prod = w[0] + w[1]*row['X'] + w[2]*row['Y']
    if prod >= 0:
        prod = 1
    else:
        prod = 0
    return prod == row['Class']

def adjust(w, row, a):
    weight = a*(2*row['Class'] - 1)

    w[0] = w[0] + weight*1
    w[1] = w[1] + weight*row['X']
    w[2] = w[2] + weight*row['Y']

w = [0.2,1,-1]
df = sample_data('samples.csv')
df_pos = df[df['Class'] == 1]
df_neg = df[df['Class'] == 0]
plt.figure(1)

for i in range(100):
    plot_line(w, i)
    misclassified = []
    for index, row in df.iterrows():
        if evaluate(row, w) != True:
            misclassified.append(row)
    
    if len(misclassified) == 0:
        break

    adjust(w, random.choice(misclassified), 0.2)

plot_line(w, 100)
plt.axis([0,1.2,0,1.2])
plt.plot(df_pos['X'],df_pos['Y'],'k.',df_neg['X'],df_neg['Y'],'r.')
plt.gca().set_aspect('equal', adjustable='box')


w = [-0.2,1,0]
df = sample_data('samples_new.csv')
df_pos = df[df['Class'] == 1]
df_neg = df[df['Class'] == 0]
plt.figure(2)

for i in range(100):
    plot_line_2(w, i)
    misclassified = []
    for index, row in df.iterrows():
        if evaluate(row, w) != True:
            misclassified.append(row)
    
    if len(misclassified) == 2:
        break

    adjust(w, random.choice(misclassified), 0.2)

plot_line_2(w, 100)
plt.axis([0,1,-0.2,0.2])
plt.plot(df_pos['X'],df_pos['Y'],'k.',df_neg['X'],df_neg['Y'],'r.')

plt.show()