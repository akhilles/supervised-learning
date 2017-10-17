import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def training_data(filename):
    df = pd.read_csv(filename)
    return df

df = training_data('trainingdata.csv')
df_pos = df[df['Class'] == 'P']
df_neg = df[df['Class'] == 'N']

xvals = np.arange(-10,10,0.1)
yvals = -1 * xvals + 4
yvalsu = yvals + 2
yvalsd = yvals - 2

plt.figure(1)
plt.plot(df_pos['X'],df_pos['Y'],'k.',df_neg['X'],df_neg['Y'],'r.')
plt.plot(xvals,yvals,'-',xvals,yvalsu,'--',xvals,yvalsd,'--',color='0.5')
plt.axis('equal')
plt.axis([-1,3,-1,7])


df = training_data('trainingdata_new.csv')
df_pos = df[df['Class'] == 'P']
df_neg = df[df['Class'] == 'N']

xvals = np.arange(-10,10,0.1)
yvals = -1 * xvals + 4
yvalsu = yvals + 2
yvalsd = yvals - 2

plt.figure(2)
plt.plot(df_pos['X'],df_pos['Y'],'k.',df_neg['X'],df_neg['Y'],'r.')
plt.plot(xvals,yvals,'-',xvals,yvalsu,'--',xvals,yvalsd,'--',color='0.5')
plt.axis('equal')
plt.axis([-1,3,-1,7])

plt.show()