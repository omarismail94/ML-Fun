import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks")

df = pd.read_csv('College.csv',index_col=0)



df["Elite"] = pd.cut(df['Top10perc'],[0,50,100],labels=["No","Yes"])


sns.pairplot(df.iloc[:,0:5], hue='Private')
# sns.boxplot(x=df['Private'],y=df['Outstate'])
# sns.boxplot(x=df['Elite'],y=df['Outstate'])

plt.show()