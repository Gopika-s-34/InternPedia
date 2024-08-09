import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime as dt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import HTML

df = pd.read_csv('Unemployment_Rate.csv')
df.columns =['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate','Region','longitude','latitude']

df.head()
df.shape
df.info()
df.isnull().sum()

df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Frequency']= df['Frequency'].astype('category')
df['Month'] =  df['Date'].dt.month
df['MonthNumber'] = df['Month'].apply(lambda x : int(x))
df['MonthName'] =  df['MonthNumber'].apply(lambda x: calendar.month_abbr[x])
df['Region'] = df['Region'].astype('category')
df.drop(columns='Month',inplace=True)
df.head(3)
df.describe()

round(df[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']].describe().T,2)
regionStats = df.groupby(['Region'])[['Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate']].mean().reset_index()
round(regionStats,2)
heatMap = df[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate', 'longitude', 'latitude', 'MonthNumber']]
heatMap = heatMap.corr()
plt.figure(figsize=(23,8))
sns.heatmap(heatMap, annot=True,cmap='twilight_shifted', fmt='.3f', linewidths=1)
plt.title('heatMap')
plt.show()

plt.figure(figsize=(12, 6))
sns.regplot(x='Estimated Employed', y='Estimated Unemployment Rate', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
plt.xlabel('Estimated Employed')
plt.ylabel('Estimated Unemployment Rate')
plt.title('Scatter Plot with Regression Line')
plt.show()

df_melted = df.melt(id_vars=['Region'], value_vars=['Estimated Unemployment Rate', 'Estimated Employed'],var_name='Metric', value_name='Value')

plt.figure(figsize=(14, 7))
sns.barplot(x='Region', y='Value', hue='Metric', data=df_melted, palette='Set1')
plt.xlabel('Region')
plt.ylabel('Values')
plt.title('Comparison of Employment and Unemployment Rates by Region')
plt.xticks(rotation=45)
plt.show()
