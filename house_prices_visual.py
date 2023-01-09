import streamlit as st
import pandas as pd 

import numpy as np 
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats 


@st.cache
def get_data(path):
    df = pd.read_csv(path,index_col='Id')
    return df

@st.cache
def get_data1(path):
    df = pd.read_csv(path,index_col='Unnamed: 0')
    return df

@st.cache
def get_data2(path):
    df = pd.read_csv(path)
    return df

st.title("""
Визуализация данныx по Ames Housing

Анализ корелляции цены дома по его составляющим
""")

df1 = get_data('https://raw.githubusercontent.com/millavasilieva/amos_pricing_analisys/main/train.csv')
df2 = get_data('https://raw.githubusercontent.com/millavasilieva/amos_pricing_analisys/main/test.csv')
df = pd.concat([df1, df2]).reset_index(drop=True)

dfprep = get_data1('https://raw.githubusercontent.com/millavasilieva/amos_pricing_analisys/main/prepared_dataset_house_pricing.csv')



option = st.selectbox ('Выберите тип визуализации',('Целевой признак','Степень корреляции с целевым признаком', 'Построение моделей'))

if option == 'Целевой признак':

    st.header('Гистограмма распределения значений целевого признака')

    fig1=px.histogram(df1,x='SalePrice')
    fig1.update_traces(marker_color='slateblue', 
                        selector=dict(type='histogram'))

    fig1.update_layout(margin=dict(l=5,r=5,b=10,t=10))
    st.write(fig1)

    st.header('Логарифмизация признака')

    fig4 = plt.figure(figsize = (14,8))

    # необработанныt данные
    fig4.add_subplot(1,2,1)
    res = stats.probplot(df1['SalePrice'], plot=plt)

    # мы прологарифмировали 'SalePrice'
    fig4.add_subplot(1,2,2)
    res = stats.probplot(np.log1p(df1['SalePrice']), plot=plt)
    st.write(fig4)

    st.header('После логарифмизации признака')
    fig1=px.histogram(np.log1p(df1['SalePrice']))
    fig1.update_traces(marker_color='slateblue', 
                        selector=dict(type='histogram'))

    fig1.update_layout(margin=dict(l=5,r=5,b=10,t=10))
    st.write(fig1)


if option == 'Степень корреляции с целевым признаком':

    #correlations
    st.markdown("""
    Feature Selection""")

    num_feat=df1.columns[df1.dtypes!=object]
    num_feat=num_feat[1:-1] 
    labels = []
    values = []
    for col in num_feat:
        labels.append(col)
        values.append(np.corrcoef(df1[col].values, df1.SalePrice.values)[0,1])

    ind = np.arange(len(labels))
    width = 0.9
    
    sns.set_color_codes()
    fig2, ax = plt.subplots(figsize=(12,40))
    rects = ax.barh(ind, np.array(values), color='blue')
    ax.set_yticks(ind+((width)/2.))
    ax.set_yticklabels(labels, rotation='horizontal')
    ax.set_xlabel("Коеффициент корреляции")
    ax.set_title("Коеффициент корреляции с  Sale Price")
    st.write(fig2)

    corrMatrix=df[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

    sns.set(font_scale=1.10)
    fig3, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
                square=True,annot=True,cmap='plasma',linecolor="white")
    plt.title('Корреляция между признаками')
    st.write(fig3)

if option == 'Построение моделей':

    train,test = dfprep.loc[:1459], dfprep.loc[1460:].drop(['SalePrice'], axis=1)
    train = train[['OverallQual', 'GrLivArea', '1stFlrSF', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'YearBuilt', 'CentralAir', 'LotArea',
             'OverallCond', 'YearRemodAdd', 'MSZoning', 'GarageFinish', '2ndFlrSF', 'GarageType', 'GarageYrBlt', 'LotFrontage', 'Neighborhood', 'Fireplaces',
                  'OpenPorchSF', 'MoSold', 'TotRmsAbvGrd', 'WoodDeckSF', 'MasVnrArea','SalePrice']]

    x, y = train.loc[:, train.columns != 'SalePrice'],train['SalePrice']
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=42, test_size=0.25)


    st.header('Настройка гиперпараметров')
    st.markdown("""
    ### Gradient Boosting
    """)
    st.write("""Results from Grid Search 

    The best parameters across ALL searched params:
    {'learning_rate': 0.04, 'max_depth': 4, 'n_estimators': 500} """)

    st.markdown("""
    ### CatBoost
    """)
    st.write("""Results from Grid Search 

    The best parameters across ALL searched params:
    {'depth': 6, 'learning_rate': 0.03, 'n_estimators': 1000} """)

    st.markdown("""
    ### Берем среднее полученнных моделей
    """)

    sub = get_data2('https://raw.githubusercontent.com/millavasilieva/ames_pricing_analisys/main/final1_sub.csv')  
    st.write(sub)

    st.markdown("""
    ### Лучший результат = 0.13185,, место - 1216
    """)
    

    