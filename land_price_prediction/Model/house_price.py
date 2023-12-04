#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import warnings as w
w.simplefilter(action='ignore')


# In[81]:


df1 = pd.read_csv("Bengaluru_House_Data.csv")
# df1.head(5)


# In[82]:


# df1.describe()


# # cleaning

# In[ ]:





# In[83]:


# df1.shape


# In[84]:


df1.groupby('area_type')['area_type'].agg('count')


# In[ ]:





# In[ ]:





# In[85]:


df2 = df1.drop(['area_type','availability','society','balcony'],axis='columns')
# df2.head()


# In[86]:


# df2.shape


# In[87]:


df2.isnull().sum()


# In[ ]:





# In[88]:


df3 = df2.dropna()
df3.isnull().sum()


# In[89]:


# df3.shape


# In[90]:


df3['size'].unique()


# In[91]:


df3['bhk']= df3['size'].apply (lambda x: int(x.split(' ')[0]))


# In[92]:


# df3.head()


# In[ ]:





# In[93]:


df3['bhk'].unique()


# In[94]:


df3[df3.bhk>20]


# In[95]:


df3['total_sqft'].unique()


# In[96]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[97]:


df3[df3['total_sqft'].apply(is_float)].head(10)


# In[98]:


def convert_sqft_range_to_num(x):
    tokens = x.split('-') 
    if len(tokens)==2:
        return (float(tokens[0])) + float( (tokens[1] ))/2
    try:
        return float(x)
    except:
        return None


# In[99]:


df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_range_to_num)
# df4.head()


# In[100]:


# print(df4.loc[30])
# df3.loc[30]


# In[ ]:





# In[ ]:





# In[101]:


# stage 3
# feature enginering / dimentionalty reduction technique


# In[102]:


# df4.head(2)


# In[103]:


df5=df4.copy()
df5['price_per_sqft']= df5['price']*100000/df5['total_sqft']
# df5.head()


# In[104]:


# len(df5.location.unique())


# In[105]:


# 1304 diffrent loc. can lead to dimentianalty curse.


# In[106]:


df5.location = df5.location.apply(lambda x: x.strip())

loction_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
# loction_stats


# In[107]:


# len(loction_stats[loction_stats<=10])


# In[108]:


# these 1052 loc. are repeating less ferquently (<=10) so we will now combine them as one category


# In[109]:


loction_stats_less_than_10 = loction_stats[loction_stats<=10]
# loction_stats_less_than_10


# In[110]:


df5.location = df5.location.apply(lambda x : 'others' if x in loction_stats_less_than_10 else x)


# In[111]:


# len(df5.location.unique())


# In[112]:


# df5.head(10)


# In[ ]:





# In[113]:


# df5.info()


# In[114]:


# 4 . outlier detection and removal


# In[115]:


plt.scatter(df5['total_sqft'],df5['price'])
# plt.show()


# In[ ]:





# In[116]:


import seaborn as sns


# In[117]:


sns.boxplot(df5['total_sqft'])


# In[118]:


plt.boxplot(df5['total_sqft'])


# In[119]:


# df5[df5.total_sqft/df5.bhk < 300].head()


# In[120]:


# area of an ideal room can not be less than 300 sqft. so we will remove these rows


# In[121]:


# df5[df5.total_sqft/df5.bhk < 300].shape


# In[122]:


# df5.shape


# In[123]:


df6 = df5[~(df5.total_sqft/df5.bhk < 300)]
# df6.shape


# In[ ]:





# In[124]:


# df6.price_per_sqft.describe()


# In[125]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st) ) & (subdf.price_per_sqft < (m+st) )]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out

df7 = remove_pps_outliers(df6)
# df7.shape


# In[126]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='red',label = '2bhk',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='green',marker='^',label = '3bhk',s=50)
    plt.xlabel('Total square feet area')
    plt.ylabel('Price per square feet')
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Hebbal")


# In[ ]:





# In[127]:


def remobe_bhk_outliers(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]= {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats =bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices= np.append(exclude_indices , bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 =remobe_bhk_outliers(df7)
# df8.shape


# In[128]:


plot_scatter_chart(df8,"Hebbal")


# In[129]:


plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel('price per sqft')
plt.ylabel('count')


# In[130]:


# df8.head()


# In[131]:


df8['bath'].unique()


# In[132]:


df8[df8['bath']>10]


# In[133]:


plt.hist(df8['bath'] , rwidth=0.8)
plt.xlabel('count')
plt.ylabel('number of bathrooms')


# In[134]:


df8[df8.bath>df8.bhk+2]


# In[135]:


df9 = df8[~(df8.bath>df8.bhk+2)] # or df9 = df8[df8.bath < df8.bhk+2]
# df9.shape


# In[136]:


df10=df9.drop(['size','price_per_sqft'],axis='columns')
# df10.head()


# In[137]:


# df10.shape


# In[138]:


# 5 .  Model bulding


# In[139]:


# hot end coding


# In[140]:


dummies = pd.get_dummies(df10.location)
# dummies.head()


# In[141]:


df11 = pd.concat([df10,dummies.drop('others',axis='columns')],axis='columns')
# df11.head(3)


# In[142]:


df12 = df11.drop('location',axis='columns')
# df12.head(2)


# In[143]:


# df12.shape


# In[144]:


x = df12.drop(['price'],axis='columns')
# x.head(3)


# In[145]:


y = df12.price
# y.head(3)


# In[ ]:





# In[146]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# In[147]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)


# In[ ]:





# In[148]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5 , test_size=0.2,random_state=3)

cross_val_score(LinearRegression(),x,y,cv=cv)


# In[149]:


from sklearn.model_selection import KFold,StratifiedKFold

kf = KFold(5 , shuffle=True ,random_state=1,)

cross_val_score(lr_clf,x,y,cv=kf,scoring="r2")


# In[150]:


# sklearn.metrics.scorers.keys


# In[151]:


# skf = StratifiedKFold(5 , shuffle=True ,random_state=1)

# cross_val_score(lr_clf,x,y,cv=skf)

#
# we get error because StratifiedKFold works for classification problems
#


# In[ ]:





# In[152]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        
#          'Gradient_Boosting_Regressor' : {
#             'model':GradientBoostingRegressor(),
#             'params': {
#                 'n_estimators': [50, 100, 200],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'max_depth': [3, 4, 5]
#             } 
#         },
        
#         'xGB_Regressor' : {
#             'model':xgb.XGBRegressor(),
#             'params': {
#                 'n_estimators': [50, 100, 200],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'max_depth': [3, 5, 7]
#             } 
#         },
        
#         'Random_Forest_Regressor': {
#             'regressor': RandomForestRegressor(),
#             'param_grid': {
#                 'n_estimators': [50, 100, 200],
#                 'max_depth': [None, 10, 20],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]
#             }
#         },
        
        
        'linear_regression' : {
            'model':LinearRegression(),
            'params': {
                'positive':[True,False],
                'fit_intercept':[True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params': {
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decision_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter' :['best','random']
            }
        }
    }
    scores=[]
    cv = ShuffleSplit(n_splits = 5, test_size=0.2,random_state=0)
    
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'],config['params'],cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)


# In[153]:


# linear regression is the best model for the prediction.


# In[154]:


# x.columns


# In[155]:


loc_index = np.where(x.columns == "1st Phase JP Nagar")[0][0]
# loc_index


# In[156]:


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns == location)[0][0]
    xx=np.zeros(len(x.columns))
    xx[0]=sqft
    xx[1]=bath
    xx[2]=bhk
    if loc_index >= 0:
        xx[loc_index] = 1
    return lr_clf.predict([xx])[0]


# In[157]:


# predict_price("1st Phase JP Nagar",1000,2,2)


# In[158]:


# predict_price("1st Phase JP Nagar",1000,2,5)


# In[159]:


# predict_price("Indira Nagar",1000,2,2)


# In[160]:


# predict_price("Indira Nagar",1000,3,1)


# In[161]:


# creating and exporting pickel file
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[162]:


# creating and exporting json file
import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open ("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[163]:


# 6 . writing python flask server /  stremlet .


# In[164]:


print("Hello world")


# In[166]:


# lr_clf.save('your_model.h5')


# In[ ]:


# !pip install streamlit


# In[168]:


import pickle
import json
import numpy as np
import streamlit as st

# Load the model
model_path = 'banglore_home_prices_model.pickle'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load columns
columns_path = 'columns.json'
with open(columns_path, 'r') as f:
    columns = json.load(f)['data_columns']

def predict_price(location, sqft, bath, bhk):
    loc_index = columns.index(location.lower())
    input_data = np.zeros(len(columns))
    input_data[0] = sqft
    input_data[1] = bath
    input_data[2] = bhk
    if loc_index >= 0:
        input_data[loc_index] = 1
    return lr_clf.predict([input_data])[0]

def main():
    st.title("Bengaluru House Price Prediction")

    location = st.selectbox("Select Location", columns[3:])  # Exclude first three columns (sqft, bath, bhk)
    sqft = st.slider("Total Square Feet", 100, 10000, 1500)
    bath = st.slider("Number of Bathrooms", 1, 10, 2)
    bhk = st.slider("Number of Bedrooms (BHK)", 1, 10, 2)

    if st.button("Predict Price"):
        result = predict_price(location, sqft, bath, bhk)
        st.success(f"The predicted price for the house is {result:.2f} Lakhs INR")

if __name__ == "__main__":
    main()


# In[ ]:




