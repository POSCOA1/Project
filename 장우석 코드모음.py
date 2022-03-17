#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


matplotlib.rc('font',family='Malgun Gothic')
matplotlib.rc('axes', unicode_minus=False)


# In[3]:


df= pd.read_csv('result1.csv',encoding='euc-kr')


# In[6]:


from datetime import datetime

df["start hour"] = 0
df["end hour"] = 0
df["weekday"] = 0

for i in range(len(df)):
  df.loc[i, 'start hour'] = str(df.loc[i, '방송시작시간']).split('T')[1].split(":")[0]
  df.loc[i, 'end hour'] = str(df.loc[i, '방송종료시간']).split('T')[1].split(":")[0]
  date = str(df.loc[i, '방송시작시간']).split("T")[0]
  datetime_date = datetime.strptime(date, '%Y-%m-%d')
  df.loc[i, 'weekday'] = datetime_date.weekday()

df.head()


# In[8]:


df["총판매매출"] = 0
profit = df["상품주문금액"] * df["수수료율"] 
df["총판매매출"] = profit.round(2)

df.head()


# In[9]:


df.fillna('undefined',inplace=True)


# In[10]:


a=df[['PD','SH','TD','총판매매출','부서명']]
a.head(30)


# ## 부서별 총매출 평균과 방송노출 횟수

# In[11]:


b = a[(a['부서명']=='생활1팀')|(a['부서명']=='생활2팀')|(a['부서명']=='생활3팀')|(a['부서명']=='생활4팀')]
c = a[(a['부서명']=='식품1팀')|(a['부서명']=='식품2팀')|(a['부서명']=='식품3팀')|(a['부서명']=='식품4팀')]
d = a[a['부서명']=='무형상품팀']
e = a[a['부서명']=='상생협력팀']


# In[12]:


avg = []
count = []
department = ['생활','식품','무형','상생'] 


# In[13]:


avg.append(b['총판매매출'].mean())
avg.append(c['총판매매출'].mean())
avg.append(d['총판매매출'].mean())
avg.append(e['총판매매출'].mean())


# In[14]:


count.append(len(b))
count.append(len(c))
count.append(len(d))
count.append(len(e))


# In[15]:


df_pd = pd.DataFrame(index=department)
df_pd['avg'] = avg
df_pd['count'] = count
df_pd.round(3)
df_pd.reset_index(inplace=True)


# In[16]:


df_pd


# In[17]:


sns.barplot(data=df_pd,x='index',y='avg')


# In[18]:


fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(7,5))
sns.barplot(data=df_pd,x='index',y='avg',ax=ax[0])
sns.barplot(data=df_pd,x='index',y='count',ax=ax[1])
plt.tight_layout()
plt.show()


# In[19]:


df_max = pd.concat([b[b['총판매매출']==b['총판매매출'].max()],c[c['총판매매출']==c['총판매매출'].max()],
                   d[d['총판매매출']==d['총판매매출'].max()],e[e['총판매매출']==e['총판매매출'].max()]],axis=0)
df_max.head()


# ## 운영진과 총판매매출의 상관관계를 통해 최적 조합 추출

# In[20]:


pd_a=[]
sh_a=[]
td_a=[]


# In[21]:


f=pd.get_dummies(b[['PD','총판매매출']])
f_corr=f.corr()
pd_a.append(f_corr.sort_values('총판매매출',ascending=False).index[2])


# In[22]:


g=pd.get_dummies(b[['SH','총판매매출']])
g_corr=g.corr()
sh_a.append(g_corr.sort_values('총판매매출',ascending=False).index[2])


# In[23]:


h=pd.get_dummies(b[['TD','총판매매출']])
h_corr=h.corr()
td_a.append(h_corr.sort_values('총판매매출',ascending=False).index[2])


# In[24]:


i=pd.get_dummies(c[['PD','총판매매출']])
i_corr=i.corr()
pd_a.append(i_corr.sort_values('총판매매출',ascending=False).index[2])


# In[25]:


j=pd.get_dummies(c[['SH','총판매매출']])
j_corr=j.corr()
sh_a.append(j_corr.sort_values('총판매매출',ascending=False).index[2])


# In[26]:


k=pd.get_dummies(c[['TD','총판매매출']])
k_corr=k.corr()
td_a.append(k_corr.sort_values('총판매매출',ascending=False).index[2])


# In[27]:


l=pd.get_dummies(d[['PD','총판매매출']])
l_corr=l.corr()
pd_a.append(l_corr.sort_values('총판매매출',ascending=False).index[2])


# In[28]:


m=pd.get_dummies(d[['SH','총판매매출']])
m_corr=m.corr()
sh_a.append(m_corr.sort_values('총판매매출',ascending=False).index[2])


# In[29]:


n=pd.get_dummies(d[['TD','총판매매출']])
n_corr=h.corr()
td_a.append(n_corr.sort_values('총판매매출',ascending=False).index[2])


# In[30]:


o=pd.get_dummies(e[['PD','총판매매출']])
o_corr=o.corr()
pd_a.append(o_corr.sort_values('총판매매출',ascending=False).index[2])


# In[31]:


p=pd.get_dummies(e[['SH','총판매매출']])
p_corr=p.corr()
sh_a.append(p_corr.sort_values('총판매매출',ascending=False).index[2])


# In[32]:


q=pd.get_dummies(e[['TD','총판매매출']])
q_corr=q.corr()
td_a.append(q_corr.sort_values('총판매매출',ascending=False).index[2])


# In[36]:


pd.DataFrame({'PD':pd_a,'SH':sh_a,'TD':td_a},index=['생활','식품','무형','상생'])


# ## 방송시작시간, 방송종료시간, 방송요일과 가중분의 관계

# In[53]:


df_box = df[['start hour','end hour','weekday','가중분']]


# In[54]:


df_00=df_box[df_box['start hour']=='00']['가중분'].reset_index(drop=True)
df_01=df_box[df_box['start hour']=='01']['가중분'].reset_index(drop=True)
df_02=df_box[df_box['start hour']=='02']['가중분'].reset_index(drop=True)
df_03=df_box[df_box['start hour']=='03']['가중분'].reset_index(drop=True)
df_04=df_box[df_box['start hour']=='04']['가중분'].reset_index(drop=True)
df_05=df_box[df_box['start hour']=='05']['가중분'].reset_index(drop=True)
df_06=df_box[df_box['start hour']=='06']['가중분'].reset_index(drop=True)
df_07=df_box[df_box['start hour']=='07']['가중분'].reset_index(drop=True)
df_08=df_box[df_box['start hour']=='08']['가중분'].reset_index(drop=True)
df_09=df_box[df_box['start hour']=='09']['가중분'].reset_index(drop=True)
df_10=df_box[df_box['start hour']=='10']['가중분'].reset_index(drop=True)
df_11=df_box[df_box['start hour']=='11']['가중분'].reset_index(drop=True)
df_12=df_box[df_box['start hour']=='12']['가중분'].reset_index(drop=True)
df_13=df_box[df_box['start hour']=='13']['가중분'].reset_index(drop=True)
df_14=df_box[df_box['start hour']=='14']['가중분'].reset_index(drop=True)
df_15=df_box[df_box['start hour']=='15']['가중분'].reset_index(drop=True)
df_16=df_box[df_box['start hour']=='16']['가중분'].reset_index(drop=True)
df_17=df_box[df_box['start hour']=='17']['가중분'].reset_index(drop=True)
df_18=df_box[df_box['start hour']=='18']['가중분'].reset_index(drop=True)
df_19=df_box[df_box['start hour']=='19']['가중분'].reset_index(drop=True)
df_20=df_box[df_box['start hour']=='20']['가중분'].reset_index(drop=True)
df_21=df_box[df_box['start hour']=='21']['가중분'].reset_index(drop=True)
df_22=df_box[df_box['start hour']=='22']['가중분'].reset_index(drop=True)
df_23=df_box[df_box['start hour']=='23']['가중분'].reset_index(drop=True)


# In[55]:


df_total = pd.concat([df_00,df_01,df_02,df_03,df_04,df_05,df_06,df_07,df_08,df_09,df_10,df_11,df_12,df_13,
                     df_14,df_15,df_16,df_17,df_18,df_19,df_20,df_21,df_22,df_23],axis=1)


# In[56]:


df_total.columns=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14',
                  '15','16','17','18','19','20','21','22','23']


# In[57]:


df_total.dropna()


# In[58]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df_total[['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']])


# In[59]:


df_00=df_box[df_box['end hour']=='00']['가중분'].reset_index(drop=True)
df_01=df_box[df_box['end hour']=='01']['가중분'].reset_index(drop=True)
df_02=df_box[df_box['end hour']=='02']['가중분'].reset_index(drop=True)
df_03=df_box[df_box['end hour']=='03']['가중분'].reset_index(drop=True)
df_04=df_box[df_box['end hour']=='04']['가중분'].reset_index(drop=True)
df_05=df_box[df_box['end hour']=='05']['가중분'].reset_index(drop=True)
df_06=df_box[df_box['end hour']=='06']['가중분'].reset_index(drop=True)
df_07=df_box[df_box['end hour']=='07']['가중분'].reset_index(drop=True)
df_08=df_box[df_box['end hour']=='08']['가중분'].reset_index(drop=True)
df_09=df_box[df_box['end hour']=='09']['가중분'].reset_index(drop=True)
df_10=df_box[df_box['end hour']=='10']['가중분'].reset_index(drop=True)
df_11=df_box[df_box['end hour']=='11']['가중분'].reset_index(drop=True)
df_12=df_box[df_box['end hour']=='12']['가중분'].reset_index(drop=True)
df_13=df_box[df_box['end hour']=='13']['가중분'].reset_index(drop=True)
df_14=df_box[df_box['end hour']=='14']['가중분'].reset_index(drop=True)
df_15=df_box[df_box['end hour']=='15']['가중분'].reset_index(drop=True)
df_16=df_box[df_box['end hour']=='16']['가중분'].reset_index(drop=True)
df_17=df_box[df_box['end hour']=='17']['가중분'].reset_index(drop=True)
df_18=df_box[df_box['end hour']=='18']['가중분'].reset_index(drop=True)
df_19=df_box[df_box['end hour']=='19']['가중분'].reset_index(drop=True)
df_20=df_box[df_box['end hour']=='20']['가중분'].reset_index(drop=True)
df_21=df_box[df_box['end hour']=='21']['가중분'].reset_index(drop=True)
df_22=df_box[df_box['end hour']=='22']['가중분'].reset_index(drop=True)
df_23=df_box[df_box['end hour']=='23']['가중분'].reset_index(drop=True)


# In[60]:


df_total = pd.concat([df_00,df_01,df_02,df_03,df_04,df_05,df_06,df_07,df_08,df_09,df_10,df_11,df_12,df_13,
                     df_14,df_15,df_16,df_17,df_18,df_19,df_20,df_21,df_22,df_23],axis=1)


# In[61]:


df_total.columns=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14',
                  '15','16','17','18','19','20','21','22','23']


# In[62]:


df_total.dropna()


# In[63]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df_total[['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']])


# In[64]:


df_00=df_box[df_box['weekday']==0]['가중분'].reset_index(drop=True)
df_01=df_box[df_box['weekday']==1]['가중분'].reset_index(drop=True)
df_02=df_box[df_box['weekday']==2]['가중분'].reset_index(drop=True)
df_03=df_box[df_box['weekday']==3]['가중분'].reset_index(drop=True)
df_04=df_box[df_box['weekday']==4]['가중분'].reset_index(drop=True)
df_05=df_box[df_box['weekday']==5]['가중분'].reset_index(drop=True)
df_06=df_box[df_box['weekday']==6]['가중분'].reset_index(drop=True)


# In[65]:


df_total = pd.concat([df_00,df_01,df_02,df_03,df_04,df_05,df_06],axis=1)


# In[66]:


df_total.columns=['MON','TUE','WED','THU','FRI','SAT','SUN']


# In[67]:


df_total.dropna()


# In[68]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df_total[['MON','TUE','WED','THU','FRI','SAT','SUN']])


# ### PD, SH, TD 조합 클러스터링

# In[92]:


from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


# In[93]:


df_k = df[['PD','SH','TD','총판매매출']]
df_k=df_k[(df_k['PD']!='undefined')&(df_k['SH']!='undefined')&(df_k['TD']!='undefined')].reset_index(drop=True)
df_k_x=df_k[['PD','SH','TD','총판매매출']]
df_k_y=df_k[['총판매매출']]


# In[94]:


df_k_x=pd.get_dummies(df_k_x)


# In[95]:


# 최대 군집 생성 개수
n_iter_cluster = 15
cluster_range = [i+1 for i in range(n_iter_cluster)]
clus_error = []
for v_n_clus in cluster_range:
    clus = KMeans(v_n_clus)
    clus.fit(df_k_x)
    
    # 각 데이터로부터 가장 가까운 군집 중심점까지 거리 제곱합
    clus_error.append(clus.inertia_)
ds_error = pd.DataFrame({"NumberofCluster": cluster_range, "Error": clus_error})
ds_error.round(3)


# In[96]:


plt.figure(figsize=(10,10))
plt.plot(ds_error["NumberofCluster"], ds_error["Error"])
plt.title("Sum of squared distance")
plt.xlabel("Clusters")
plt.ylabel("Sum of squared distance")


# In[97]:


clus = KMeans(6)
clus.fit(df_k_x)
cluster_kmeans = [i+1 for i in clus.labels_]
df_k["ClusterKmeans"] = cluster_kmeans
df_k


# In[98]:


df_k[df_k['ClusterKmeans']==1]


# In[99]:


df_k[df_k['ClusterKmeans']==2]


# In[100]:


df_k[df_k['ClusterKmeans']==3]


# In[101]:


df_k[df_k['ClusterKmeans']==4]


# In[102]:


df_k[df_k['ClusterKmeans']==5]


# In[103]:


df_k[df_k['ClusterKmeans']==6]


# In[104]:


df_k['ClusterKmeans']=df_k['ClusterKmeans'].replace(1,'D').replace(2,'F').replace(3,'C').replace(4,'E').replace(5,'A').replace(6,'B')


# In[105]:


df_k


# In[106]:


A=df_k[df_k['ClusterKmeans']=='A']['총판매매출'].mean()
B=df_k[df_k['ClusterKmeans']=='B']['총판매매출'].mean()
C=df_k[df_k['ClusterKmeans']=='C']['총판매매출'].mean()
D=df_k[df_k['ClusterKmeans']=='D']['총판매매출'].mean()
E=df_k[df_k['ClusterKmeans']=='E']['총판매매출'].mean()
F=df_k[df_k['ClusterKmeans']=='F']['총판매매출'].mean()


# In[107]:


grade = pd.DataFrame({'Grade':[A,B,C,D,E,F]},index=['A','B','C','D','E','F'])
grade.reset_index(inplace=True)


# In[108]:


grade


# In[109]:


sns.barplot(data=grade,x='index',y='Grade')


# In[ ]:


A=df_k[df_k['ClusterKmeans']=='A'].head(1)
B=df_k[df_k['ClusterKmeans']=='B'].head(1)
C=df_k[df_k['ClusterKmeans']=='C'].head(1)
D=df_k[df_k['ClusterKmeans']=='D'].head(1)
E=df_k[df_k['ClusterKmeans']=='E'].head(1)
F=df_k[df_k['ClusterKmeans']=='F'].head(1)


# In[110]:


df_k[df_k['ClusterKmeans']=='A']['총판매매출'].max()


# In[111]:


df_k[df_k['ClusterKmeans']=='A']['총판매매출'].min()


# In[112]:


df_k[df_k['ClusterKmeans']=='B']['총판매매출'].max()


# In[113]:


df_k[df_k['ClusterKmeans']=='B']['총판매매출'].min()


# In[114]:


df_k[df_k['ClusterKmeans']=='C']['총판매매출'].max()


# In[115]:


df_k[df_k['ClusterKmeans']=='C']['총판매매출'].min()


# In[116]:


df_k[df_k['ClusterKmeans']=='D']['총판매매출'].max()


# In[117]:


df_k[df_k['ClusterKmeans']=='D']['총판매매출'].min()


# In[118]:


df_k[df_k['ClusterKmeans']=='E']['총판매매출'].max()


# In[119]:


df_k[df_k['ClusterKmeans']=='E']['총판매매출'].min()


# In[120]:


df_k[df_k['ClusterKmeans']=='F']['총판매매출'].max()


# In[121]:


df_k[df_k['ClusterKmeans']=='F']['총판매매출'].min()


# ### 시간대, 요일별과 총판매매출과의 관계 모델링

# In[123]:


from sklearn.preprocessing import StandardScaler
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[124]:


def root_mean_squared_error(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


# In[126]:


df_r=df[['소요분','가중분','상품소요분','start hour','end hour','weekday','총판매매출','대분류','방송구분','매입과세구분','매입형태']]


# In[127]:


df_r_x=df_r[['소요분', '가중분', '상품소요분', 'start hour', 'end hour', 'weekday','대분류', '방송구분', '매입과세구분', '매입형태']]
df_r_y=df_r['총판매매출']


# In[128]:


df_train_x,df_test_x,df_train_y,df_test_y = train_test_split(df_r_x,df_r_y,test_size=0.2,random_state=1234)


# In[129]:


df_train_x_num=df_train_x[['start hour','end hour','weekday','소요분','가중분','상품소요분']]
df_train_x_cat=df_train_x[['대분류','방송구분','매입과세구분','매입형태']]
df_train_x_cat=pd.get_dummies(df_train_x_cat)

df_test_x_num=df_test_x[['start hour','end hour','weekday','소요분','가중분','상품소요분']]
df_test_x_cat=df_test_x[['대분류','방송구분','매입과세구분','매입형태']]
df_test_x_cat=pd.get_dummies(df_test_x_cat)


# In[130]:


df_train_x_cat, df_test_x_cat = df_train_x_cat.align(df_test_x_cat, join='inner', axis=1)


# In[131]:


df_train_x_cat.reset_index(inplace=True,drop=False)
df_test_x_cat.reset_index(inplace=True,drop=False)


# In[132]:


scaler=StandardScaler()
scaler.fit(df_train_x_num)
train_x_scaled=scaler.transform(df_train_x_num)
test_x_scaled=scaler.transform(df_test_x_num)


# In[133]:


x_cols=['start hour','end hour','weekday','소요분','가중분','상품소요분']


# In[134]:


train_x_scaled=pd.DataFrame(train_x_scaled,columns=x_cols)
test_x_scaled=pd.DataFrame(test_x_scaled,columns=x_cols)


# In[135]:


df_train_x = pd.concat([pd.DataFrame(train_x_scaled), df_train_x_cat], axis=1)
df_test_x = pd.concat([pd.DataFrame(test_x_scaled), df_test_x_cat], axis=1)


# In[136]:


del df_train_x['index']
del df_test_x['index']


# In[137]:


tree_uncustomized = DecisionTreeRegressor(random_state=1234)
tree_uncustomized.fit(df_train_x,df_train_y)
print(tree_uncustomized.score(df_train_x,df_train_y))
print(tree_uncustomized.score(df_test_x,df_test_y))


# In[138]:


rf_pred=tree_uncustomized.predict(df_test_x)


# In[139]:


r2_score(df_test_y,rf_pred)


# In[140]:


r2_score(df_test_y,rf_pred)
mean_squared_error(df_test_y,rf_pred)


# In[141]:


mean_squared_error(df_test_y,rf_pred)


# ## 랜덤포레스트

# In[142]:


rf_uncustomized = RandomForestRegressor(random_state=1234)
rf_uncustomized.fit(df_train_x,df_train_y)
print(rf_uncustomized.score(df_train_x,df_train_y))
print(rf_uncustomized.score(df_test_x,df_test_y))


# In[143]:


train_score=[]
test_score=[]

para_n_tree = [n_tree*10 for n_tree in range(1,11)]

for v_n_estimators in para_n_tree:
    rf = RandomForestRegressor(n_estimators=v_n_estimators,random_state=1234)
    rf.fit(df_train_x,df_train_y)
    train_score.append(rf.score(df_train_x,df_train_y))
    test_score.append(rf.score(df_test_x,df_test_y))
    
df_score_n = pd.DataFrame()
df_score_n['n_estimators']=para_n_tree
df_score_n['TrainScore']=train_score
df_score_n['TestScore']=test_score
df_score_n.round(3)


# In[144]:


plt.figure(figsize=(5,5))
plt.plot(para_n_tree,train_score,linestyle='-',label='TrainScore')
plt.plot(para_n_tree,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[145]:


train_score=[]
test_score=[]

para_leaf = [n_leaf*1 for n_leaf in range(1,21)]

for v_min_samples_leaf in para_leaf:
    rf = RandomForestRegressor(n_estimators=100,min_samples_leaf=v_min_samples_leaf,random_state=1234)
    rf.fit(df_train_x,df_train_y)
    train_score.append(rf.score(df_train_x,df_train_y))
    test_score.append(rf.score(df_test_x,df_test_y))
    
df_score_leaf = pd.DataFrame()
df_score_leaf['MinSamplesLeaf']=para_leaf
df_score_leaf['TrainScore']=train_score
df_score_leaf['TestScore']=test_score
df_score_leaf.round(3)


# In[146]:


plt.figure(figsize=(5,5))
plt.plot(para_leaf,train_score,linestyle='-',label='TrainScore')
plt.plot(para_leaf,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[147]:


train_score=[]
test_score=[]

para_split = [n_split*2 for n_split in range(2,21)]

for v_min_samples_split in para_split:
    rf = RandomForestRegressor(n_estimators=100,min_samples_leaf=17,min_samples_split=v_min_samples_split,random_state=1234)
    rf.fit(df_train_x,df_train_y)
    train_score.append(rf.score(df_train_x,df_train_y))
    test_score.append(rf.score(df_test_x,df_test_y))
    
df_score_split = pd.DataFrame()
df_score_split['MinSamplessplit']=para_split
df_score_split['TrainScore']=train_score
df_score_split['TestScore']=test_score
df_score_split.round(3)


# In[148]:


plt.figure(figsize=(5,5))
plt.plot(para_split,train_score,linestyle='-',label='TrainScore')
plt.plot(para_split,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[149]:


train_score=[]
test_score=[]

para_depth = [depth for depth in range(2,21)]

for v_max_depth in para_depth:
    rf = RandomForestRegressor(n_estimators=100,min_samples_leaf=17,min_samples_split=4,max_depth=v_max_depth,random_state=1234)
    rf.fit(df_train_x,df_train_y)
    train_score.append(rf.score(df_train_x,df_train_y))
    test_score.append(rf.score(df_test_x,df_test_y))
    
df_score_depth = pd.DataFrame()
df_score_depth['MinSamplesdepth']=para_depth
df_score_depth['TrainScore']=train_score
df_score_depth['TestScore']=test_score
df_score_depth.round(3)


# In[150]:


plt.figure(figsize=(5,5))
plt.plot(para_depth,train_score,linestyle='-',label='TrainScore')
plt.plot(para_depth,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[151]:


rf_final = RandomForestRegressor(random_state=1234,n_estimators=100,min_samples_leaf=17,min_samples_split=4,max_depth=9)
rf_final.fit(df_train_x,df_train_y)
print(rf_final.score(df_train_x,df_train_y))
print(rf_final.score(df_test_x,df_test_y))


# In[152]:


rf_final_pred=rf_final.predict(df_test_x)


# In[153]:


r2_score(df_test_y,rf_final_pred)


# In[154]:


mean_squared_error(df_test_y,rf_final_pred)


# ## 그레디언트 부스팅

# In[155]:


gb_uncustomized = GradientBoostingRegressor(random_state=1234)
gb_uncustomized.fit(df_train_x,df_train_y)
print(gb_uncustomized.score(df_train_x,df_train_y))
print(gb_uncustomized.score(df_test_x,df_test_y))


# In[156]:


train_score=[]
test_score=[]

para_n_tree = [n_tree*10 for n_tree in range(1,11)]

for v_n_estimators in para_n_tree:
    gb = GradientBoostingRegressor(n_estimators=v_n_estimators,random_state=1234)
    gb.fit(df_train_x,df_train_y)
    train_score.append(gb.score(df_train_x,df_train_y))
    test_score.append(gb.score(df_test_x,df_test_y))
    
df_score_n = pd.DataFrame()
df_score_n['n_estimators']=para_n_tree
df_score_n['TrainScore']=train_score
df_score_n['TestScore']=test_score
df_score_n.round(3)


# In[157]:


plt.figure(figsize=(5,5))
plt.plot(para_n_tree,train_score,linestyle='-',label='TrainScore')
plt.plot(para_n_tree,test_score,linestyle='--',label='TestScore')
plt.ylabel('score')
plt.xlabel('n_estimators')
plt.legend()


# In[158]:


train_score=[]
test_score=[]

para_leaf = [n_leaf*1 for n_leaf in range(1,21)]

for v_min_samples_leaf in para_leaf:
    gb = GradientBoostingRegressor(n_estimators=90,min_samples_leaf=v_min_samples_leaf,random_state=1234)
    gb.fit(df_train_x,df_train_y)
    train_score.append(gb.score(df_train_x,df_train_y))
    test_score.append(gb.score(df_test_x,df_test_y))
    
df_score_leaf = pd.DataFrame()
df_score_leaf['MinSamplesLeaf']=para_leaf
df_score_leaf['TrainScore']=train_score
df_score_leaf['TestScore']=test_score
df_score_leaf.round(3)


# In[159]:


plt.figure(figsize=(5,5))
plt.plot(para_leaf,train_score,linestyle='-',label='TrainScore')
plt.plot(para_leaf,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[160]:


train_score=[]
test_score=[]

para_split = [n_split*2 for n_split in range(2,21)]

for v_min_samples_split in para_split:
    gb = GradientBoostingRegressor(n_estimators=90,min_samples_leaf=18,min_samples_split=v_min_samples_split,random_state=1234)
    gb.fit(df_train_x,df_train_y)
    train_score.append(gb.score(df_train_x,df_train_y))
    test_score.append(gb.score(df_test_x,df_test_y))
    
df_score_split = pd.DataFrame()
df_score_split['MinSamplessplit']=para_split
df_score_split['TrainScore']=train_score
df_score_split['TestScore']=test_score
df_score_split.round(3)


# In[161]:


plt.figure(figsize=(5,5))
plt.plot(para_split,train_score,linestyle='-',label='TrainScore')
plt.plot(para_split,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[162]:


train_score=[]
test_score=[]

para_depth = [depth for depth in range(2,21)]

for v_max_depth in para_depth:
    gb = GradientBoostingRegressor(n_estimators=90,min_samples_leaf=18,min_samples_split=4,max_depth=v_max_depth,random_state=1234)
    gb.fit(df_train_x,df_train_y)
    train_score.append(gb.score(df_train_x,df_train_y))
    test_score.append(gb.score(df_test_x,df_test_y))
    
df_score_depth = pd.DataFrame()
df_score_depth['MinSamplesdepth']=para_depth
df_score_depth['TrainScore']=train_score
df_score_depth['TestScore']=test_score
df_score_depth.round(3)


# In[163]:


plt.figure(figsize=(5,5))
plt.plot(para_depth,train_score,linestyle='-',label='TrainScore')
plt.plot(para_depth,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[164]:


train_score=[]
test_score=[]

para_lr = [lr*0.1 for lr in range(1,10)]

for v_learning_rate in para_lr:
    gb = GradientBoostingRegressor(n_estimators=90,min_samples_leaf=18,min_samples_split=4,max_depth=10,learning_rate=v_learning_rate,random_state=1234)
    gb.fit(df_train_x,df_train_y)
    train_score.append(gb.score(df_train_x,df_train_y))
    test_score.append(gb.score(df_test_x,df_test_y))
    
df_score_lr = pd.DataFrame()
df_score_lr['MinSamplesdepth']=para_lr
df_score_lr['TrainScore']=train_score
df_score_lr['TestScore']=test_score
df_score_lr.round(3)


# In[165]:


plt.figure(figsize=(5,5))
plt.plot(para_lr,train_score,linestyle='-',label='TrainScore')
plt.plot(para_lr,test_score,linestyle='--',label='TestScore')
plt.legend()


# In[166]:


gb_final = GradientBoostingRegressor(n_estimators=90,min_samples_leaf=18,min_samples_split=4,max_depth=10,learning_rate=0.1,random_state=1234)
gb_final.fit(df_train_x,df_train_y)

print(gb_final.score(df_train_x,df_train_y))
print(gb_final.score(df_test_x,df_test_y))


# In[167]:


gb_final_pred = gb_final.predict(df_test_x)


# In[168]:


r2_score(df_test_y,gb_final_pred)


# In[169]:


mean_squared_error(df_test_y,gb_final_pred)


# ## XGboost

# In[174]:


# xgboost 패키지 불러오기 
from xgboost import XGBRegressor
from xgboost import plot_importance


# In[175]:


xgboost_user= XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 5)
xgboost_user.fit(df_train_x, df_train_y)
xgb_pred_train = xgboost_user.predict(df_train_x)

xgb_pred_test = xgboost_user.predict(df_test_x)

print(xgboost_user.score(df_train_x,df_train_y))
print(xgboost_user.score(df_test_x,df_test_y))


# In[176]:


xgb_stop = XGBRegressor(n_estimators = 300, learning_rate = 0.1 , max_depth = 6)

eval_df = [(df_test_x, df_test_y)]

xgb_stop.fit(df_train_x, df_train_y, early_stopping_rounds = 150, 
                eval_metric="logloss", eval_set = eval_df, verbose=True)


# In[177]:


fig, ax = plt.subplots(figsize=(10, 10))
plot_importance(xgb_stop, ax=ax)


# In[178]:


y_pred=xgboost_user.predict(df_test_x)
y_pred


# In[179]:


r2_score(df_test_y,y_pred)


# In[180]:


mean_squared_error(df_test_y,y_pred)


# In[181]:


y_pred=xgb_stop.predict(df_test_x)
y_pred = pd.DataFrame(y_pred,columns=['predict'])
df_test_y.reset_index(inplace=True,drop=True)
result = pd.concat([y_pred,df_test_y],axis=1)
result


# In[182]:


result.describe()

