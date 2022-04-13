import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/Khurram Minhas/OneDrive/Desktop/Fetal_datafetal_health.csv')
df.head(5)
df.info()
df.shape
X = df['fetal_health’].value_counts()']
labels = ['Normal','Suspect', 'Pathological']
plt.get_cmap('hsv')
plt.figure(figsize = (14,7))
plt.title('Distribution of Fetal Health', fontsize = 17)
colors = sns.color_palette('pastel')[0:5]
plt.pie(X, labels = labels, autopct='%.0f%%', colors = colors, explode=[0,0,.3])
plt.legend()
plt.show()
corrmat= df.corr()
plt.figure(figsize=(15,15))  

cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)

sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)
sns.lmplot(data =df,x="accelerations",y="fetal_movement",palette=colors, hue="fetal_health",legend_out=False)
plt.show()
sns.lmplot(data =df,x="prolongued_decelerations",y="fetal_movement",palette=colors, hue="fetal_health",legend_out=False)
plt.show()
sns.lmplot(data =df,x="abnormal_short_term_variability",y="fetal_movement",palette=colors, hue="fetal_health",legend_out=False)
plt.show()
obj_df = df.select_dtypes(exclude=['object']).copy()
FS_Cols=obj_df.columns
FS_Cols
X1=df[FS_Cols]
y=df.fetal_health
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(score_func=mutual_info_classif, k='all')
 


Xfs = selector.fit_transform(X1,y)
names = X1.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'Mutual_info'])
ns_df_sorted = ns_df.sort_values(['Mutual_info', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)
plt.figure(figsize=(20,10))
plt.xticks(rotation = 90)
sns.boxplot(data=df)
plt.show()
from sklearn.utils import resample
 

df_majority = df[df.fetal_health==1.0]
df_minority = df[df.fetal_health==2.0]
df_minority_2 = df[df.fetal_health==3.0]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=1655,    
                                 random_state=123) 
 

df_minority_upsampled_2 = resample(df_minority_2, 
                                 replace=True,     
                                 n_samples=1655,    
                                 random_state=123) 
 

df_upsampled = pd.concat([df_majority, df_minority_upsampled, df_minority_upsampled_2])
 

print(df_upsampled.describe())
 
df_upsampled.fetal_health.value_counts()
 

df = df_upsampled
sns.countplot(x= 'fetal_health', data=df_upsampled)
X = X1.loc[:, ~X1.columns.isin(['fetal_health'])]
y = X1["fetal_health"] 
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1, stratify=y)
 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import RandomTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
 
pipe_dt=Pipeline([('scl',StandardScaler()),
                 ('pca',PCA(n_components=2))
                 ])
x_train = pipe_dt.fit_transform(X_train)
x_test = pipe_dt.fit_transform(X_test)
print(X_train.shape, y_train.shape, X_test.shape)
no_tune = DecisionTreeClassifier(random_state=1024)
no_tune.fit(x_train, y_train)
y_pred = no_tune.predict(x_test)
 
print(classification_report(y_test, no_tune.predict(x_test)))

cnf_matrix = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cnf_matrix,
                     index = ['Normal', 'Suspect','Pathological'], 
                     columns = ['Normal', 'Suspect','Pathological'])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tick_marks = [0.5, 1.5]
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [2, 3, 5, 10, 20, 25],
              'min_samples_leaf': [5, 10, 20, 50, 100],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier(random_state=1024)
grid_search = GridSearchCV(estimator=tree_clas,
param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)
 
print(classification_report(y_test, best_estimator.predict(X_test)))
no_tune = RandomForestClassifier(random_state=1024)
no_tune.fit(x_train, y_train)
y_pred = no_tune.predict(x_test)
 
print(classification_report(y_test, no_tune.predict(x_test)))
param_grid = {'max_depth': [5, 10, 20, 25],
              'min_samples_leaf': [10, 20, 50, 100],
              'criterion' :['gini', 'entropy']
             }
tree_clas = RandomForestClassifier(random_state=1024)
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_train, y_train)
 
best_estimator = grid_search.best_estimator_
 
y_pred = best_estimator.predict(X_test)
 
print(classification_report(y_test, best_estimator.predict(X_test)))
clf = RandomForestClassifier(n_estimators = 50, 
criterion='entropy', max_depth=20, min_samples_leaf=10,
                       random_state=1024)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
 
print(classification_report(y_test, best_estimator.predict(X_test)))

n_estimators = 100

clf = RandomForestClassifier(n_estimators = 100,
criterion='entropy', max_depth=20, min_samples_leaf=10,
                       random_state=1024)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
 
print(classification_report(y_test, best_estimator.predict(X_test)))
