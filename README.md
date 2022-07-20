
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv')
df.head()
CustomerId	Surname	CreditScore	Geography	Gender	Age	Tenure	Balance	Num Of Products	Has Credit Card	Is Active Member	Estimated Salary	Churn
0	15634602	Hargrave	619	France	Female	42	2	0.00	1	1	1	101348.88	1
1	15647311	Hill	608	Spain	Female	41	1	83807.86	1	0	1	112542.58	0
2	15619304	Onio	502	France	Female	42	8	159660.80	3	1	0	113931.57	1
3	15701354	Boni	699	France	Female	39	1	0.00	2	0	0	93826.63	0
4	15737888	Mitchell	850	Spain	Female	43	2	125510.82	1	1	1	79084.10	0
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 13 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   CustomerId        10000 non-null  int64  
 1   Surname           10000 non-null  object 
 2   CreditScore       10000 non-null  int64  
 3   Geography         10000 non-null  object 
 4   Gender            10000 non-null  object 
 5   Age               10000 non-null  int64  
 6   Tenure            10000 non-null  int64  
 7   Balance           10000 non-null  float64
 8   Num Of Products   10000 non-null  int64  
 9   Has Credit Card   10000 non-null  int64  
 10  Is Active Member  10000 non-null  int64  
 11  Estimated Salary  10000 non-null  float64
 12  Churn             10000 non-null  int64  
dtypes: float64(2), int64(8), object(3)
memory usage: 1015.8+ KB
df.duplicated('CustomerId').sum()
0
df = df.set_index('CustomerId')
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 15634602 to 15628319
Data columns (total 12 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Surname           10000 non-null  object 
 1   CreditScore       10000 non-null  int64  
 2   Geography         10000 non-null  object 
 3   Gender            10000 non-null  object 
 4   Age               10000 non-null  int64  
 5   Tenure            10000 non-null  int64  
 6   Balance           10000 non-null  float64
 7   Num Of Products   10000 non-null  int64  
 8   Has Credit Card   10000 non-null  int64  
 9   Is Active Member  10000 non-null  int64  
 10  Estimated Salary  10000 non-null  float64
 11  Churn             10000 non-null  int64  
dtypes: float64(2), int64(7), object(3)
memory usage: 1015.6+ KB
df['Geography'].value_counts()
France     5014
Germany    2509
Spain      2477
Name: Geography, dtype: int64
df.replace({'Geography': {'France': 2, 'Germany': 1, 'Spain': 0}}, inplace=True)
df['Gender'].value_counts()
Male      5457
Female    4543
Name: Gender, dtype: int64
df.replace({'Gender': {'Male': 0, 'Female': 1}}, inplace=True)
df['Num Of Products'].value_counts()
1    5084
2    4590
3     266
4      60
Name: Num Of Products, dtype: int64
df.replace({'Num of Products': {1: 0, 2: 1, 3: 1, 4: 1}}, inplace=True)
df['Has Credit Card'].value_counts()
1    7055
0    2945
Name: Has Credit Card, dtype: int64
df['Is Active Member'].value_counts()
1    5151
0    4849
Name: Is Active Member, dtype: int64
df.loc[(df['Balance']==0), 'Churn'].value_counts()
0    3117
1     500
Name: Churn, dtype: int64
df['Zero Balance'] = np.where(df['Balance']>0,1,0)
df['Zero Balance'].hist()
<matplotlib.axes._subplots.AxesSubplot at 0x7f7514b3c410>

df.groupby(['Churn', 'Geography']).count()
Surname	CreditScore	Gender	Age	Tenure	Balance	Num Of Products	Has Credit Card	Is Active Member	Estimated Salary	Zero Balance
Churn	Geography											
0	0	2064	2064	2064	2064	2064	2064	2064	2064	2064	2064	2064
1	1695	1695	1695	1695	1695	1695	1695	1695	1695	1695	1695
2	4204	4204	4204	4204	4204	4204	4204	4204	4204	4204	4204
1	0	413	413	413	413	413	413	413	413	413	413	413
1	814	814	814	814	814	814	814	814	814	814	814
2	810	810	810	810	810	810	810	810	810	810	810
df.columns
Index(['Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
       'Balance', 'Num Of Products', 'Has Credit Card', 'Is Active Member',
       'Estimated Salary', 'Churn', 'Zero Balance'],
      dtype='object')
x = df.drop(['Surname', 'Churn'], axis = 1)
y = df['Churn']
x.shape, y.shape
((10000, 11), (10000,))
df['Churn'].value_counts()
0    7963
1    2037
Name: Churn, dtype: int64
sns.countplot(x = 'Churn', data = df)
<matplotlib.axes._subplots.AxesSubplot at 0x7f7514a8e210>

x.shape, y.shape
((10000, 11), (10000,))
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 122529)
x_rus, y_rus = rus.fit_resample(x,y)
x_rus.shape, y_rus.shape, x.shape, y.shape
((4074, 11), (4074,), (10000, 11), (10000,))
y.value_counts()
0    7963
1    2037
Name: Churn, dtype: int64
y_rus.value_counts()
0    2037
1    2037
Name: Churn, dtype: int64
y_rus.plot(kind = 'hist')
<matplotlib.axes._subplots.AxesSubplot at 0x7f7512938bd0>

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = 122529)
x_ros, y_ros = ros.fit_resample(x,y)
x_ros.shape, y_ros.shape, x.shape, y.shape
((15926, 11), (15926,), (10000, 11), (10000,))
y.value_counts()
0    7963
1    2037
Name: Churn, dtype: int64
y_ros.value_counts()
1    7963
0    7963
Name: Churn, dtype: int64
y_ros.plot(kind = 'hist')
<matplotlib.axes._subplots.AxesSubplot at 0x7f75128ba690>

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 122529)
x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size = 0.3, random_state = 122529)
x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size = 0.3, random_state = 122529)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[['CreditScore', 'Age','Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(x_train[['CreditScore','Age', 'Tenure', 'Balance', 'Estimated Salary']])
x_test[['CreditScore', 'Age','Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(x_test[['CreditScore','Age', 'Tenure', 'Balance', 'Estimated Salary']])
x_train_rus[['CreditScore', 'Age','Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(x_train_rus[['CreditScore','Age', 'Tenure', 'Balance', 'Estimated Salary']])
x_test_rus[['CreditScore', 'Age','Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(x_test_rus[['CreditScore','Age', 'Tenure', 'Balance', 'Estimated Salary']])
x_train_ros[['CreditScore', 'Age','Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(x_train_ros[['CreditScore','Age', 'Tenure', 'Balance', 'Estimated Salary']])
x_test_ros[['CreditScore', 'Age','Tenure', 'Balance', 'Estimated Salary']] = sc.fit_transform(x_test_ros[['CreditScore','Age', 'Tenure', 'Balance', 'Estimated Salary']])
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
SVC()
y_pred = svc.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, y_pred)
array([[2335,   42],
       [ 408,  215]])
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.85      0.98      0.91      2377
           1       0.84      0.35      0.49       623

    accuracy                           0.85      3000
   macro avg       0.84      0.66      0.70      3000
weighted avg       0.85      0.85      0.82      3000

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10], 'gamma':[1,0.1,0.01], 'kernel':['rbf'], 'class_weight':['balanced']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 2)
grid.fit(x_train,y_train)
Fitting 2 folds for each of 9 candidates, totalling 18 fits
[CV] END ..C=0.1, class_weight=balanced, gamma=1, kernel=rbf; total time=   1.8s
[CV] END ..C=0.1, class_weight=balanced, gamma=1, kernel=rbf; total time=   1.8s
[CV] END C=0.1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   1.3s
[CV] END C=0.1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   1.7s
[CV] END C=0.1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   2.9s
[CV] END C=0.1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   2.8s
[CV] END ....C=1, class_weight=balanced, gamma=1, kernel=rbf; total time=   2.8s
[CV] END ....C=1, class_weight=balanced, gamma=1, kernel=rbf; total time=   3.5s
[CV] END ..C=1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   3.0s
[CV] END ..C=1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   2.1s
[CV] END .C=1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   3.3s
[CV] END .C=1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   2.6s
[CV] END ...C=10, class_weight=balanced, gamma=1, kernel=rbf; total time=   3.4s
[CV] END ...C=10, class_weight=balanced, gamma=1, kernel=rbf; total time=   2.8s
[CV] END .C=10, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   1.3s
[CV] END .C=10, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   1.0s
[CV] END C=10, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   1.2s
[CV] END C=10, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   1.1s
GridSearchCV(cv=2, estimator=SVC(),
             param_grid={'C': [0.1, 1, 10], 'class_weight': ['balanced'],
                         'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']},
             verbose=2)
print(grid.best_estimator_)
SVC(C=10, class_weight='balanced', gamma=1)
grid_predictions = grid.predict(x_test)
confusion_matrix(y_test,grid_predictions)
array([[2164,  213],
       [ 395,  228]])
print(classification_report(y_test,grid_predictions))
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      2377
           1       0.52      0.37      0.43       623

    accuracy                           0.80      3000
   macro avg       0.68      0.64      0.65      3000
weighted avg       0.78      0.80      0.78      3000

svc_rus = SVC()
svc_rus.fit(x_train_rus, y_train_rus)
SVC()
y_pred_rus = svc_rus.predict(x_test_rus)
confusion_matrix(y_test_rus, y_pred_rus)
array([[478, 138],
       [169, 438]])
print(classification_report(y_test_rus, y_pred_rus))
              precision    recall  f1-score   support

           0       0.74      0.78      0.76       616
           1       0.76      0.72      0.74       607

    accuracy                           0.75      1223
   macro avg       0.75      0.75      0.75      1223
weighted avg       0.75      0.75      0.75      1223

param_grid = {'C':[0.1,1,10],'gamma':[1,0.1,0.01],'kernel':['rbf'],'class_weight':['balanced']}
grid_rus = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv =2)
grid_rus.fit(x_train_rus,y_train_rus)
Fitting 2 folds for each of 9 candidates, totalling 18 fits
[CV] END ..C=0.1, class_weight=balanced, gamma=1, kernel=rbf; total time=   0.3s
[CV] END ..C=0.1, class_weight=balanced, gamma=1, kernel=rbf; total time=   0.3s
[CV] END C=0.1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   0.2s
[CV] END C=0.1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   0.2s
[CV] END C=0.1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   0.3s
[CV] END C=0.1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   0.3s
[CV] END ....C=1, class_weight=balanced, gamma=1, kernel=rbf; total time=   0.3s
[CV] END ....C=1, class_weight=balanced, gamma=1, kernel=rbf; total time=   0.3s
[CV] END ..C=1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   0.2s
[CV] END ..C=1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   0.2s
[CV] END .C=1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   0.2s
[CV] END .C=1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   0.2s
[CV] END ...C=10, class_weight=balanced, gamma=1, kernel=rbf; total time=   0.3s
[CV] END ...C=10, class_weight=balanced, gamma=1, kernel=rbf; total time=   0.3s
[CV] END .C=10, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   0.2s
[CV] END .C=10, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   0.2s
[CV] END C=10, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   0.2s
[CV] END C=10, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   0.2s
GridSearchCV(cv=2, estimator=SVC(),
             param_grid={'C': [0.1, 1, 10], 'class_weight': ['balanced'],
                         'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']},
             verbose=2)
print(grid_rus.best_estimator_)
SVC(C=1, class_weight='balanced', gamma=0.1)
grid_predictions_rus = grid_rus.predict(x_test_rus)
confusion_matrix(y_test_rus,grid_predictions_rus)
array([[478, 138],
       [171, 436]])
print(classification_report(y_test_rus,grid_predictions_rus))
              precision    recall  f1-score   support

           0       0.74      0.78      0.76       616
           1       0.76      0.72      0.74       607

    accuracy                           0.75      1223
   macro avg       0.75      0.75      0.75      1223
weighted avg       0.75      0.75      0.75      1223

svc_ros = SVC()
svc_ros.fit(x_train_ros, y_train_ros)
SVC()
y_pred_ros = svc_ros.predict(x_test_ros)
confusion_matrix(y_test_ros, y_pred_ros)
array([[1942,  434],
       [ 526, 1876]])
print(classification_report(y_test_ros, y_pred_ros))
              precision    recall  f1-score   support

           0       0.79      0.82      0.80      2376
           1       0.81      0.78      0.80      2402

    accuracy                           0.80      4778
   macro avg       0.80      0.80      0.80      4778
weighted avg       0.80      0.80      0.80      4778

param_grid = {'C':[0.1,1,10],'gamma':[1,0.1,0.01],'kernel':['rbf'],'class_weight': ['balanced']}
grid_ros = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 2)
grid_ros.fit(x_train_ros,y_train_ros)
Fitting 2 folds for each of 9 candidates, totalling 18 fits
[CV] END ..C=0.1, class_weight=balanced, gamma=1, kernel=rbf; total time=   4.4s
[CV] END ..C=0.1, class_weight=balanced, gamma=1, kernel=rbf; total time=   4.4s
[CV] END C=0.1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   3.2s
[CV] END C=0.1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   3.2s
[CV] END C=0.1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   3.6s
[CV] END C=0.1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   3.7s
[CV] END ....C=1, class_weight=balanced, gamma=1, kernel=rbf; total time=   3.6s
[CV] END ....C=1, class_weight=balanced, gamma=1, kernel=rbf; total time=   3.6s
[CV] END ..C=1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   2.8s
[CV] END ..C=1, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   2.8s
[CV] END .C=1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   3.2s
[CV] END .C=1, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   3.2s
[CV] END ...C=10, class_weight=balanced, gamma=1, kernel=rbf; total time=   3.4s
[CV] END ...C=10, class_weight=balanced, gamma=1, kernel=rbf; total time=   3.4s
[CV] END .C=10, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   2.9s
[CV] END .C=10, class_weight=balanced, gamma=0.1, kernel=rbf; total time=   2.9s
[CV] END C=10, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   3.0s
[CV] END C=10, class_weight=balanced, gamma=0.01, kernel=rbf; total time=   3.0s
GridSearchCV(cv=2, estimator=SVC(),
             param_grid={'C': [0.1, 1, 10], 'class_weight': ['balanced'],
                         'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']},
             verbose=2)
print(grid_ros.best_estimator_)
SVC(C=10, class_weight='balanced', gamma=1)
grid_predictions_ros = grid_ros.predict(x_test_ros)
confusion_matrix(y_test_ros,grid_predictions_ros)
array([[2070,  306],
       [  60, 2342]])
print(classification_report(y_test_ros,grid_predictions_ros))
              precision    recall  f1-score   support

           0       0.97      0.87      0.92      2376
           1       0.88      0.98      0.93      2402

    accuracy                           0.92      4778
   macro avg       0.93      0.92      0.92      4778
weighted avg       0.93      0.92      0.92      4778

print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.85      0.98      0.91      2377
           1       0.84      0.35      0.49       623

    accuracy                           0.85      3000
   macro avg       0.84      0.66      0.70      3000
weighted avg       0.85      0.85      0.82      3000

print(classification_report(y_test,grid_predictions))
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      2377
           1       0.52      0.37      0.43       623

    accuracy                           0.80      3000
   macro avg       0.68      0.64      0.65      3000
weighted avg       0.78      0.80      0.78      3000

print(classification_report(y_test_rus, y_pred_rus))
              precision    recall  f1-score   support

           0       0.74      0.78      0.76       616
           1       0.76      0.72      0.74       607

    accuracy                           0.75      1223
   macro avg       0.75      0.75      0.75      1223
weighted avg       0.75      0.75      0.75      1223

print(classification_report(y_test_rus,grid_predictions_rus))
              precision    recall  f1-score   support

           0       0.74      0.78      0.76       616
           1       0.76      0.72      0.74       607

    accuracy                           0.75      1223
   macro avg       0.75      0.75      0.75      1223
weighted avg       0.75      0.75      0.75      1223

print(classification_report(y_test_ros, y_pred_ros))
              precision    recall  f1-score   support

           0       0.79      0.82      0.80      2376
           1       0.81      0.78      0.80      2402

    accuracy                           0.80      4778
   macro avg       0.80      0.80      0.80      4778
weighted avg       0.80      0.80      0.80      4778

print(classification_report(y_test_ros,grid_predictions_ros))
              precision    recall  f1-score   support

           0       0.97      0.87      0.92      2376
           1       0.88      0.98      0.93      2402

    accuracy                           0.92      4778
   macro avg       0.93      0.92      0.92      4778
weighted avg       0.93      0.92      0.92      4778
