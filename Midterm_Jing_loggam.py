
from __future__ import division, print_function, unicode_literals

import pandas as pd
import numpy as np
import sklearn as sk
import random as rnd
np.random.seed(42)

# python run output no wrap
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objs as go

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Census"
def save_fig(fig_id, tight_layout=True):
    #path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_id + ".png",format='png', dpi=300)


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the data
train_df = pd.read_csv("census_train.csv")
test_df = pd.read_csv("census_test.csv")

print("----------------------------------------------------------- \n train_df")
print(train_df.head())
print(train_df.info())
print(train_df["income"].value_counts())
print(train_df.describe())
print(test_df.describe())
print(train_df.columns.values)
print(train_df.describe())
print('census_train["sex"].value_counts():', '\n', train_df["sex"].value_counts())


# find missing values
# Create a new function:
def num_missing(x):
    return sum(x.isnull())


# Applying per column:
print("Missing values per column:")
print(train_df.apply(num_missing, axis=0))  # axis=0 defines that function is to be applied on each column
# Applying per row:
print("Missing values per row:")
print(train_df.apply(num_missing, axis=1).head())  # axis=1 defines that function is to be applied on each row

# Table
my_tab1 = pd.crosstab(index=train_df["sex"],  # Make a crosstab
                      columns="count")  # Name the count column
print('sex table: \n', my_tab1)
my_tab2 = pd.crosstab(index=train_df["relationship"],  # Make a crosstab
                      columns="count")  # Name the count column
print('relationship table: \n', my_tab2)
my_tab3 = pd.crosstab(index=train_df["native-country"],  # Make a crosstab
                      columns="count")  # Name the count column
print('native-country table: \n', my_tab3)
my_tab4 = pd.crosstab(index=train_df["workclass"],  # Make a crosstab
                      columns="count")  # Name the count column
print('workclass table: \n', my_tab4)
my_tab5 = pd.crosstab(index=train_df["education"],  # Make a crosstab
                      columns="count")  # Name the count column
print('education table: \n', my_tab5)
my_tab6 = pd.crosstab(index=train_df["marital-status"],  # Make a crosstab
                      columns="count")  # Name the count column
print('marital-status table: \n', my_tab6)
my_tab7 = pd.crosstab(index=train_df["occupation"],  # Make a crosstab
                      columns="count")  # Name the count column
print('occupation table: \n', my_tab7)
my_tab8 = pd.crosstab(index=train_df["race"],  # Make a crosstab
                      columns="count")  # Name the count column
print('race table: \n', my_tab8)
my_tab9 = pd.crosstab(index=train_df["income"],  # Make a crosstab
                      columns="count")  # Name the count column
print('income table: \n', my_tab9)


# drop rows where SEX is missing
train_df = train_df[train_df.sex.notnull()]
# Create Gender column with female-1, male-0
train_df['Gender'] = train_df['sex'].map({' Male': 0, ' Female': 1}).astype(int)
test_df['Gender'] = test_df['sex'].map({' Male': 0, ' Female': 1}).astype(int)
# recode "income"
#train_df["income"] = train_df["income"].fillna(' <=50K').astype(int)
train_df['INCOME'] = train_df['income'].map({' <=50K': 0, ' >50K': 1}).astype(int)
test_df['INCOME'] = test_df['income'].map({' <=50K': 0, ' >50K': 1}).astype(int)
# recode "race"
train_df['RACE'] = train_df['race'].map({' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4 }).astype(int)
test_df['RACE'] = test_df['race'].map({' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4 }).astype(int)
'''
# recode "education"
train_df['EDU'] = train_df['education'].map({' 10th': 0,' 11th': 0,' 12th': 0,' 1st-4th': 0,' 5th-6th': 0, ' 7th-8th': 0, 
        ' 9th': 0,' Preschool': 0, ' HS-grad': 1, ' Some-college': 2, ' Assoc-acdm': 3, ' Assoc-voc': 3, ' Bachelors': 4, ' Masters': 5, ' Doctorate': 6,
        ' Prof-school': 7}).astype(int)
test_df['EDU'] = train_df['education'].map({' 10th': 0,' 11th': 0,' 12th': 0,' 1st-4th': 0,' 5th-6th': 0, ' 7th-8th': 0, 
        ' 9th': 0,' Preschool': 0, ' HS-grad': 1, ' Some-college': 2, ' Assoc-acdm': 3, ' Assoc-voc': 3, ' Bachelors': 4, ' Masters': 5, ' Doctorate': 6,
        ' Prof-school': 7}).astype(int)
'''
# recode "marital-status"
train_df['MARRI'] = train_df['marital-status'].map({' Married-AF-spouse': 0, ' Married-civ-spouse': 0, ' Divorced': 1,
        ' Married-spouse-absent': 1, ' Separated': 1, ' Widowed' :1, ' Never-married': 2 }).astype(int)
test_df['MARRI'] = train_df['marital-status'].map({' Married-AF-spouse': 0, ' Married-civ-spouse': 0, ' Divorced': 1,
        ' Married-spouse-absent': 1, ' Separated': 1, ' Widowed' :1, ' Never-married': 2 }).astype(int)
 # recode "workclass"
train_df['WORKC'] = train_df['workclass'].map({' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc' :2, ' Federal-gov': 3,
        ' State-gov': 3, ' Local-gov': 3,  ' ?': 4, ' Without-pay': 4, ' Never-worked': 4 }).astype(int)
test_df['WORKC'] = train_df['workclass'].map({' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc' :2, ' Federal-gov': 3,
        ' State-gov': 3, ' Local-gov': 3,  ' ?': 4, ' Without-pay': 4, ' Never-worked': 4 }).astype(int)
#
def COUNTRY(series):
    if series == ' United-States':
        return 0
    else:
        return 1
train_df['COUNTRY'] = train_df['native-country'].apply(COUNTRY)
train_df['COUNTRY'].value_counts(sort=False)
train_df[["native-country", "COUNTRY"]].head(11)

test_df['COUNTRY'] = test_df['native-country'].apply(COUNTRY)
test_df['COUNTRY'].value_counts(sort=False)
test_df[["native-country", "COUNTRY"]].head(11)
# recode country
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
#train_df["Country"] = lb_make.fit_transform(train_df["native-country"])
#train_df[["native-country", "Country"]].head(11)
# recode occupation
train_df["OCCUP"] = lb_make.fit_transform(train_df["occupation"])
train_df[["occupation", "OCCUP"]].head(11)
# recode relationship
train_df["RELATION"] = lb_make.fit_transform(train_df["relationship"])
train_df[["relationship", "RELATION"]].head(11)

lb_make = LabelEncoder()
#test_df["Country"] = lb_make.fit_transform(test_df["native-country"])
#test_df[["native-country", "Country"]].head(11)
# recode occupation
test_df["OCCUP"] = lb_make.fit_transform(test_df["occupation"])
test_df[["occupation", "OCCUP"]].head(11)
# recode relationship
test_df["RELATION"] = lb_make.fit_transform(test_df["relationship"])
test_df[["relationship", "RELATION"]].head(11)


# #Determine pivot table
impute_grps = train_df.pivot_table(values=["INCOME"], index=["sex","race"], aggfunc=np.mean)
print (impute_grps)
## crosstab
pd.crosstab(train_df["INCOME"],train_df["sex"],margins=True)
#crosstab, percent
def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(train_df["INCOME"],train_df["sex"],margins=True).apply(percConvert, axis=1)

my_tabc3 = train_df[['race', 'INCOME']].groupby(['race'], as_index=False).mean().sort_values(by='INCOME', ascending=False)
print('race table: \n', my_tabc3)
my_tabc4 = train_df[["Gender", "INCOME"]].groupby(['Gender'], as_index=False).mean().sort_values(by='INCOME', ascending=False)
print('Sex table: \n', my_tabc4)
print("----------------------------------------------------------- \n end1 \n")



#### Histgram
pclass_xt = pd.crosstab(train_df['sex'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by sex')
plt.ylabel('INCOME')
plt.xlabel('sex')
save_fig("Mid.1.INCOME_sex_plot")
train_df.boxplot(column="INCOME",by="sex")

pclass_xt = pd.crosstab(train_df['occupation'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by occupation')
plt.ylabel('INCOME')
plt.xlabel('occupation')
save_fig("Mid.1.INCOME_occupation_plot")

pclass_xt = pd.crosstab(train_df['native-country'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by native-country')
plt.ylabel('INCOME')
plt.xlabel('native-country')
save_fig("Mid.1.INCOME_native-country_plot")

pclass_xt = pd.crosstab(train_df['race'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by race')
plt.ylabel('INCOME')
plt.xlabel('race')
save_fig("Mid.1.INCOME_race_plot")

pclass_xt = pd.crosstab(train_df['education'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by education')
plt.ylabel('INCOME')
plt.xlabel('education')
save_fig("Mid.1.INCOME_education_plot")

pclass_xt = pd.crosstab(train_df['education-num'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by education-num')
plt.ylabel('INCOME')
plt.xlabel('education-num')
save_fig("Mid.1.INCOME_education-num_plot")


pclass_xt = pd.crosstab(train_df['workclass'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by workclass')
plt.ylabel('INCOME')
plt.xlabel('workclass')
save_fig("Mid.1.INCOME_workclass_plot")

pclass_xt = pd.crosstab(train_df['WORKC'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by WORKC')
plt.ylabel('INCOME')
plt.xlabel('WORKC')
save_fig("Mid.1.INCOME_WORKC_plot")


pclass_xt = pd.crosstab(train_df['relationship'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by relationship')
plt.ylabel('INCOME')
plt.xlabel('relationship')
save_fig("Mid.1.INCOME_relationship_plot")

pclass_xt = pd.crosstab(train_df['marital-status'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by marital-status')
plt.ylabel('INCOME')
plt.xlabel('marital-status')
save_fig("Mid.1.INCOME_marital-status_plot")

pclass_xt = pd.crosstab(train_df['MARRI'], train_df['INCOME'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(int), axis=0)
pclass_xt_pct.plot(kind='bar', stacked=True, title='INCOME by MARRI')
plt.ylabel('INCOME')
plt.xlabel('MARRI')
save_fig("Mid.1.INCOME_MARRI_plot")


'''
# create a DataFrame of dummy variables for EDU
EDU_dummies = pd.get_dummies(train_df['EDU'], prefix='EDU')
EDU_dummies.drop(EDU_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
train_df = pd.concat([train_df, EDU_dummies], axis=1)
print(train_df.info())
'''
# create a DataFrame of dummy variables for WORKC (train_df)
WORKC_dummies = pd.get_dummies(train_df['WORKC'], prefix='WORKC')
WORKC_dummies.drop(WORKC_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
train_df = pd.concat([train_df, WORKC_dummies], axis=1)
print(train_df.info())
# create a DataFrame of dummy variables for MARRI
MARRI_dummies = pd.get_dummies(train_df['MARRI'], prefix='MARRI')
MARRI_dummies.drop(MARRI_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
train_df = pd.concat([train_df, MARRI_dummies], axis=1)
print(train_df.info())
# create a DataFrame of dummy variables for RACE
RACE_dummies = pd.get_dummies(train_df['RACE'], prefix='RACE')
RACE_dummies.drop(RACE_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
train_df = pd.concat([train_df, RACE_dummies], axis=1)
print(train_df.info())
# create a DataFrame of dummy variables for RALATION
RELATION_dummies = pd.get_dummies(train_df['RELATION'], prefix='RELATION')
RELATION_dummies.drop(RELATION_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
train_df = pd.concat([train_df, RELATION_dummies], axis=1)
print(train_df.info())
# create a DataFrame of dummy variables for OCCUP
OCCUP_dummies = pd.get_dummies(train_df['OCCUP'], prefix='OCCUP')
OCCUP_dummies.drop(OCCUP_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
train_df = pd.concat([train_df, OCCUP_dummies], axis=1)


# create a DataFrame of dummy variables for WORKC (test_df)
WORKC_dummies = pd.get_dummies(test_df['WORKC'], prefix='WORKC')
WORKC_dummies.drop(WORKC_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
test_df = pd.concat([test_df, WORKC_dummies], axis=1)
print(test_df.info())
# create a DataFrame of dummy variables for MARRI
MARRI_dummies = pd.get_dummies(test_df['MARRI'], prefix='MARRI')
MARRI_dummies.drop(MARRI_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
test_df = pd.concat([test_df, MARRI_dummies], axis=1)
print(test_df.info())
# create a DataFrame of dummy variables for RACE
RACE_dummies = pd.get_dummies(test_df['RACE'], prefix='RACE')
RACE_dummies.drop(RACE_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
test_df = pd.concat([test_df, RACE_dummies], axis=1)
print(test_df.info())
# create a DataFrame of dummy variables for RALATION
RELATION_dummies = pd.get_dummies(test_df['RELATION'], prefix='RELATION')
RELATION_dummies.drop(RELATION_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
test_df = pd.concat([test_df, RELATION_dummies], axis=1)
print(test_df.info())
# create a DataFrame of dummy variables for OCCUP
OCCUP_dummies = pd.get_dummies(test_df['OCCUP'], prefix='OCCUP')
OCCUP_dummies.drop(OCCUP_dummies.columns[0], axis=1, inplace=True)
# concatenate the original DataFrame and the dummy DataFrame
test_df = pd.concat([test_df, OCCUP_dummies], axis=1)
print(test_df.info())


#train_df.drop('MARRI_2', axis=1)


# define X and y
feature_int = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week','COUNTRY','Gender']
feature_cols0 = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                'hours-per-week', 'WORKC_1', 'WORKC_2', 'WORKC_3', 'WORKC_4', 'MARRI_1',
                'MARRI_2']
feature_cols1 = ['age',  'fnlwgt',  'education-num', 'capital-gain', 'capital-loss','RACE_1','RACE_2','RACE_3','RACE_4',
                 'hours-per-week',   'WORKC_1', 'WORKC_2' ,'WORKC_3', 'WORKC_4' ,'MARRI_1', 'MARRI_2',
                  'COUNTRY','RELATION_1', 'OCCUP_1',
                 'OCCUP_2', 'OCCUP_3' ,'OCCUP_4', 'OCCUP_5' ,'OCCUP_6', 'OCCUP_7', 'OCCUP_8',
                 'OCCUP_9', 'OCCUP_10', 'OCCUP_11', 'OCCUP_12', 'OCCUP_13', 'OCCUP_14']
feature_cols2 = ['age',  'fnlwgt',  'education-num', 'capital-gain', 'capital-loss','RACE_1','RACE_2','RACE_3','RACE_4',
                 'hours-per-week',   'WORKC_1', 'WORKC_2' ,'WORKC_3', 'WORKC_4' ,'MARRI_1', 'MARRI_2',
                  'COUNTRY','RELATION_1', 'RELATION_2', 'RELATION_3', 'RELATION_4', 'RELATION_5', 'OCCUP_1',
                 'OCCUP_2', 'OCCUP_3' ,'OCCUP_4', 'OCCUP_5' ,'OCCUP_6', 'OCCUP_7', 'OCCUP_8',
                 'OCCUP_9', 'OCCUP_10', 'OCCUP_11', 'OCCUP_12', 'OCCUP_13', 'OCCUP_14']
feature_ds = ['Gender','RACE_1','RACE_2','RACE_3','RACE_4','WORKC_1','WORKC_2','WORKC_3','WORKC_4','MARRI_1','MARRI_2']
XI = train_df[feature_int]
X = train_df[feature_cols1]
y = train_df.INCOME

# CORR
from pandas.plotting import scatter_matrix
attributes = ['INCOME','age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scatter_matrix(train_df[attributes], figsize=(12, 8))
save_fig("Mid.2.1.scatter_matrix_plot")
plt.show()
# METHOD 2.
print(train_df.corr())#https://matplotlib.org/examples/color/colormaps_reference.html
plt.matshow(train_df[attributes].corr(),cmap=plt.cm.OrRd)#cmap=plt.cm.viridis/gray/bone/pink/hot
plt.xticks(range(len(train_df[attributes].columns)), train_df[attributes].columns)
plt.yticks(range(len(train_df[attributes].columns)), train_df[attributes].columns)
plt.colorbar()
save_fig("Mid.2.2.scatter_matshow_plot")
plt.show()






################################################################################################################################
###############################################################################################################################
# train/test split
# X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=1) #test_size=0.2
# random_state : int or RandomState: Pseudo-random number generator state used for random sampling.
# train_test_split splits arrays or matrices into random train and test subsets. That means that everytime you run it without specifying random_state, you will get a different result.
# if you use random_state=some_number, then you can guarantee that the output of Run 1 will be equal to the output of Run 2, i.e. your split will be always the same. It doesn't matter what the actual random_state number is 42, 0, 21, ... The important thing is that everytime you use 42, you will always get the same output the first time you make the split.
X_train = train_df[feature_cols1]
X_train.info()
Y_train = train_df["INCOME"]
X_test = test_df[feature_cols1]
X_test.info()
Y_test = test_df["INCOME"]
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

# train a logistic regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e9)  # There is no way to switch off regularization in scikit-learn, but you can make it ineffective by setting the tuning parameter C to a large number.
logreg.fit(X_train, Y_train)
print(logreg.coef_)
logreg.predict_log_proba(X_train)

# make predictions for testing set
Y_pred_class = logreg.predict(X_test)
# calculate testing accuracy
from sklearn import metrics

print('\n ------------------------------------------------------------------')
print("\n calculate testing accuracy (M1. X_train):", metrics.accuracy_score(Y_test, Y_pred_class))
print('\n ------------------------------------------------------------------')
#  ROC curves and AUC
# https://www.medcalc.org/manual/roc-curves.php
# predict probability of survival
Y_pred_prob = logreg.predict_proba(X_test)[:, 1]

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# plot ROC curve. Receiver Operating Characteristic (ROC) curve the true positive rate (Sensitivity)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_prob)
fig = plt.figure()
fig.subplots_adjust(bottom=0)
fig.subplots_adjust(top=1)
fig.subplots_adjust(right=1)
fig.subplots_adjust(left=0)
plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# plt.fig.subplots_adjust()
save_fig("Mid.2.ROC curve (M1. X_train)")
plt.show()
# calculate AUC. i.e. AUROC = Area Under the Receiver Operating Characteristic curve.
from sklearn import metrics

print("\n Area under the ROC curve.:", metrics.roc_auc_score(Y_test, Y_pred_prob))
print('\n ------------------------------------------------------------------')

#####----------------------------------------------------------------------------------------------------------########
# try to improve the accurancy: simply scaling the inputs (as discussed in Chapter 2) increases accuracy above 90%:
# ：StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # standardized the variable
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))

logreg = LogisticRegression(C=1e9)  # There is no way to switch off regularization in scikit-learn, but you can make it ineffective by setting the tuning parameter C to a large number.
logreg.fit(X_train_scaled, Y_train)
print(logreg.coef_)
'''
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
'''
# make predictions for testing set
Y_scaler_pred_class = logreg.predict(X_test_scaled)
# calculate testing accuracy
from sklearn import metrics

print('\n ------------------------------------------------------------------')
print("\n calculate testing accuracy (M1. X_train_scaled):", metrics.accuracy_score(Y_test, Y_scaler_pred_class))
print('\n ------------------------------------------------------------------')
#  ROC curves and AUC
# https://www.medcalc.org/manual/roc-curves.php
# predict probability of survival
Y_scaler_pred_prob = logreg.predict_proba(X_test_scaled)[:, 1]

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
# plot ROC curve. Receiver Operating Characteristic (ROC) curve the true positive rate (Sensitivity)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_scaler_pred_prob)
fig = plt.figure()
fig.subplots_adjust(bottom=0)
fig.subplots_adjust(top=1)
fig.subplots_adjust(right=1)
fig.subplots_adjust(left=0)
plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# plt.fig.subplots_adjust()
save_fig("Mid.2.ROC curve (M1. X_train_scaled)")
plt.show()
# calculate AUC. i.e. AUROC = Area Under the Receiver Operating Characteristic curve.
from sklearn import metrics

print("\n Area under the ROC curve.:", metrics.roc_auc_score(Y_test, Y_scaler_pred_prob))
print('\n ------------------------------------------------------------------')

#########-----------------------------------------------------------------------------------------------------------##########
##########Poly
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_train_scaled_poly = poly.fit_transform(X_train_scaled.astype(np.float64))
X_test_scaled_poly = poly.fit_transform(X_test_scaled.astype(np.float64))

logreg = LogisticRegression(C=1e9)  # There is no way to switch off regularization in scikit-learn, but you can make it ineffective by setting the tuning parameter C to a large number.
logreg.fit(X_train_scaled_poly, Y_train)
print(logreg.coef_)

# make predictions for testing set
Y_scaler_poly_pred_class = logreg.predict(X_test_scaled_poly)
# calculate testing accuracy
from sklearn import metrics

print('\n ------------------------------------------------------------------')
print("\n calculate testing accuracy (M1. X_train_scaled_poly):", metrics.accuracy_score(Y_test, Y_scaler_poly_pred_class))
print('\n ------------------------------------------------------------------')
#  ROC curves and AUC
# https://www.medcalc.org/manual/roc-curves.php
# predict probability of survival
Y_scaler_poly_pred_prob = logreg.predict_proba(X_test_scaled_poly)[:, 1]

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
# plot ROC curve. Receiver Operating Characteristic (ROC) curve the true positive rate (Sensitivity)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_scaler_poly_pred_prob)
fig = plt.figure()
fig.subplots_adjust(bottom=0)
fig.subplots_adjust(top=1)
fig.subplots_adjust(right=1)
fig.subplots_adjust(left=0)
plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# plt.fig.subplots_adjust()
save_fig("Mid.2.ROC curve (M1. X_train_scaled_poly)")
plt.show()
# calculate AUC. i.e. AUROC = Area Under the Receiver Operating Characteristic curve.
from sklearn import metrics

print("\n Area under the ROC curve.:", metrics.roc_auc_score(Y_test, Y_scaler_poly_pred_prob))
print('\n ------------------------------------------------------------------')


########--------------------------------------------------------------------------------------------------------------#########
# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
from sklearn import metrics, cross_validation

np.random.seed(42)
# Use cross_val_score to automatically split, fit, and score.
'''
from sklearn import model_selection
scores = model_selection.cross_val_score(logreg,X_train_scaled, Y_train, cv=10)
print(scores)
print('average score: {}'.format(scores.mean()))
'''
logreg.fit(X_train, Y_train)
auc_log1 = cross_val_score(logreg, X_train, Y_train, cv=10, scoring='roc_auc').mean()
print("\n calculate cross-validated AUC  (M1. X_train):", auc_log1)
acc_log1 = cross_val_score(logreg, X_train, Y_train, cv=10, scoring='accuracy').mean()
print("\n calculate cross-validated accurancy  (M1. X_train):", acc_log1)
acc_logs1 = cross_validation.cross_val_predict(logreg, X_train, Y_train, cv=10)
print(metrics.accuracy_score(Y_train, acc_logs1))
print(metrics.classification_report(Y_train, acc_logs1))
print(logreg.coef_)
print('\n ------------------------------------------------------------------')
logreg.fit(X_train_scaled, Y_train)
auc_log2 = cross_val_score(logreg, X_train_scaled, Y_train, cv=10, scoring='roc_auc').mean()
print("\n calculate cross-validated AUC  (M2. X_train_scaled):", auc_log2)
acc_log2 = cross_val_score(logreg, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
print("\n calculate cross-validated accurancy  (M2. X_train_scaled):", acc_log2)
acc_logs2 = cross_validation.cross_val_predict(logreg, X_train_scaled, Y_train, cv=10)
print(metrics.accuracy_score(Y_train, acc_logs2))
print(metrics.classification_report(Y_train, acc_logs2))
print(logreg.coef_)
print('\n ------------------------------------------------------------------')
# call predict_proba() to get the list of probabilities that the classifier assigned to each instance for each class:
logreg.fit(X_train_scaled_poly, Y_train)
auc_log3 = cross_val_score(logreg, X_train_scaled_poly, Y_train, cv=10, scoring='roc_auc').mean()
print("\n calculate cross-validated AUC  (M2. X_train_scaled_poly):", auc_log2)
acc_log3 = cross_val_score(logreg, X_train_scaled_poly, Y_train, cv=10, scoring='accuracy').mean()
print("\n calculate cross-validated accurancy  (M2. X_train_scaled_poly):", acc_log2)
acc_logs3 = cross_validation.cross_val_predict(logreg, X_train_scaled_poly, Y_train, cv=10)
print(metrics.accuracy_score(Y_train, acc_logs2))
print(metrics.classification_report(Y_train, acc_logs3))
print(logreg.coef_)
print('\n ------------------------------------------------------------------')
# call predict_proba() to get the list of probabilities that the classifier assigned to each instance for each class:
###############################################################################################################################
# GAM
import pandas as pd
from pygam import LogisticGAM
# Fit a model with the default parameters
gam = LogisticGAM().fit(X_train_scaled, Y_train)
gam.summary()
print('gam.accuracy(X_train_scaled, Y_train):',gam.accuracy(X_train_scaled, Y_train))
print('gam.accuracy(X_test_scaled, Y_test):',gam.accuracy(X_test_scaled, Y_test))
acc_loggamc = cross_val_score(gam, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
print('acc_loggam_cross-validation, train_scaled',acc_loggamc)


# make predictions for testing set
Y_scaler_pred_class = logreg.predict(X_test_scaled)
# calculate testing accuracy
from sklearn import metrics

print('\n ------------------------------------------------------------------')
print("\n calculate testing accuracy (M1. X_train_scaled):", metrics.accuracy_score(Y_test, Y_scaler_pred_class))
print('\n ------------------------------------------------------------------')
#  ROC curves and AUC
# https://www.medcalc.org/manual/roc-curves.php
# predict probability of survival
Y_scaler_pred_prob_GAM = gam.predict_proba(X_test_scaled)[:, 1]

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
# plot ROC curve. Receiver Operating Characteristic (ROC) curve the true positive rate (Sensitivity)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_scaler_pred_prob_GAM)

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
# plot ROC curve. Receiver Operating Characteristic (ROC) curve the true positive rate (Sensitivity)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_scaler_pred_prob)
fig = plt.figure()
fig.subplots_adjust(bottom=0)
fig.subplots_adjust(top=1)
fig.subplots_adjust(right=1)
fig.subplots_adjust(left=0)
plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# plt.fig.subplots_adjust()
save_fig("Mid.2.ROC curve (M1. X_train_scaled_GAMLOG)")
plt.show()
# calculate AUC. i.e. AUROC = Area Under the Receiver Operating Characteristic curve.
from sklearn import metrics

print("\n Area under the ROC curve.:", metrics.roc_auc_score(Y_test, Y_scaler_pred_prob))
print('\n ------------------------------------------------------------------')
'''

'''''''''
##############################################################################################################################
# Support Vector Machines
svc = SVC(C=1, kernel='linear')
svc.fit(X_train_scaled, Y_train)
Y_pred_svc = svc.predict(X_test_scaled)
acc_svc = round(svc.score(X_test_scaled, Y_test) * 100, 2)
acc_svc
clf = SVC(C=1, kernel='poly', degree=3)
acc_svcc = cross_val_score(clf, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
acc_svcc

######################################################################################################
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, Y_train)
Y_pred_knn = knn.predict(X_test_scaled)
acc_knn = round(knn.score(X_test_scaled, Y_test) * 100, 2)
acc_knn
acc_knnc = cross_val_score(knn, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
acc_knnc

######################################################################################################
## Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train_scaled, Y_train)
Y_pred_lsvc = linear_svc.predict(X_test_scaled)
acc_linear_svc = round(linear_svc.score(X_test_scaled, Y_test) * 100, 2)
acc_linear_svc

######################################################################################################
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train_scaled, Y_train)
Y_pred_sgd = sgd.predict(X_test_scaled)
acc_sgd = round(sgd.score(X_test_scaled, Y_test) * 100, 2)
acc_sgd
acc_sgdc = cross_val_score(sgd, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
acc_sgdc

######################################################################################################
## Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train_scaled, Y_train)
Y_pred = gaussian.predict(X_test_scaled)
acc_gaussian = round(gaussian.score(X_test_scaled, Y_test) * 100, 2)
acc_gaussian
acc_gauc = cross_val_score(gaussian, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
acc_gauc

######################################################################################################
## Perceptron
perceptron = Perceptron()
perceptron.fit(X_train_scaled, Y_train)
Y_pred = perceptron.predict(X_test_scaled)
acc_perceptron = round(perceptron.score(X_test_scaled, Y_test) * 100, 2)
acc_perceptron
acc_perc = cross_val_score(perceptron, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
acc_perc

######################################################################################################
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_scaled, Y_train)
Y_pred = decision_tree.predict(X_test_scaled)
acc_decision_tree = round(decision_tree.score(X_test_scaled, Y_test) * 100, 2)
acc_decision_tree
acc_detreec = cross_val_score(decision_tree, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
acc_detreec

######################################################################################################
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_scaled, Y_train)
Y_pred = random_forest.predict(X_test_scaled)
random_forest.score(X_test_scaled, Y_test)
acc_random_forest = round(random_forest.score(X_test_scaled, Y_test) * 100, 2)
acc_random_forest
acc_random_forestc = cross_val_score(random_forest, X_train_scaled, Y_train, cv=10, scoring='accuracy').mean()
acc_random_forestc

######################################################################################################

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

models_cross_validation = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Log regression', 'Log regression (Scaled)',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent',
              'Decision Tree'],
    'Score': [acc_svcc, acc_knnc, acc_log1, acc_log2,
              acc_random_forestc, acc_gauc, acc_perc,
              acc_sgdc, acc_detreec]})
print(models_cross_validation.sort_values(by='Score', ascending=False))

############################################################################################################

from sklearn import linear_model
#ridge reg
X_train_scaled.info()
Ridge_reg = linear_model.Ridge (alpha = .5)
Ridge_reg.fit (X_train_scaled, Y_train) 
Ridge_reg.coef_
Ridge_cross = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
Ridge_cross.fit (X_train_scaled, Y_train) 


##############################################################################################################
#
