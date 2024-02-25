# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 05:05:20 2023

@author: User
"""
import numpy as np
import pandas as pd


import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance1 = np.array(importance)
    feature_importance = feature_importance1[feature_importance1 > 1/2*max(feature_importance1)]
    feature_names = np.array(names)
    feature_names = feature_names[feature_importance1 > 1/2*max(feature_importance1)]
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Data_cols_df = pd.read_csv("Data_cols.csv", sep=',', names =["Columns"]) 
Data_cols =Data_cols_df["Columns"].tolist()
Data_Clnd = pd.read_csv('Data_Addtnal_info_UnNorm_2_add_Clnd.csv', sep=',', names = Data_cols) # header=None)

###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()
Targets = Targets_all
target = 'Birads'
Targets.remove(target)

Counter(Data_Clnd[target])

train = Data_Clnd.drop(Targets, axis = 1)

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    Vals = dtrain[target].values
    Actuals = np.zeros((len(dtrain[target].values) , 1))
    Preds = np.zeros((len(dtrain_predictions) , 1))
    
    for i in range(len(Actuals)):
        if Vals[i] == 1:
            Actuals[i] = 2 
        elif Vals[i] == 2:
            Actuals[i] = 3
        elif Vals[i] == 3:
            Actuals[i] = 4
        elif Vals[i] == 4:
            Actuals[i] = 1
        
        if dtrain_predictions[i] == 1:
            Preds[i] = 2 
        elif dtrain_predictions[i] == 2:
            Preds[i] = 3
        elif dtrain_predictions[i] == 3:
            Preds[i] = 4
        elif dtrain_predictions[i] == 4:
            Preds[i] = 1
            
    for i in range(len(Actuals)):
        if dtrain_predictions[i] == 1:
            Preds[i] = 4
            break
    for i in range(len(Actuals)):
        if dtrain_predictions[i] == 3:
            Preds[i] = 2
            break
        
            
    print(Counter(Vals))
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(Actuals, Preds))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(Actuals, Preds, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(Actuals, Preds, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(Actuals, Preds, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['0', '2', '3', '4', '5'], 
                     columns = ['0', '2', '3', '4', '5'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    
    plot_feature_importance(alg.feature_importances_, predictors,target + ' ')
    
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()
Targets = Targets_all
target = 'ACR'
Targets.remove(target)

Counter(Data_Clnd[target])

train = Data_Clnd.drop(Targets, axis = 1)

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    Vals = dtrain[target].values
    Actuals = dtrain[target].values
    Preds = dtrain_predictions
    
            
    print(Counter(Vals))
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(Actuals, Preds))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(Actuals, Preds, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(Actuals, Preds, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(Actuals, Preds, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['a', 'b', 'c', 'd'], 
                     columns = ['a', 'b', 'c', 'd'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    
    plot_feature_importance(alg.feature_importances_, predictors,target + ' ')
    
    
predictors = [x for x in train.columns if (x not in [target] and 'Breast' in x)]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

target = 'Deviation benign or malignant'
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

Counter(Data_Clnd[target])


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 3
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['Probably benign', 'Probably malignant', 'Malignant'], 
                     columns = ['Probably benign', 'Probably malignant', 'Malignant'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors,target + ' ')
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 3,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)

###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i=3
target = Targets_all[i]
print(Targets_all[i])
grouped = Data_Clnd[target]

i=4
target = Targets_all[i]
print(Targets_all[i])
segmental = Data_Clnd[target]

i=5
target = Targets_all[i]
print(Targets_all[i])
diffusely_distributed = Data_Clnd[target]

i=6
target = Targets_all[i]
print(Targets_all[i])
regional = Data_Clnd[target]

Cal_dist = np.zeros((len(grouped), 1))
for i in range(len(grouped)):
    if grouped[i] == 1:
        Cal_dist[i] = 1 
    if segmental[i] == 1:
        Cal_dist[i] = 2 
    if diffusely_distributed[i] == 1:
        Cal_dist[i] = 3 
    if regional[i] == 1:
        Cal_dist[i] = 4 

# Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)
train['Cal_dist'] = Cal_dist
target = 'Cal_dist'

Counter(train[target])


import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    Vals = dtrain[target].values
    Actuals = dtrain[target].values
    Preds = dtrain_predictions
        
            
    print(Counter(Vals))
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(Actuals, Preds))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(Actuals, Preds, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(Actuals, Preds, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(Actuals, Preds, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['No Calcification', 'grouped', 'segmental', 'diffusely distributed', 'regional'], 
                     columns = ['No Calcification', 'grouped', 'segmental', 'diffusely distributed', 'regional'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    
    plot_feature_importance(alg.feature_importances_, predictors, 'calcification Dist ')
    
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i=3
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

Counter(Data_Clnd[target])



def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
    
    y_unique = np.unique(dtrain[target].values)
    mcm = multilabel_confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    print(mcm)
                    
    # print(alg.get_booster().get_fscore())
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    return alg.get_booster().get_fscore(), mcm
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
feat_imp_grouped, mcm_grouped = modelfit(xgb1, train, predictors)



###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i=4
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

Counter(Data_Clnd[target])

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
    
    y_unique = np.unique(dtrain[target].values)
    mcm = multilabel_confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    print(mcm)
                    
    # print(alg.get_booster().get_fscore())
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    return alg.get_booster().get_fscore(), mcm
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
feat_imp_segmental, mcm_segmental = modelfit(xgb1, train, predictors)



###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 5 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

Counter(Data_Clnd[target])


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
    
    y_unique = np.unique(dtrain[target].values)
    mcm = multilabel_confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    print(mcm)
                    
    # print(alg.get_booster().get_fscore())
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    return alg.get_booster().get_fscore(), mcm
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

feat_imp_diffusely_distributed, mcm_diffusely_distributed = modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 6 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

Counter(Data_Clnd[target])


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    # print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
    
    y_unique = np.unique(dtrain[target].values)
    mcm = multilabel_confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    print(mcm)
                    
    # print(alg.get_booster().get_fscore())
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    return alg.get_booster().get_fscore(), mcm
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
feat_imp_regional, mcm_regional = modelfit(xgb1, train, predictors)

###############################************************************###########################

feat_imp_grouped_n = list(feat_imp_grouped.keys())

feat_imp_regional_n = list(feat_imp_regional.keys())

feat_imp_diffusely_distributed_n = list(feat_imp_diffusely_distributed.keys())

feat_imp_segmental_n = list(feat_imp_segmental.keys())

def intersection(lst1, lst2):
     
    return [item for item in lst1 if item in lst2]

Feat_in_All_calc = intersection(feat_imp_grouped_n, intersection(feat_imp_regional_n, intersection(feat_imp_diffusely_distributed_n, feat_imp_segmental_n)))


feat_imp_grouped_v = list(feat_imp_grouped.values())

feat_imp_regional_v = list(feat_imp_regional.values())

feat_imp_diffusely_distributed_v = list(feat_imp_diffusely_distributed.values())

feat_imp_segmental_v = list(feat_imp_segmental.values())


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 7 
target = Targets_all[i] # 'Architecture Distortion' # 
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

Counter(Data_Clnd[target])

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['No', 'Yes'], 
                     columns = ['No', 'Yes'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors,'Architectural Distortion ')
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 8 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)
Counter(Data_Clnd[target])


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['No', 'Yes'], 
                     columns = ['No', 'Yes'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors,'Additional calcification ')
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)



###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 9 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

Counter(Data_Clnd[target])


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['No', 'Yes'], 
                     columns = ['No', 'Yes'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors,target + ' ')
    
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 10 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)



def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['No', 'Yes'], 
                     columns = ['No', 'Yes'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors,target + ' ')
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 11 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)



def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['No', 'Yes'], 
                     columns = ['No', 'Yes'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors,target + ' ')
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 12 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['No', 'Yes'], 
                     columns = ['No', 'Yes'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors, 'Additional mass ')
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)



###############################***********************************##############################

Targets_all = pd.read_csv("Targets_all.csv", names =["Targets"]) 
Targets_all =Targets_all["Targets"].tolist()

Targets = Targets_all

i = 13 
target = Targets_all[i]
print(Targets_all[i])
Targets.remove(target)

train = Data_Clnd.drop(Targets, axis = 1)



def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 5
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          )

        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print(dtrain_predictions)
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    print("Precision : %.4g" % metrics.precision_score(dtrain[target].values, dtrain_predictions, average='macro'))
    print("f1_score (Train): %f" % metrics.f1_score(dtrain[target].values, dtrain_predictions, average='macro'))
    
                    
    # print(alg.get_booster().get_fscore())
    
    from sklearn.metrics import confusion_matrix
    y_unique = np.unique(dtrain[target].values)
    mcm = confusion_matrix(dtrain[target].values, dtrain_predictions, labels = y_unique)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(mcm,
                     index = ['N/A', 'suspicious', 'very suspicious'], 
                     columns = ['N/A', 'suspicious', 'very suspicious'], 
                     dtype = int)
    print(cm_df)
    #Plotting the confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    sns.color_palette("hls", 8)
    sns.heatmap(cm_df, annot=True, linewidth=.5, center=30, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    plot_feature_importance(alg.feature_importances_, predictors, target + ' ')
    
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 num_class = 5,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)



