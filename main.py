# Thank you to StatQuest and his amazing video about XGBoost for helping me get started with XGBoost.
# https://www.youtube.com/watch?v=GrJP9FLV3FE&t=2635s

import pandas
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib
from sklearn import preprocessing


Admissions = pandas.read_excel('/Users/carstenjuliansavage/Desktop/IPEDS_data.xlsx')
pandas.set_option('display.max_columns', None)

# Filtering dataset for input and output variables only

AdmissionsSlim = (Admissions
    .filter(['Percent admitted - total',
             'ACT Composite 75th percentile score',
             'Historically Black College or University',
             'Total  enrollment',
             'Total price for out-of-state students living on campus 2013-14',
             'Percent of total enrollment that are White',
             'Percent of total enrollment that are women'])
    .dropna()
)

AdmissionsSlim.columns

AdmissionsSlim.columns = ['Per_Admit','ACT_75TH','Hist_Black','Total_ENROLL','Total_Price','Per_White','Per_Women']

# Defining 'Selective' as an Admittance Rate Under 30%
AdmissionsSlim['Per_Admit'] = np.where(AdmissionsSlim['Per_Admit'] < 50,1,0)
AdmissionsSlim['Hist_Black'] = np.where(AdmissionsSlim['Hist_Black'] == 'Yes',1,0)

# Create a new variable, which is the percentage of total enrollment that are non-white.
AdmissionsSlim = (AdmissionsSlim
    .assign(Per_Non_White=lambda a: 100-a.Per_White)
)

X = AdmissionsSlim[['ACT_75TH',
                    'Hist_Black',
                    'Total_ENROLL',
                    'Total_Price',
                    'Per_Non_White',
                    'Per_Women']]
y = AdmissionsSlim[['Per_Admit']]

# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X_train)
pandas.set_option('display.max_columns', None)
X_Stats.describe()

y_train_Stats = pandas.DataFrame(y_train)
y_test_Stats = pandas.DataFrame(y_test)
y_train_Stats.describe()
y_test_Stats.describe()

# We can see that the data has stratified as intended.

XGB = xgb.XGBClassifier(objective='binary:logistic',
                            missing=0,
                            seed=47)
XGB.fit(X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=15,
        eval_metric = 'aucpr',
        eval_set=[(X_test,y_test)])

plot_confusion_matrix(XGB,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Selective=N","Selective=Y"])

# Already, our XGBoost is doing a better job than the neural network at classifying Selective/Not Selective.

# Let's attempt to impose model error costs on the Bankrupt obs.

# Let's optimize the parameters - First Pass.
To_Optimize_Parameters = {
    'max_depth':[1,2,3,4,5],
    'learning_rate':[1.0,0.1,0.01,0.001],
    'gamma':[0,0.5,1.0],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,3,5]
}

# I am getting the following after the first pass:
# {'gamma': 0, 'learning_rate': 1.0, 'max_depth': 3, 'reg_lambda': 10.0, 'scale_pos_weight': 1}


# It appears that reg_lambda and scale_pos_weight are signaling that the optimal value is higher.
# Let's rerun the parameter optimization with a higher upper bound of possible values.
To_Optimize_Parameters = {
    'max_depth':[3],
    'learning_rate':[1.0],
    'gamma':[0],
    'reg_lambda':[10.0, 25.0, 50.0, 75.0, 100.0],
    'scale_pos_weight':[0.1, 0.5, 1.0]
}

# I am getting the following after the second pass:
# {'gamma': 0, 'learning_rate': 1.0, 'max_depth': 3, 'reg_lambda': 10.0, 'scale_pos_weight': 1.0}
# At this point, it looks like we are done optimizing the parameters.

# Run the following chunks for each pass (for each To_Optimize_Parameters)
optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=47,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid = To_Optimize_Parameters,
    scoring = 'roc_auc',
    verbose = 0,
    n_jobs = 10,
    cv = 3
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=15,
                   eval_metric='auc',
                   eval_set=[(X_test,y_test)],
                   verbose=False)

# This function will give us the optimal parameters for each pass.
print(optimal_params.best_params_)

# Now that we have the optimal parameters, let's try rerunning the XGBoost algorithm.
# {'gamma': 0, 'learning_rate': 1.0, 'max_depth': 3, 'reg_lambda': 10.0, 'scale_pos_weight': 1.0}


XGB_Refined = xgb.XGBClassifier(seed = 47,
                                objective='binary:logistic',
                                gamma=0,
                                learn_rate=1.0,
                                max_depth=3,
                                reg_lambda=10.0,
                                scale_pos_weights=1.0,
                                subsample=0.9,
                                colsample_bytree=0.5)

XGB_Refined.fit(X_train,
                y_train,
                verbose=True,
                early_stopping_rounds=15,
                eval_metric='aucpr',
                eval_set=[(X_test,y_test)])

plot_confusion_matrix(XGB_Refined,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Selective=N","Selective=Y"])

# It looks like we've reduced the number of erroneous classifications from 33 to 30.
# The parameter tuning did work. The model now predicts fewer Selective schools, however, than before.
# That's the tradeoff we made to predict more Non-Selective schools.