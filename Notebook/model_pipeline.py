# ================================
# Model Pipeline Configuaration
# ================================

#===============================
# IMPORTS
#===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
)

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

#===============================
# DATA BRINGING
#===============================
data = pd.read_csv("cleaned_loan_data.csv")

# Train Test Split
X = data.drop(columns=['loan_status_binary'])
y = data['loan_status_binary']

# First split into train and test
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
                                                    test_size=0.4,  # 40% becomes temp(val+test)
                                                    random_state=42, # class balance kept
                                                    stratify=y)
# Then split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                y_temp,
                                                test_size = 0.5, # half of temp becomes test that is half of the 20%
                                                random_state=42,
                                                stratify=y_temp)
# Feature Scaling
# Scale only numerical features from the training set
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#===============================
# CROSS VALIDATION SETUP
#===============================
""" 

Stratified K-Fold Cross Validation

We split the data  into 3 parts, 
but we keep the same percentage of Good/Bad in each split.
This is fair for training

"""
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Our main scoring metrics because we are about catching bad loans
scoring = "recall"

#===============================
# MODLE LIST + SMALL GRIDS 
#===============================
"""

we test 5 models:

1. LightGBM
2. XGBoost
3. CatBoost
4. Random Forest
5. Logistic Regression

"""

estimators_and_grids = {
    "lightgbm":{
        "estimator": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "clf__n_estimators":[100, 300],
            "clf__num_leaves":[31, 50],
            "clf__learning_rate":[0.1, 0.01]
        }
    },
    "xgboost":{
        "estimator": XGBClassifier(use_label_encoder=False,
                                   eval_metrics="logloss",
                                   random_state=RANDOM_STATE,
                                   n_jobs=-1
                                ),
        "params":{
            "clf__n_estimators":[100,300],
            "clf__max_depth":[3,6],
            "clf__learning_rate":[0.1, 0.01],
                              
        }
                            
    },
    "catboost":{
        "estimator": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
        "params":{
            "clf__iterations":[100,300],
            "clf__depth":[4,6],
            "clf__learning_rate":[0.1,0.01],
        }
    },
    "random_forest":{
        "estimator":RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params":{
            "cl__n_estimators":[100,300],
            "clf__max_depth":[None, 10, 20],
        }
    },
    "logistic_regression":{
        "estimator":LogisticRegression(
            random_state=RANDOM_STATE,
            solver="saga",
            max_iter=2000
        ),
        "params":{
            "clf__C":[0.1,1.0, 10.0],
            "clf__class_weight":[None, "balanced"]
        }

    }
}

#===============================
# FUNCTION TO TRAIN EACH MODEL
#===============================

def run_grid_search(name, estimator, param_grid):
    print("\n\n=========", name.upper(), "=========")

    """
    """

