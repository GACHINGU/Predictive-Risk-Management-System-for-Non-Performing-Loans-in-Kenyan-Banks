# =========================================
# CREDIT DEFAULT PREDICTION — FINAL PIPELINE
# CLEAN, SAFE, FULLY COMMENTED
# =========================================

# =============
# IMPORTS
# =============

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# PIPELINE + CV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

# METRICS
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
)

# SAVING MODELS
import joblib

# VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42



# ============================
# 1) LOAD CLEANED DATA
# ============================

data = pd.read_csv(r"C:\Users\ADMIN\Documents\OWN_Projects\Predictive-Risk-Management-System-for-Non-Performing-Loans-in-Kenyan-Banks\Notebook\cleaned_preprocessed_loans_data.csv")

# X = features (input data)
X = data.drop(columns=["loan_status_binary"])

# y = target (what we want to predict)
y = data["loan_status_binary"]



# ============================
# 2) TRAIN / VALIDATION / TEST SPLIT
# ============================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)



# ============================
# 3) CROSS VALIDATION RULE
# ============================

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

scoring = "recall"



# ============================
# 4) MODEL LIST + FIXED PARAM GRIDS
# ============================

estimators_and_grids = {
    "lightgbm": {
        "estimator": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=0),
        "params": {
            "clf__n_estimators": [100, 300],
            "clf__num_leaves": [31, 50],
            "clf__learning_rate": [0.1, 0.01]
        }
    },

    "xgboost": {
        "estimator": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        "params": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 6],
            "clf__learning_rate": [0.1, 0.01]
        }
    },

    "catboost": {
        "estimator": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
        "params": {
            "clf__iterations": [100, 300],
            "clf__depth": [4, 6],
            "clf__learning_rate": [0.1, 0.01],
        }
    },

    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=0),
        "params": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 10, 20],
        }
    },

    "logistic_regression": {
        "estimator": LogisticRegression(
            random_state=RANDOM_STATE,
            solver="saga",
            max_iter=2000
        ),
        "params": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"]
        }
    }
}



# ============================
# 5) TRAINING FUNCTION
# ============================

def run_grid_search(name, estimator, param_grid):

    print("\n\n===========================")
    print(f" TRAINING: {name.upper()}")
    print("===========================\n")

    pipe = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", estimator)
        ])

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )

    gs.fit(X_train, y_train)

    print("\nBest parameters:", gs.best_params_)
    print(f"Best CV Recall: {gs.best_score_:.4f}")

    joblib.dump(gs.best_estimator_, f"{name}_best_model.pkl")
    print(f"Model saved as {name}_best_model.pkl")

    print("\nVALIDATION PERFORMANCE")

    y_val_pred = gs.predict(X_val)
    y_val_proba = gs.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, y_val_pred, digits=4))
    print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))

    precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
    print("Validation PR AUC:", auc(recall, precision))

    return gs.best_estimator_, gs



# ============================
# 6) RUN TRAINING FOR ALL MODELS
# ============================

results = {}
for name, cfg in estimators_and_grids.items():
    try:
        best_model, gs_obj = run_grid_search(name, cfg["estimator"], cfg["params"])
        results[name] = {"estimator": best_model, "grid": gs_obj}
    except Exception as e:
        print(f"FAILED FOR {name}: {e}")



# ============================
# 7) FIND BEST MODEL
# ============================

summary = []
for name, info in results.items():
    summary.append({
        "model": name,
        "best_score": info["grid"].best_score_,
        "best_params": info["grid"].best_params_
    })

summary_df = pd.DataFrame(summary).sort_values(by="best_score", ascending=False)
print("\nMODEL RANKING:")
print(summary_df)

top_model_name = summary_df.iloc[0]["model"]
top_model = results[top_model_name]["estimator"]

print("\nBEST MODEL IS:", top_model_name)



# ============================
# 8) FINAL TEST SET EVALUATION
# ============================

print("\nFINAL TEST METRICS (REAL LIFE PERFORMANCE)")

y_test_pred = top_model.predict(X_test)
y_test_proba = top_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_test_pred, digits=4))
print("Test Recall:", recall_score(y_test, y_test_pred))
print("Test Precision:", precision_score(y_test, y_test_pred))
print("Test F1:", f1_score(y_test, y_test_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_test_proba))

precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
print("Test PR AUC:", auc(recall, precision))



# ============================
# 9) CONFUSION MATRIX
# ============================

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{top_model_name} - Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# ============================
# 10) SAVE FINAL MODEL
# ============================

joblib.dump(top_model, f"top_model_{top_model_name}.pkl")
print("\nFinal model saved successfully!")