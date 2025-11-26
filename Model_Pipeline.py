# =========================================
# CREDIT DEFAULT PREDICTION — FINAL PIPELINE
# =========================================

# =============
# IMPORTS & SAFE ENV
# =============

# Set environment variables early to avoid pandas/pyarrow import hangs.
import os
os.environ["PANDAS_COPY_ON_WRITE"] = "1"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

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
    roc_curve
)

# SAVING MODELS
import joblib

# VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# Create folders for plots (safe)
os.makedirs("plots", exist_ok=True)


# ============================
# 1) LOAD CLEANED DATA
# ============================

# Grade 3: load file
data = pd.read_csv("cleaned_preprocessed_loans_data.csv")

# Grade 3: fix column names
data.columns = data.columns.str.replace(" ", "_")

# ============================
# 1.a SAFETY: REMOVE POTENTIAL LEAKAGE & DUPLICATES
# ============================
# Grade 3: make sure the model cannot see the answer directly
# If your dataframe still has the original textual loan_status column, drop it.
if 'loan_status' in data.columns:
    data = data.drop(columns=['loan_status'])

# If there are other columns that directly include target-like information, drop them here.
# Grade 3: common leak columns, update this list if you know other leak column names:
leak_cols = [c for c in data.columns if c.lower().strip() in ('default_flag', 'charged_off_flag', 'paid_flag', 'status_flag')]
if leak_cols:
    data = data.drop(columns=leak_cols)

# Grade 3: Remove exact duplicate rows to avoid memorization.
dups = data.duplicated().sum()
if dups > 0:
    data = data.drop_duplicates().reset_index(drop=True)

# ============================
# 1.b TARGET (ensure exists)
# ============================
# Grade 3: ensure the binary target column exists and is integer type
if 'loan_status_binary' not in data.columns:
    raise ValueError("Expected column 'loan_status_binary' not found. Create it before running pipeline.")

data['loan_status_binary'] = data['loan_status_binary'].astype(int)

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
        "estimator": LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0
        ),
        "params": {
            "clf__n_estimators": [100, 300],
            "clf__num_leaves": [31, 50],
            "clf__learning_rate": [0.1, 0.01],
        },
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
            "clf__learning_rate": [0.1, 0.01],
        },
    },

    "catboost": {
        "estimator": CatBoostClassifier(
            verbose=0, random_state=RANDOM_STATE
        ),
        "params": {
            "clf__iterations": [100, 300],
            "clf__depth": [4, 6],
            "clf__learning_rate": [0.1, 0.01],
        },
    },

    "random_forest": {
        "estimator": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0
        ),
        "params": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 10, 20],
        },
    },

    "logistic_regression": {
        # Grade 3: increase max_iter so solver can finish on large data
        "estimator": LogisticRegression(
            random_state=RANDOM_STATE,
            solver="saga",
            max_iter=5000
        ),
        "params": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
    },
}


# ============================
# Helper plotting functions (small additions)
# ============================
# these are helpers to draw and save plots for each model.
def plot_and_save_confusion(cm, title, filepath):
    # Grade 3: show the confusion numbers in a nice box and save it.
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_and_save_roc(y_true, y_scores, title, filepath):
    # drawing ROC curve and show AUC number, then save the picture.
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_and_save_pr(y_true, y_scores, title, filepath):
    # drawing precision-recall curve and show PR AUC, then save the picture.
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# ============================
# 5) TRAINING FUNCTION
# ============================

def run_grid_search(name, estimator, param_grid):
    """
    Trains a single estimator inside ImbPipeline:
    [StandardScaler -> SMOTE -> clf]
    StandardScaler is included so Logistic Regression converges.
    SMOTE is after scaling and is inside pipeline to avoid leakage.
    """
    print("\n\n===========================")
    print(f" TRAINING: {name.upper()}")
    print("===========================\n")

    # Grade 3: pipeline scales numbers, creates synthetic samples, then trains model
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", estimator),
    ])

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )

    # Train model
    gs.fit(X_train, y_train)

    print("\nBest parameters:", gs.best_params_)
    print(f"Best CV Recall: {gs.best_score_:.4f}")

    # Save model (existing behavior)
    joblib.dump(gs.best_estimator_, f"{name}_best_model.pkl")
    print(f"Model saved as {name}_best_model.pkl")

    # ============================
    # VALIDATION PERFORMANCE (existing)
    # ============================

    print("\nVALIDATION PERFORMANCE")

    # use best estimator for validation predictions
    best_pipe = gs.best_estimator_

    y_val_pred = best_pipe.predict(X_val)

    # Grade 3: try to get probability for positive class. If not available, use decision score.
    try:
        y_val_proba = best_pipe.predict_proba(X_val)[:, 1]
    except Exception:
        try:
            val_scores = best_pipe.decision_function(X_val)
            # convert scores to 0-1 via min-max (simple)
            y_val_proba = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-8)
        except Exception:
            y_val_proba = y_val_pred.copy()

    print(classification_report(y_val, y_val_pred, digits=4))
    # guard: roc_auc_score requires at least two unique labels in y_val_proba; wrap in try
    try:
        print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))
    except Exception:
        print("Validation ROC AUC: could not compute (need more than one unique score)")

    precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
    print("Validation PR AUC:", auc(recall, precision))

    # ============================
    # PLOT & SAVE validation figures (NEW)
    # ============================
    # Grade 3: make and save confusion, ROC and PR plots for validation.
    cm_val = confusion_matrix(y_val, y_val_pred)
    plot_and_save_confusion(cm_val, f"{name} - Validation Confusion Matrix", f"plots/{name}_val_confusion.png")
    # only plot ROC/PR when we have continuous scores
    if len(np.unique(y_val_proba)) > 1:
        plot_and_save_roc(y_val, y_val_proba, f"{name} - Validation ROC Curve", f"plots/{name}_val_roc.png")
        plot_and_save_pr(y_val, y_val_proba, f"{name} - Validation PR Curve", f"plots/{name}_val_pr.png")

    # ============================
    # TEST PERFORMANCE for this model (NEW)
    # ============================
    # Grade 3: now test the best version of this model on the test set and save metrics + plots.
    y_test_pred_local = best_pipe.predict(X_test)

    try:
        y_test_proba_local = best_pipe.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            test_scores = best_pipe.decision_function(X_test)
            y_test_proba_local = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
        except Exception:
            y_test_proba_local = y_test_pred_local.copy()

    print("\nTEST PERFORMANCE FOR THIS MODEL")
    print(classification_report(y_test, y_test_pred_local, digits=4))
    print("Test Recall (this model):", recall_score(y_test, y_test_pred_local))
    print("Test Precision (this model):", precision_score(y_test, y_test_pred_local))
    print("Test F1 (this model):", f1_score(y_test, y_test_pred_local))
    try:
        print("Test ROC AUC (this model):", roc_auc_score(y_test, y_test_proba_local))
    except Exception:
        print("Test ROC AUC (this model): could not compute (need more than one unique score)")

    precision_t, recall_t, _ = precision_recall_curve(y_test, y_test_proba_local)
    print("Test PR AUC (this model):", auc(recall_t, precision_t))

    # Grade 3: save test plots
    cm_test = confusion_matrix(y_test, y_test_pred_local)
    plot_and_save_confusion(cm_test, f"{name} - Test Confusion Matrix", f"plots/{name}_test_confusion.png")
    if len(np.unique(y_test_proba_local)) > 1:
        plot_and_save_roc(y_test, y_test_proba_local, f"{name} - Test ROC Curve", f"plots/{name}_test_roc.png")
        plot_and_save_pr(y_test, y_test_proba_local, f"plots/{name}_test_pr.png")

    return best_pipe, gs


# ============================
# 6) RUN TRAINING FOR ALL MODELS
# ============================

results = {}

for name, cfg in estimators_and_grids.items():
    try:
        best_model, gs_obj = run_grid_search(
            name, cfg["estimator"], cfg["params"]
        )
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
        "best_params": info["grid"].best_params_,
    })

summary_df = pd.DataFrame(summary).sort_values(
    by="best_score", ascending=False
)

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
try:
    y_test_proba = top_model.predict_proba(X_test)[:, 1]
except Exception:
    try:
        test_scores = top_model.decision_function(X_test)
        y_test_proba = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
    except Exception:
        y_test_proba = y_test_pred.copy()

print(classification_report(y_test, y_test_pred, digits=4))
print("Test Recall:", recall_score(y_test, y_test_pred))
print("Test Precision:", precision_score(y_test, y_test_pred))
print("Test F1:", f1_score(y_test, y_test_pred))
try:
    print("Test ROC AUC:", roc_auc_score(y_test, y_test_proba))
except Exception:
    print("Test ROC AUC: could not compute (need more than one unique score)")

precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
print("Test PR AUC:", auc(recall, precision))


# ============================
# 9) CONFUSION MATRIX (for the best model)
# ============================

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(5, 4))
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