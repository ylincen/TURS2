from sklearn.tree import DecisionTreeClassifier

import copy

import time
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
colnames = X.columns


X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_no_scale.csv')
X = X.loc[:, colnames]
y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')
X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_no_scale.csv')
X_test = X_test.loc[:, colnames]
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')

X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, test_size=0.3)

clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

val_roc_auc = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict_proba(X_val)
    val_roc_auc.append(roc_auc_score(y_val, y_hat[:, 1]))

best_ccp_alpha = np.argmax(val_roc_auc)
best_clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alphas[best_ccp_alpha])
best_clf.fit(X, y)

y_test_pred = best_clf.predict_proba(X_test)
y_train_pred = best_clf.predict_proba(X)

print("CART AUC Train: ", roc_auc_score(y, y_train_pred[:, 1]))
print("CART AUC Test: ", roc_auc_score(y_test, y_test_pred[:, 1]))
print("CART brier score Train/Test: ", brier_score_loss(y, y_train_pred[:,1]), brier_score_loss(y_test, y_test_pred[:,1]))

preds_test = []
counter = []
for pred_unique in np.unique(y_test_pred[:, 1]):
    indices = np.where(abs(y_test_pred[:, 1] - pred_unique) < 1e-5)[0]
    counter.append( len(indices) )
    pred_on_test_data = np.mean(y_test.to_numpy().ravel()[indices] == 1)
    preds_test.append(pred_on_test_data)

print(pd.DataFrame({"probs est. test": preds_test, "probs pred.": np.unique(y_test_pred[:, 1]), "counter": counter}))
