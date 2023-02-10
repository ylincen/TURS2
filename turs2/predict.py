import pickle
import pandas as pd

from turs2.Ruleset import *


fh = open("rulelist.pkl", "rb")
ruleset = pickle.load(fh)
fh.close()
X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_StandardScaler_meanimputation_missing_features_dropped.csv')
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')

res = predict_rulelist(ruleset, X_test, y_test)