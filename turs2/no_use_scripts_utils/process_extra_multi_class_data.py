import os
import numpy as np
import pandas as pd


files = os.listdir("../extra_multi_class_data_raw/")

d_list = []
for file in files:
    if file == ".DS_Store":
        continue
    d = pd.read_csv("../extra_multi_class_data_raw/" + file, header=None)
    d_list.append(d)

datasets_with_header = ["Vehicle", "DryBeans"]
datasets_without_header = ["glass", "pendigits", "HeartCleveland"]

