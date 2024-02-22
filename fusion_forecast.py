import numpy as np
import pandas as pd


from preprocess.feature_engineer import FeatureEngineering

read_csv = False

if read_csv:
    df = pd.read_csv("data/CF_physics_April2_23.csv")
    df.reset_index(inplace=True)
    feat_eng_obj  = FeatureEngineering(df)
    data = feat_eng_obj.feature_engineer()
    data.to_pickle("fusion_df.pkl", compression="zip")
else:
    data = pd.read_pickle("fusion_df.pkl", compression="zip")







a =1

