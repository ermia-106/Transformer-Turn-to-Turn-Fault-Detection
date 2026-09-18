# Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
from Feature_Extraction_Inference import extract_features

# Model Features
feature_name = ['RMS_Ia','RMS_Ib','RMS_Ic',
                 'Peak_Ia','Peak_Ib','Peak_Ic',
                 'Std_Ia','Std_Ib','Std_Ic',
                 'Crest_Ia','Crest_Ib','Crest_Ic',
                 'THD_Ia','THD_Ib','THD_Ic',
                 'I0_Zero','I1_Pos','I2_Neg','Unbalance',
                 'Energy_a_D1','Energy_a_D2','Energy_a_D3','Energy_a_D4','Energy_a_D5',
                 'Energy_b_D1','Energy_b_D2','Energy_b_D3','Energy_b_D4','Energy_b_D5',
                 'Energy_c_D1','Energy_c_D2','Energy_c_D3','Energy_c_D4','Energy_c_D5',
                 'Entropy_D1_a','Entropy_D2_a','Entropy_D3_a','Entropy_D4_a','Entropy_D5_a',
                 'Var_D1_a','Var_D2_a','Var_D3_a','Var_D4_a','Var_D5_a',
                 'Std_D1_a','Std_D2_a','Std_D3_a','Std_D4_a','Std_D5_a',
                 'Load_Power']

# Import Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR,'..','Models','XGBoost_Model.pkl')

# Load XGBoost Model
_model = None
def load_model() :
    global _model
    if _model is None :
        with open(MODEL_PATH,'rb') as file :
            _model = pickle.load(file)
    if _model is None :
        raise RuntimeError('XGBoost model couldn\'t be loaded.')
    return _model

# Inference Function
def predict_fault(Ia,Ib,Ic,load_power) :
    Ia = np.asarray(Ia,dtype = float)
    Ib = np.asarray(Ib,dtype = float)
    Ic = np.asarray(Ic,dtype = float)
    if len(Ia) != 80 :
        raise ValueError('Ia must contain exactly 80 samples.')
    if len(Ib) != 80 :
        raise ValueError('Ib must contain exactly 80 samples.')
    if len(Ic) != 80 :
        raise ValueError('Ic must contain exactly 80 samples.')
    features49 = extract_features(Ia,Ib,Ic)
    if len(features49) != 49 :
        raise ValueError(f'Feature extraction returned {len(features49)} features instead of 49.')
    features50 = np.append(features49,float(load_power))
    x = pd.DataFrame([features50],columns = feature_name)
    model = load_model()
    prediction = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]
    healthy_probability = probabilities[0]
    fault_probability = probabilities[1]
    if prediction == 0 :
        result = 'Healthy'
    else :
        result = 'Fault'
    return {'prediction' : int(prediction),
            'result' : result,
            'healthy_probability' : float(healthy_probability),
            'fault_probability' : float(fault_probability),
            'features' : features50}