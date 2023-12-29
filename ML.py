import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from sklearn.manifold import TSNE
from torch.distributions import Normal
from matplotlib.animation import FuncAnimation
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn.svm import SVR
from scipy.stats import norm


from sklearn.model_selection import RandomizedSearchCV


from sklearn.model_selection import train_test_split







##### Data loading #####

data = pd.read_csv('./result.csv', low_memory=False)
mof_name = data['mof'].values
target_columns = ['S_pX']
columns_drop = ['mof', 'nmolecule', 'nmolecule1', 'nmolecule2', 'nmolecule3', 'nmolecule4',
                'std_nmolecule', 'std_nmolecule1', 'std_nmolecule2', 'std_nmolecule3', 'std_nmolecule4',
                'atomic_mass', 'S_pX', 'S_mX', 'S_oX', 'S_EB', 'N_pX', 'N_mX', 'N_oX', 'N_EB']
X_orignal = data.drop(columns=columns_drop)
y_orignal = data[target_columns]
target_npx = ['N_pX']
fontdict_t = {'fontsize': 14, 'weight': 'bold', 'ha': 'center'}
fontdict_x = {'fontsize': 12, 'weight': 'bold', 'ha': 'center'}
fontdict_y = {'fontsize': 12, 'weight': 'bold', 'va': 'baseline', 'ha': 'center'}
y_npx = data[target_npx]
##### Data processing #####
scale = MinMaxScaler()
X_scaled = scale.fit_transform(X_orignal)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_scale)

X_scale = pd.DataFrame(X_scaled, columns=X_orignal.columns)
#y_scale = scaler.fit_transform(y)
#y_scale = pd.DataFrame(y_scale , columns = y.columns)
#y_scale = scaler.fit_transform(y_orignal)
y_scale = pd.DataFrame(y_orignal, columns=y_orignal.columns)

X_train = X_scale.values
y_train = y_scale.values
y_train = np.reshape(y_train, (np.size(y_train), 1))
print("shape of X:", np.shape(X_train))
print("shape of y:", np.shape(y_train))
X = X_train
y = y_train
y_npx  = y_npx.values
y_npx = torch.from_numpy(y_npx)
#X_test = torch.from_numpy(X_test)


#tsne = PCA(n_components=2, random_state=1)

#original_tsne = tsne.fit_transform(X_train)

#def EntropySearch():

def ML(X, y, niter, nsamp, batch, which_model):
    param_grid_rf = {
    'n_estimators': [100, 200, 300, 400],      # hyperparameters for RFR
    'max_depth': [None, 10, 20, 40, 60],
    'max_features': [28, 'sqrt', 'log2']
    }
    param_grid_xgb = {
    'n_estimators': [100,200, 300, 400],      # hyperparameters for XGBR
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [None, 10, 20, 40, 60],
    }
    param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf'],      # hyperparameters for SVR
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5]
    }
    y_n = y.copy()
    get_index = np.random.choice(np.arange(len(y)),size = nsamp,replace =False)
    index_max = np.empty(len(y))
    batch_index = []




    for i in range(niter):
       #print("iteration:", i, end="\r")





       if which_model == 'XGBR':
           xgb = XGBRegressor(random_state=42)
           xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid_xgb, n_iter=50, cv=5, random_state=42, n_jobs=-1)
           xgb_random.fit(X[get_index, :], y[get_index])
           best_params_xgb = xgb_random.best_params_
           model = XGBRegressor(**best_params_xgb, random_state=42)


       elif which_model == 'RF':
           rf = RandomForestRegressor(random_state=1)
           rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf, n_iter=50, cv=5, random_state=42, n_jobs=-1)
           rf_random.fit(X[get_index, :], y[get_index])
           best_params_rf = rf_random.best_params_
           model = RandomForestRegressor(**best_params_rf, random_state=1)

       elif which_model == 'SVR':
           svr = SVR()
           svr_random = RandomizedSearchCV(estimator=svr, param_distributions=param_grid_svr, n_iter=50, cv=5, random_state=42, n_jobs=-1)
           svr_random.fit(X[get_index, :], y[get_index])
           best_params_svr = svr_random.best_params_
           model = SVR(**best_params_svr)


       model.fit(X[get_index, :], y[get_index])
       y_pred = model.predict(X)
       sorted_index = np.argsort(y_pred)
       for index in sorted_index:
           if not index.item() in get_index:
              index_max = index.item()
              break
       batch_index.append(index_max)
       get_index = np.concatenate((get_index, [index_max]))




    return get_index, batch_index


ml_result = dict()
ml_result['get_index'] = []
#ml_result['batch_index'] = []


which_model = 'XGBR'
nruns = 100
niter =500
nsamp = 10
batch_size  = 10
num_runs = 100
for i in range(num_runs):

    get_index, batch_index = ML(X, y, niter, nsamp, batch_size, which_model)
    ml_result['get_index'].append(get_index)
    #ml_result['batch_index'].append(batch_index)



result_file = f'./ML_single_test_1_initiate_with_{which_model}.pkl'

# Save the results to a file
with open(result_file, 'wb') as file:
    pickle.dump(bo_result, file)
