import numpy as np
import pandas as pd
import torch
import multiprocessing
import matplotlib.pyplot as plt
import pickle
from botorch.models import SingleTaskGP, FixedNoiseGP,MultiTaskGP, ModelListGP, MultiTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from sklearn.manifold import TSNE
from torch.distributions import Normal
from matplotlib.animation import FuncAnimation
from PIL import Image
from sklearn.decomposition import PCA
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.svm import SVR
from scipy.stats import norm
filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler







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

y_scale = pd.DataFrame(y_orignal, columns=y_orignal.columns)

X_train = X_scale.values
y_train = y_scale.values
y_train = np.reshape(y_train, (np.size(y_train), 1))
print("shape of X:", np.shape(X_train))
print("shape of y:", np.shape(y_train))
X = torch.from_numpy(X_train)
y = torch.from_numpy(y_train)
y_npx  = y_npx.values
y_npx = torch.from_numpy(y_npx)
#X_test = torch.from_numpy(X_test)

#tsne = PCA(n_components=2, random_state=1)

#original_tsne = tsne.fit_transform(X_train)
def EI(mean,sigma, epsilon, y_best ): # expected improvement

    ui = (mean - y_best) / sigma
    normal = Normal(torch.zeros_like(ui), torch.ones_like(ui))
    ucdf = normal.cdf(ui)
    updf = torch.exp(normal.log_prob(ui))
    ei = sigma * updf + (mean - y_best -epsilon) * ucdf
    return ei



def al(X, y, acq, niter, nsamp, batch, parallel_acq):
    batch_index = []
    explore_exploit = np.empty((niter, 2))
    y_n = y.copy() #creating copy
    explore_exploit[:] = np.NaN
    get_index = np.random.choice(np.arange(len(y)),size = nsamp,replace =False)
    X_unsqueezed = X.unsqueeze(1)
    index_max = np.empty(len(y))
    for i in range(niter):
        print("iteration:", i, end="\r")


        y_s = y[get_index]
        #y_n = y_npx[get_index]


        gp = SingleTaskGP(X[get_index, :], y_s)

       # gp_s = SingleTaskGP(X[get_index, :], y_s)
       # gp_n = SingleTaskGP(X[get_index, :], y_n)
       # gp_s = [gp_s]
       # gp_n = [gp_n]

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        #if acq == "UCB":
        #    acquisition_function = UpperConfidenceBound(model, beta=0.1)
        #    with torch.no_grad():
        #        acquisition_values = acquisition_function.forward(X_unsqueezed)

            # Select the next batches point with the highest acquisition value



        acquisition_function = ExpectedImprovement(gp, best_f=y_s.max().item(), maximize=True) # acquisition function
        with torch.no_grad():
            acquisition_values = acquisition_function.forward(X_unsqueezed)
        index_sorted = acquisition_values.argsort(descending=True)

        for index in index_sorted:
            if not index.item() in get_index:
                index_max = index.item()
                break
        get_index = np.concatenate((get_index, [index_max])) # acquired next point for evaluation
        batch_index.append(index_max)


    return get_index, batch_index



al_result = dict()
al_result['get_index'] = []
#al_result['batch_index'] = []
#parallel_acq = 'KB'
acq = 'EI'

niter =500
nsamp = 10
#batch_size  = 5
num_runs = 100
for i in range(num_runs):
    print(i)
    get_index, batch_index = al(X, y, acq, niter, nsamp, batch_size, parallel_acq)
    al_result['get_index'].append(get_index)
    #al_result['batch_index'].append(batch_index)


result_file = f'./AL_test_1_initiate_with_{nsamp}.pkl'

# Save the results to a file
with open(result_file, 'wb') as file:
    pickle.dump(al_result, file)
