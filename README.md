Active Learning Regression Framework
This repository contains Python code to facilitate active learning and evolutionary optimization workflows for regression modeling tasks.

Contents
There are 3 main scripts:

1. Gaussian Process Active Learning
Implements an active learning loop for regression using Gaussian Process models and Expected Improvement acquisition.
Loads data from 'result.csv', preprocesses features, and splits into train/test.
Iteratively selects the most informative point to query based on the current GP model.
Saves index history of selected points into a pickle file.
2. Multi-Model Active Learning
Performs active learning for regression using schemes like Random Forest, XGBoost, SVR.
Loads data, scales features, and searches hyperparameters for each model.
Chooses new data points to query based on prediction uncertainty.
Saves index history of selected points into a pickle file.
3. Evolutionary Optimization
Implements a CMA-ES evolutionary strategy for regression optimization.
Loads and preprocesses data and prepares input/output pairs.
Generates and selects candidate solutions based on proximity to acquired data points.
Saves index history of selected points into a pickle file.
Usage
Update result.csv with your regression dataset
Install dependencies like Scikit-Learn, CMA, etc
Execute scripts in sequence to run simulations
Output pickle files contain indices of selected points
Next Steps
Tweak hyperparameters and parameters to customize simulations
Analyze index histories to evaluate selection strategies
Use index history to refine models with new queried data points
![Acquisition_animation_vf](https://github.com/aqibm08/al_vs_ml/assets/131434938/1f37f936-44cc-4386-ae47-dae4aa534131)
