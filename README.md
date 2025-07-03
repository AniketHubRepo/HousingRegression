# HousingRegression
**MLOps Assignment 1: Boston Housing Price Prediction**

---
## Overview

- Predict house prices using the Boston Housing dataset.
- Compare Linear Regression, Random Forest, and SVR on MSE & R².
- Extend with hyperparameter tuning via 3 Models (Ridge, Random Forest, SVR).
- Automate everything via GitHub Actions (CI pipeline).

---
## Repo Layout

| File/Folder                  | Description            |
|------------------------------|------------------------|
| .github/workflows/ci_pipe.yml| CI pipeline definition |
| utils.py                     | Data loading & metrics |
| regression.py                | Model training/scripts |
| requirements.txt             | Python dependencies    |

---

## Steps

### 1. Conda Environment & Install

conda create -n housing_as1    
conda activate housing_as1    
conda install pip    
pip install -r requirements.txt    
python regression.py    

### 2. Git Workflow
#### #0- Initially just create a repo on git named “HousingRegression”

#### #1- Than first, Create a reg_branch
git checkout -b reg_branch    
git add –all    
git commit -m "Add the regression models"    
git push origin reg_branch    

#### #2- Now, Merge reg_branch to main
git checkout main    
git merge reg_branch    
git push origin main    

Now, I did some codes changes in regression.py file i.e., added the codes for tuned models    

#### #3- Then, Create hyper_branch
git checkout -b hyper_branch    
git add –all    
git commit -m "Adding the hyperparameter tuning"    
git push origin hyper_branch    

#### #4- Finally, Merge hyper_branch to main
git checkout main    
git merge hyper_branch    
git push origin main    


### 3.Model Comparisons
Model	             MSE	R²    
Linear Regression	24.29	0.67    
Random Forest	    8.13	0.89    
SVR	                52.84	0.28    

### 4.Hyperparameter Tuning Summary
Ridge:    
{'alpha': 0.1, 'fit_intercept': True, 'solver': 'auto'}    

Random Forest:    
{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 50}    

SVR:    
{'C': 10, 'gamma': 'scale', 'kernel': 'linear'}    

### 5.CI/CD Pipeline
Runs on every push to any branch (main, reg_branch, hyper_branch)    
Installs dependencies, runs regression.py, fails if errors are encountered    
See Actions tab in GitHub for logs     

### 6.Details
Assignment Title: Housing Price Prediction    
Name: Aniket Srivastava    
Email: g24ai1077@iitj.ac.in    
Roll No: G24AI1077    

### 7. Project Structure
HousingRegression/    
├── .github/workflows/ci_pipe.yml    
├── utils.py    
├── regression.py    
├── requirements.txt    
└── README.md     

### 8.Summary
Project implements a complete machine learning pipeline for predicting Boston house prices using classical regression models—Linear Regression, Random Forest, and SVR. It covers environment setup, modular code structure, GitHub-based workflow management with branching, model comparison using MSE and R², and extends to automated hyperparameter tuning. All code and experiments are integrated with GitHub Actions for CI, ensuring the automation.