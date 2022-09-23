# -*- coding: utf-8 -*-
import os
import sys
from time import time

from numpy.core.numeric import count_nonzero
from sklearn.utils import shuffle

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import pandas as pd

import torch

from models.pyod_utils import min_max_normalization, AUC_and_Gmean, get_measure
from models.pyod_utils import precision_n_scores, gmean_scores
from models.utils import parse_args
from sklearn.metrics import roc_auc_score
from models.EX_GAN_MINIST import EX_GAN

from models.Imbalanced_MINIST import MNIST
from torchvision import transforms

n_folds = 10
result_dir = "./results/"

df_columns = ['IR', 'AUC', 'F1', 'time', 'time_std']
roc_df = pd.DataFrame(columns=df_columns)

# initialize the container for saving the results
result_df = pd.DataFrame(columns=df_columns)
args = parse_args()


#define the dataloader
    
ir = args.ir

# construct containers for saving results
roc_list = [ir]

time_mat = np.zeros([n_folds, 1])
roc_mat = np.zeros([n_folds, 1])
fscore_mat = np.zeros([n_folds, 1])

#repeat the k-fold cross validation n_iterations times
count = 0
for _ in range(n_folds):
    t0 = time()
    #todo: add my method
    cb_gan = EX_GAN(args)
    auc_train, f_train, auc_test, f_test = cb_gan.fit()
    #clf.fit(X_train_norm)
    #test_scores = clf.decision_function(X_test_norm)
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    print('AUC:{roc}, F_score:{f_score}, train_AUC:{train_auc} train_f_score:{train_f_score}'  
                'execution time: {duration}'.format(roc=auc_test, f_score=f_test, train_auc=auc_train, train_f_score=f_train, duration=duration))

    time_mat[count, 0] = duration
    roc_mat[count, 0] = auc_test
    fscore_mat[count, 0] = f_test
    count += 1

roc_list = roc_list + np.mean(roc_mat, axis=0).tolist() + np.std(roc_mat, axis=0).tolist() + \
                    np.mean(fscore_mat, axis=0).tolist() + np.std(fscore_mat, axis=0).tolist() + \
                    np.mean(time_mat, axis=0).tolist() + np.std(time_mat, axis=0).tolist()
temp_df = pd.DataFrame(roc_list).transpose()
temp_df.columns = df_columns
roc_df = pd.concat([roc_df, temp_df], axis=0)

# Save the results for each run
save_path1 = os.path.join(result_dir, 'EX-GAN.csv')
roc_df.to_csv(save_path1, index=False, float_format='%.3f')