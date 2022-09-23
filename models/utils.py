import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from numpy import *
import argparse

def load_data_V2(data_name):
    data_path = os.path.join('./data/', data_name)
    data = pd.read_table('{path}'.format(path = data_path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1).astype('category')
    data_x = data.values
    
    #data_y = y.cat.codes.values
    #print(data_x.size())
    data_y = np.zeros((data_x.shape[0],) ,dtype = np.int)
    idx = (y.values=='out')
    data_y[idx] = 1
    min_label = 1
    #min_label = 0 if zeros_counts<ones_counts else 1
    #class_mapping = {label:idx for idx,label in enumerate(set(y.values))}
    #data_y = y.map(class_mapping).values

    n_classes = int(max(data_y)+1)
    return data_x, data_y, min_label, n_classes

def load_data(data_name):
    data_path = os.path.join('./data/', data_name)
    data = pd.read_table('{path}'.format(path = data_path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1).astype('category')
    data_x = data.values
    data_y = y.cat.codes.values
    zeros_counts = (data_y==0).sum()
    ones_counts = (data_y==1).sum()
    min_label = 0 if zeros_counts<ones_counts else 1
    
    n_classes = int(max(data_y)+1)
    #print("minLabel_f=%d" % (min_label))
    return data_x, data_y, min_label, n_classes
 

def CSV_data_Loading(path):
    # loading data
    df = pd.read_csv(path) 
    
    labels = df['class']
    
    x_df = df.drop(['class'], axis=1)
    
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    x = np.array(x)
    labels = np.array(labels)
    
    return x, labels;

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ensemble.")
    parser.add_argument('--data_name', nargs='?', default='Annthyroid',
                        help='Input data name.')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--print_epochs', type=int, default=20,
                        help='print the loss per print_epochs.')
    parser.add_argument('--lr_g', type=float, default=0.01,
                        help='Learning rate of generator.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--num_class', type=int, default=2,
                        help='the number of classes in the task.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for each class.')
    parser.add_argument('--dis_layer', type=int, default=1,
                        help='hidden_layer number in dis.')
    parser.add_argument('--gen_layer', type=int, default=1,
                        help='hidden_layer number in gen.')
    parser.add_argument('--ensemble_num', type=int, default=10,
                        help='the number of dis in ensemble.')
    parser.add_argument('--ir', type=float, default=5,
                        help='hidden_layer number in dis.')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='if GPU used')
    parser.add_argument('--SN_used', type=bool, default=True,
                        help='if spectral Normalization used')
    parser.add_argument('--init_type', nargs='?', default="ortho",
                        help='init method for both gen and dis, including ortho,N02,xavier')
    parser.add_argument('--log_path', type=str, default="./log/CycleGAN/",
                        help='the dir to save log')
    parser.add_argument('--print', type=bool, default=True)
    return parser.parse_args()

def ensemble_parse_args():
    parser = argparse.ArgumentParser(description="Run Ensemble.")
    parser.add_argument('--data_name', nargs='?', default='Annthyroid',
                        help='Input data name.')
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--print_epochs', type=int, default=20,
                        help='print the loss per print_epochs.')
    parser.add_argument('--lr_g', type=float, default=0.01,
                        help='Learning rate of generator.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--num_class', type=int, default=2,
                        help='the number of classes in the task.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for each class.')
    parser.add_argument('--kFold', type=int, default=10,
                        help='k for k-fold cross validation.')
    parser.add_argument('--dim_z', type=int, default=64,
                        help='dim for latent noise.')
    parser.add_argument('--dis_layer', type=int, default=1,
                        help='hidden_layer number in dis.')
    parser.add_argument('--gen_layer', type=int, default=1,
                        help='hidden_layer number in gen.')
    parser.add_argument('--ensemble_num', type=int, default=10,
                        help='the number of dis in ensemble.')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='if GPU used')
    parser.add_argument('--input_path', type=str, default='data_csv')
    parser.add_argument('--SN_used', type=bool, default=True,
                        help='if spectral Normalization used')
    parser.add_argument('--init_type', nargs='?', default="ortho",
                        help='init method for both gen and dis, including ortho,N02,xavier')
    parser.add_argument('--log_path', type=str, default="./log",
                        help='the dir to save log')

    parser.add_argument('--over_sample', type=str, default="MOTE",
                        help='the dir to save log')


    parser.add_argument('--print', type=bool, default=True)
    return parser.parse_args()

