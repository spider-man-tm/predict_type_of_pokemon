import numpy as np
import pandas as pd
import os
import random
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,
                            accuracy_score,
                            recall_score,
                            precision_score,
                            f1_score)
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_generation_yml(generation):
    """return dict"""
    with open(f'config/generation{generation}_config.yml', 'rb') as f:
        pokemons = yaml.load(f, Loader=yaml.SafeLoader)
        return pokemons


def load_typ_yml():
    """return dict"""
    with open('config/pokemon_type_config.yml', 'rb') as f:
        typ = yaml.load(f, Loader=yaml.SafeLoader)
        return typ


def return_type(s, type_dic):
    """return list"""
    s = s[:-4]
    typ = s.split('_')[1:]
    typ = [type_dic[c] for c in typ]
    return typ


def type_to_array(typ):
    """
    return type arr
    shape: (18)
    """
    arr = np.zeros(18)
    for i in typ:
        arr[i] = 1
    return arr


def metric(true, pred):
    if np.sum(true)==1:
        if np.argmax(true)==np.argmax(pred):
            return 1
        else:
            return 0
    else:
        pred_max_idx = set(np.argpartition(-pred, 2)[:2])
        true_max_idx = set(np.argpartition(-true, 2)[:2])
        
        return 1 - len(true_max_idx - pred_max_idx)/2


def predict_to_binary(true_df, pred_df):
    true = true_df.iloc[:, 1:]
    pred = pred_df.copy()
    for i in range(len(true)):
        if np.sum(true.iloc[i, :].values)==1:
            arg = np.argmax(pred.iloc[i, :].values)
            for n in range(18):
                if n==arg:
                    pred.iloc[i, n] = 1
                else:
                    pred.iloc[i, n] = 0
        else:
            pred_max_idx = list(np.argpartition(-pred.iloc[i, :].values, 2)[:2])
            for n in range(18):
                if n in pred_max_idx:
                    pred.iloc[i, n] = 1
                else:
                    pred.iloc[i, n] = 0
    return pred


def confusion_df(true_df, pred_df):
    confusion = {}
    true_df = true_df.iloc[:, 1:]
    for i, c in enumerate(true_df.columns):
        true = true_df[c]
        pred = pred_df.iloc[:, i]
        score = list(confusion_matrix(true, pred).flatten())
        acc = accuracy_score(true, pred)
        rec = recall_score(true, pred)
        pre = precision_score(true, pred)
        f1 = f1_score(true, pred)
        score.append(acc)
        score.append(rec)
        score.append(pre)
        score.append(f1)
        confusion[i] = score

    confusion = pd.DataFrame(confusion)
    confusion.columns = [c for c in true_df.columns]
    confusion.index = ['True Negative', 'False Positive', 'False Negative', 'True Positive', 'Accuracy', 'Recall', 'Precision', 'F1_score']

    return confusion