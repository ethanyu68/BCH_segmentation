import os
import pandas as pd
import numpy as np

data_fdr = './data/npy/etv167_NIFTI_pre_reviewed_0219/'
ls_patients = os.listdir(data_fdr)
permuted_idx = np.random.permutation(len(ls_patients))
dict_fold = {'fold1':[], 'fold2':[], 'fold3':[], 'fold4':[], 'fold5':[]}
for i in range(len(ls_patients)//5 *5):
    fold = i%5
    patient = ls_patients[permuted_idx[i]]
    dict_fold['fold' + str(fold+1)].append(patient)

df = pd.DataFrame(dict_fold)
df.to_csv(data_fdr  + 'patients_list_5folds.csv')