import numpy as np
import dicom2nifti
import os, glob
from pydicom import dcmread
import argparse
#from utils import *
import pydicom
import dicom2nifti.settings as settings

settings.disable_validate_slice_increment()

parser = argparse.ArgumentParser(description='dicom2nifti')
parser.add_argument("--path_dcm_root",default="./data/raw_data/BV_CSFV/", type =str, help='full path to the DICOM folder')
parser.add_argument("--path_nii_root", default="./data/raw_data/BV_CSFV_nii/", type =str, help='full path to the NIFTI folder')

opt = parser.parse_args()

ls_fdr = os.listdir(opt.path_dcm_root)

for fdr in ls_fdr:
    patient = opt.path_dcm_root + '/' + fdr
    ls_visits = [x for x in os.listdir(patient) if 'MR' in x]
    if not os.path.exists(opt.path_nii_root + fdr):
        os.makedirs(opt.path_nii_root + fdr)
    for v in ls_visits:
        fdr_v = os.path.join(patient, v)
        #ls_ax_t2 = [x for x in os.listdir(fdr_v) if ('AX' in x or '3D' in x or 'Ax' in x or 'axial' in x) and 'T2' in x]
        ls_ax_t2 = [x for x in os.listdir(fdr_v) if ('_T2' in x or '_t2' in x or 'SSFSE' in x) and '_PD_' not in x and '_Loc' not in x and '-Loc' not in x  and '_LOC_' not in x and '-LOC_' not in x and 'Survey' not in x and 'mDixon' not in x and 'Real-time' not in x and  'cervical' not in x and 'UPPER_' not in x and 'LOWER_' not in x and 'THORACIC' not in x and 'thoracic' not in x and 'Series_No' not in x and 'sag' not in x and 'SAG' not in x and 'COR' not in x and 'cor' not in x and 'flair' not in x and 'FLAIR' not in x and 'SE_w_PD' not in x and 'SE_W_PD' not in x]
        for f in ls_ax_t2:
            fdr_dcm = os.path.join(fdr_v, f)
            print(fdr_dcm)
            if len(os.listdir(fdr_dcm)) < 18:
                continue
            path_nii = os.path.join(opt.path_nii_root, fdr, fdr +'_' + v + '_' + f + '.nii.gz')
            if not os.path.exists(path_nii):
                orientation = np.array([[1, 0, 0, 0, 0, -1], [1,0,0,0,1,0], [0, 1, 0, 0, 0, -1]]) # the second orientation represents axial
                if '3_PLANES' in fdr_dcm or '3_PLANE' in fdr_dcm or '3PLANES' in fdr_dcm or '3PLANE' in fdr_dcm or '3_PL' in fdr_dcm or '3-pl' in fdr_dcm:
                    for path_dcm in os.listdir(fdr_dcm):
                        dicom = pydicom.read_file(fdr_dcm + '/' + path_dcm)
                        is_axial = np.where(np.mean(abs(orientation - np.round(dicom.ImageOrientationPatient)), 1) == 0)[0] == 1
                        if not is_axial:
                            os.remove(fdr_dcm + '/' + path_dcm)
                dicom2nifti.convert_dicom.dicom_series_to_nifti(fdr_dcm, output_file=path_nii)


