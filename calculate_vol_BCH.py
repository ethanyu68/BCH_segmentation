import os
import nibabel
import numpy as np
import pandas as pd
import argparse
from datetime import date
import xarray as xr
import xarray_einstats.stats as xr_stats
from scans_timepoints import check_date
from anthro_growthstandards_main.growthstandards import rv_sel, rv_interp, GrowthStandards
from utils import *

parser = argparse.ArgumentParser(
        description="Process some integers.")
parser.add_argument("--src",default='/home/ethanyu/data/seg_results/EAF_FOHR_STUDY/', help='the folder that contains the NIFTI segmentation files')
parser.add_argument("--dst",default='/home/ethanyu/data/seg_results/eaf_fohr_volume.xlsx', help='the path to the location to save the volume info table.')
parser.add_argument("--info",default='/home/ethanyu/data/Extra_axial_changes.xlsx', help='the spreadsheet that has all the information of the patients.')
parser.add_argument("--info_sheet",default='Sheet1', help='the spreadsheet that has all the information of the patients.')

parser.add_argument("--select_mrn", default=0, help='1:select patients listed in <MRN>; 0: select all patients in the src folder.')
parser.add_argument("--list_scans",default='/home/ethanyu/data/etv_for_longitudinal_0820.xlsx', help='the xlsx file that contains <MRN> and <mri_date>')
parser.add_argument("--select_date", default=0, help='1: select scans in <mri_date>; 0: select all scans of the selected patients.')

parser.add_argument("--col_mrn", default='studyid', help='the column in info sheet about MRN')
parser.add_argument("--col_dob", default='DOB_corr', help='the column in info sheet about patient date of birth')
parser.add_argument("--col_date_sx", default='date_sx', help='the column in info sheet about date of surgery')
parser.add_argument("--col_gender", default='Gender', help='the column in info sheet about patient gender')

args = parser.parse_args()
def main():
    root = args.src
    # load info table
    info = pd.read_excel(args.info, sheet_name=args.info_sheet)
    ls_fdr = [x for x in os.listdir(root + 'NIFTI/')]
    table_vol = {'MRN':[], 'filename':[],'mri_date':[], 'dob_corr':[], 'age_corr':[], 'date_surgery':[],'days_postop':[],'gender':[],
                 'CSF':[], 'ventricle':[], 'EAF':[], 'brain':[]}
    if args.select_mrn:
        ls_scans = pd.read_excel(args.list_scans) # this dataframe contains the scans of interest
        ls_target_MRN = set()
        for mrn in ls_scans['MRN'].tolist():
            ls_target_MRN.add(mrn)
    for fdr in ls_fdr:
        if args.select_mrn:
            mrn = int(fdr[:7])
            if mrn not in ls_target_MRN:
                continue
            if args.select_date:
                target_dates = ls_scans['mri_date'][np.where(ls_scans['MRN']==mrn)[0]].tolist() # date format: mm/dd/yyyy
        ls_seg = [x for x in os.listdir(root + 'NIFTI/' + fdr) if 'seg' in x]
        for f_nii in ls_seg:
            if args.select_date: # if reading csv, check if the scan's date is in the list; otherwise, process all scans.
                scan_date = check_date(f_nii)
                scan_date = str(int(scan_date[4:6])) + '/' + str(int(scan_date[6:8])) + '/' + str(int(scan_date[:4]))
                if scan_date not in target_dates:
                    continue
            path_nii = os.path.join(root + 'NIFTI/', fdr, f_nii)
            nii = nibabel.load(path_nii)
            if len(nii.header.get_zooms()) == 4:
                sx, sy, sz = nii.header.get_zooms()[:3]
            else:
                sx, sy, sz = nii.header.get_zooms()
            voxel = sx * sy * sz
            seg_maps = np.round(nii.get_fdata())
            # number of pixels of each class
            num_vent = np.sum(seg_maps == 3)
            num_eaf = np.sum(seg_maps == 1)
            num_csf = num_vent + num_eaf
            num_brain = np.sum(seg_maps == 2)
            # approximate volume
            vol_csf = num_csf * voxel/1000
            vol_vent = num_vent * voxel/1000
            vol_eaf = num_eaf * voxel/1000
            vol_brain = num_brain * voxel/1000
            # mri date
            mri_date = check_date(f_nii)
            mri_date = str(int(mri_date[4:6])) + '/' + str(int(mri_date[6:8])) + '/' + str(int(mri_date[:4]))
            mrn = int(fdr[:7])
            table_vol['MRN'].append(mrn)
            table_vol['filename'].append(f_nii)
            table_vol['mri_date'].append(mri_date)
            table_vol['CSF'].append(vol_csf)
            table_vol['ventricle'].append(vol_vent)
            table_vol['EAF'].append(vol_eaf)
            table_vol['brain'].append(vol_brain)
            # other info
            # dob
            dob = info[args.col_dob][np.where(info[args.col_mrn] == mrn)[0][0].item()]
            dob_y, dob_m, dob_d = str(dob)[:10].split('-')
            # if len(dob_y) == 2:
            #     dob_y = '20' + dob_y
            table_vol['dob_corr'].append(dob_m +'/' + dob_d + '/'+dob_y)
            # age
            if 'nan' not in mri_date:
                date_scan_m, date_scan_d, date_scan_y = mri_date.split('/')
                date_scan_days = date(int(date_scan_y), int(date_scan_m), int(date_scan_d))
                dob_days = date(int(dob_y), int(dob_m), int(dob_d))
                age_days = 0 if (date_scan_days - dob_days).days <0 else (date_scan_days - dob_days).days
                table_vol['age_corr'].append(age_days)
            else:
                table_vol['age_corr'].append('nan')
            # 
            date_surgery = info[args.col_date_sx][np.where(info[args.col_mrn] == int(mrn))[0][0].item()]
            if str(date_surgery) != 'NaT':
                date_surgery_y, date_surgery_m, date_surgery_d = str(date_surgery)[:10].split('-')
                if len(date_surgery_y) == 2:
                    date_surgery_y = '20' + date_surgery_y
                table_vol['date_surgery'].append(date_surgery_m + '/' + date_surgery_d + '/' + date_surgery_y)
                date_surgery_days = date(int(date_surgery_y), int(date_surgery_m), int(date_surgery_d))
                table_vol['days_postop'].append(date_scan_days - date_surgery_days)
            else:
                table_vol['date_surgery'].append('NaT')
                table_vol['days_postop'].append('NaT')
            # gender
            gender = info[args.col_gender][np.where(info[args.col_mrn] == int(mrn))[0][0].item()]
            if gender == 1:
                table_vol['gender'].append('Male')
            else:
                table_vol['gender'].append('Female')
            # zscore
            
    # Load the data and convert from pandas to xarray
    df = pd.DataFrame(table_vol)
    #df = pd.read_csv("vol_baseline.csv", header=0, index_col=None)
    ds = df.to_xarray()

    # Interpolate the standard distrbutions with the patient data
    brain_rv = rv_sel(rv_interp(GrowthStandards["brain"], {"age": ds["age_corr"]}), {"sex": ds["gender"]})
    csf_rv = rv_sel(rv_interp(GrowthStandards["csf"], {"age": ds["age_corr"]}), {"sex": ds["gender"]})

    # Compute the zscores
    brain_zscore = calc_z_score(brain_rv, ds["brain"])
    csf_zscore = calc_z_score(csf_rv, ds["CSF"])

    # Add the zscores to the xarray Dataset
    df["brain_volume_zscore"] = brain_zscore.data
    df["csf_volume_zscore"] = csf_zscore.data
    
    df.to_excel(args.dst)

def calc_z_score(rv: xr_stats.XrRV, v: float | xr.DataArray, log: bool=False, apply_kwargs=None) -> xr.DataArray:
    from scipy.special import ndtri, ndtri_exp

    coords = rv.coords
    attrs = getattr(rv, "attrs", {})
    if log:
        da = xr.where(v >= rv.median(), -ndtri_exp(rv.logsf(v, apply_kwargs=apply_kwargs)), ndtri_exp(rv.logcdf(v, apply_kwargs=apply_kwargs)))
    else:
        da = xr.where(v >= rv.median(), -ndtri(rv.sf(v, apply_kwargs=apply_kwargs)), ndtri(rv.cdf(v, apply_kwargs=apply_kwargs)))
    return da.assign_attrs(attrs).assign_coords(coords.variables)


if __name__ == '__main__':
    main()
