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
'''
the volume table should contain columns: age_corr, gender, brain, CSF
age_corr is number of days
gender is 'Male' or 'Female'
brain, CSF cm^3
'''
parser = argparse.ArgumentParser(
        description="Process some integers.")
parser.add_argument("--vol_table",default='/home/ethanyu/data/seg_results/cchu_volumes.xlsx', help='the folder that contains the NIFTI segmentation files')

args = parser.parse_args()
def main():
    vol_table = pd.read_excel(args.vol_table)
    # Load the data and convert from pandas to xarray
    df = pd.DataFrame(vol_table)
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
    
    df.to_excel(args.vol_table)

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
