import pandas as pd
import xarray as xr
import xarray_einstats.stats as xr_stats

from growthstandards import rv_sel, rv_interp, GrowthStandards

def calc_z_score(rv: xr_stats.XrRV, v: float | xr.DataArray, log: bool=False, apply_kwargs=None) -> xr.DataArray:
    from scipy.special import ndtri, ndtri_exp

    coords = rv.coords
    attrs = getattr(rv, "attrs", {})
    if log:
        da = xr.where(v >= rv.median(), -ndtri_exp(rv.logsf(v, apply_kwargs=apply_kwargs)), ndtri_exp(rv.logcdf(v, apply_kwargs=apply_kwargs)))
    else:
        da = xr.where(v >= rv.median(), -ndtri(rv.sf(v, apply_kwargs=apply_kwargs)), ndtri(rv.cdf(v, apply_kwargs=apply_kwargs)))
    return da.assign_attrs(attrs).assign_coords(coords.variables)


# Load the data and convert from pandas to xarray
df = pd.read_csv("vol_baseline.csv", header=0, index_col=None)
ds = df.to_xarray()

# Interpolate the standard distrbutions with the patient data
brain_rv = rv_sel(rv_interp(GrowthStandards["brain"], {"age": ds["age"]}), {"sex": ds["gender"]})
csf_rv = rv_sel(rv_interp(GrowthStandards["csf"], {"age": ds["age"]}), {"sex": ds["gender"]})

# Compute the zscores
brain_zscore = calc_z_score(brain_rv, ds["brain"])
csf_zscore = calc_z_score(csf_rv, ds["CSF"])

# Add the zscores to the xarray Dataset
df["brain_volume_zscore"] = brain_zscore.data
df["csf_volume_zscore"] = csf_zscore.data

# Convert to pandas to save to a csv
df.to_csv("BV_CSV_volume_age_baseline_WarfCohort_zscore.csv", index=False)