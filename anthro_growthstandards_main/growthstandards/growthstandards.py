import importlib.resources
from collections.abc import Mapping
from functools import lru_cache
from typing import Iterator

import numpy as np
import scipy.stats
import xarray as xr
import xarray_einstats.stats as xr_stats
import zarr

from .bcs_ext.scipy_ext import BCCG, BCPE
from .xr_stats_ext import ds_to_rv, rv_to_ds

ANTHRO_DAYS_OF_MONTH = 30.4375

GROWTHSTANDARD_KEYS = (
    "arm_c",
    "bmi_height",
    "bmi_length",
    "brain",
    "csf",
    "head_c",
    "height",
    "length",
    "ss",
    "ts",
    "weight",
    "wfh",
    "wfl",
)

GROWTHSTANDARD_NAMES = (*GROWTHSTANDARD_KEYS, "len_hei", "bmi", "gfl", "gfh")


@lru_cache
def load_growthstandard_ds(g: str) -> xr.Dataset:
    if g in GROWTHSTANDARD_KEYS:
        traversable = importlib.resources.files(__package__)
        store = zarr.DirectoryStore(traversable.joinpath("growthstandards.zarr"))
        return xr.open_zarr(store=store, group=g, decode_times=False).load()
    elif g == "len_hei":
        return xr.concat(
            [load_growthstandard_ds("length"), load_growthstandard_ds("height")],
            dim="age",
            combine_attrs="drop_conflicts",
        ).assign_attrs(long_name="Recumbent Length or Standing Height")
    elif g == "bmi":
        return xr.concat(
            [
                load_growthstandard_ds("bmi_length"),
                load_growthstandard_ds("bmi_height"),
            ],
            dim="age",
            combine_attrs="drop_conflicts",
        ).assign_attrs(long_name="Body Mass Index")
    elif g == "gfl":
        # Do not mutate
        gfl = load_growthstandard_ds("wfl").copy()
        gfl["m"] = gfl["m"] / gfl.coords["length"]
        return gfl.assign_attrs(long_name="Growth Metric for Recumbent Length", units="kg/cm")
    elif g == "gfh":
        # Do not mutate
        gfh = load_growthstandard_ds("wfh").copy()
        gfh["m"] = gfh["m"] / gfh.coords["height"]
        return gfh.assign_attrs(long_name="Growth Metric for Standing Height", units="kg/cm")
    else:
        raise KeyError(g)


def try_simplify_bcs_rv(rv: xr_stats.XrContinuousRV):
    """For Y ∼ BCS(μ, σ, 𝜈; r), if 𝜈 = 1 then Y has a truncated symmetric distribution with parameters μ and μσ and support (0, ∞)."""
    # TODO: Add `𝜈 = 0` simplification
    # `truncnorm` is about 20% to 30% faster than `BCCG`
    ds = rv_to_ds(rv)
    # Scipy doesn't have a truncgennorm, so for now only supporting BCCG
    if rv.dist is not BCCG or not np.allclose(ds["nu"], 1):
        return rv
    loc = ds["mu"]
    scale = ds["mu"] * ds["sigma"]
    a = -np.reciprocal(ds["sigma"])
    return xr_stats.XrContinuousRV(scipy.stats.truncnorm, loc=loc, scale=scale, a=a, b=np.inf)


class _GrowthStandards(Mapping):
    def __getitem__(self, key: str) -> xr_stats.XrContinuousRV:
        gds = load_growthstandard_ds(key)
        if key in ("brain", "csf"):
            # TODO: Move nan interpolation earlier? Maybe even before saving into the zarr?
            # Have to `interpolate_na` b/c the male & female models have coefficents at differing ages, and where they differ is filled with `NaN`.
            #   Using PCHIP 1-D monotonic cubic interpolation to preserve the shape, while being more accurate then linear
            return ds_to_rv(BCPE, gds.rename_vars({"tau": "beta"}).interpolate_na("age", method="pchip"))
        else:
            return ds_to_rv(BCCG, gds.rename_vars({"m": "mu", "s": "sigma", "l": "nu"}))

    def __iter__(self) -> Iterator:
        return iter(GROWTHSTANDARD_NAMES)

    def __len__(self) -> int:
        return len(GROWTHSTANDARD_NAMES)


GrowthStandards = _GrowthStandards()
