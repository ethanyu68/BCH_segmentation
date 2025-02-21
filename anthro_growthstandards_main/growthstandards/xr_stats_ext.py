from functools import cached_property, partial, wraps
from itertools import chain
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Union, overload

import numpy as np
import scipy.optimize as optimize
import scipy.special as sc
import scipy.stats as stats
import xarray as xr
import xarray.core.types as xr_types
import xarray.core.utils as xr_utils
import xarray_einstats.stats as xr_stats

try:
    if not hasattr(xr_stats.XrRV, "median"):
        # FIXME: upstream
        xr_stats._add_documented_method(xr_stats.XrRV, "rv_generic", ["median"], xr_stats.doc_extras)
except:
    pass


def rv_to_ds(rv: xr_stats.XrRV):
    return xr.Dataset(
        {**{da.name: da for da in rv.args}, **rv.kwargs},
        attrs=getattr(rv, "attrs", None),
    )


def _ds_to_rv_args(distr: Union[stats.rv_discrete, stats.rv_continuous], ds: xr.Dataset) -> list[xr.DataArray]:
    # TODO: use`_param_info` instead of `_shape_info`?
    args = [ds[s.name] for s in distr._shape_info()]
    if "loc" in ds:
        args.append(ds["loc"])
    if "scale" in ds:
        args.append(ds["scale"])
    return args


@overload
def ds_to_rv(distr: stats.rv_discrete, ds: xr.Dataset) -> xr_stats.XrDiscreteRV:
    ...


@overload
def ds_to_rv(distr: stats.rv_continuous, ds: xr.Dataset) -> xr_stats.XrContinuousRV:
    ...


def ds_to_rv(distr: Union[stats.rv_discrete, stats.rv_continuous], ds: xr.Dataset):
    args = _ds_to_rv_args(distr, ds)
    if isinstance(distr, stats.rv_discrete):
        rv = xr_stats.XrDiscreteRV(distr, *args)
    elif isinstance(distr, stats.rv_continuous):
        rv = xr_stats.XrContinuousRV(distr, *args)
    else:
        raise TypeError(distr)
    rv.attrs = ds.attrs
    return rv


def rv_coords(rv: xr_stats.XrRV) -> xr.Dataset:
    return xr.merge(
        (da.coords.to_dataset() for da in chain(rv.args, rv.kwargs.values()) if isinstance(da, xr.DataArray)),
        compat="broadcast_equals",
        combine_attrs="drop_conflicts",
    )


if not hasattr(xr_stats.XrRV, "coords"):
    xr_stats.XrRV.coords = property(rv_coords, doc=xr.Dataset.coords.__doc__)


@overload
def map_rv_ds(rv: xr_stats.XrContinuousRV, map_ds: Callable[[xr.Dataset], xr.Dataset]) -> xr_stats.XrContinuousRV:
    ...


@overload
def map_rv_ds(rv: xr_stats.XrDiscreteRV, map_ds: Callable[[xr.Dataset], xr.Dataset]) -> xr_stats.XrDiscreteRV:
    ...


def map_rv_ds(rv: xr_stats.XrRV, map_ds: Callable[[xr.Dataset], xr.Dataset]):
    kls = type(rv)
    ds = rv_to_ds(rv)
    ds = map_ds(ds)
    args = _ds_to_rv_args(rv.dist, ds)
    mapped_rv = kls(rv.dist, *args)
    mapped_rv.attrs = ds.attrs
    return mapped_rv


def log_trapz(log_y, x=None, dx=1.0, axis=-1):
    """Log Integrate along the given axis using the composite trapezoidal rule over the log of the integrand."""
    log_y = np.asanyarray(log_y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * log_y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = log_y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    return sc.logsumexp(np.logaddexp(log_y[tuple(slice1)], log_y[tuple(slice2)]), b=d / 2.0, axis=axis)


def xr_log_integrate(self: xr.DataArray, *coords: Hashable):
    from xarray.core.variable import Variable

    result = self
    for coord in coords:
        if coord not in self.coords and coord not in self.dims:
            raise ValueError(f"Coordinate {coord} does not exist.")
        coord_var: Variable = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError(
                "Coordinate {} must be 1 dimensional but is {}" " dimensional".format(coord, coord_var.ndim)
            )
        dim = coord_var.dims[0]
        # use partial instead of reduce(**kwargs) to prevent future collisions
        result = result.reduce(partial(log_trapz, x=coord_var.data), dim=dim)
    return result


def _vectorized_root_scalar(f, x0, bracket, fprime):
    x_min, x_max = bracket
    # return optimize._chandrupatla(f, x_min, x_max)
    # need scipy 1.12 for _chandrupatla
    # see https://github.com/scipy/scipy/issues/7242 for stabilization progress
    return np.clip(optimize.newton(f, x0, fprime=fprime), x_min, x_max)


def _compound_ppf_root(
    q,
    x_0,
    x_min,
    x_max,
    m,
    p_m,
    *shape_params,
    dist: stats.rv_continuous,
    shape_keys: tuple[str, ...],
):
    kwds = dict(zip(shape_keys, shape_params))

    def fun(x):
        return np.trapz(p_m * dist.cdf(x[..., None], **kwds), m) - q

    def dfun(x):
        return np.trapz(p_m * dist.pdf(x[..., None], **kwds), m)

    return _vectorized_root_scalar(fun, x0=x_0, bracket=(x_min, x_max), fprime=dfun)


def _compound_isf_root(
    q,
    x_0,
    x_min,
    x_max,
    m,
    p_m,
    *shape_params,
    dist: stats.rv_continuous,
    shape_keys: tuple[str, ...],
):
    kwds = dict(zip(shape_keys, shape_params))

    def fun(x):
        return q - np.trapz(p_m * dist.sf(x[..., None], **kwds), m)

    def dfun(x):
        return np.trapz(p_m * dist.pdf(x[..., None], **kwds), m)

    return _vectorized_root_scalar(fun, x0=x_0, bracket=(x_min, x_max), fprime=dfun)


class XrCompoundRV:
    # TODO: support multiple marginal rvs, change arg to `marginals: Mappable[Hashable, XrContinuousRV]` and/or
    #       `**marginal_kwargs: XrContinuousRV`
    def __init__(
        self,
        cond_rv: xr_stats.XrContinuousRV,
        marginal_rv: xr_stats.XrContinuousRV,
        marginal_coord: Hashable,
    ):
        self.coord = marginal_coord
        self.marginal = rv_coords(cond_rv)[marginal_coord]
        self.cond_rv = cond_rv
        self.marginal_rv = marginal_rv
        self.attrs = getattr(cond_rv, "attrs", {}).copy()

    @cached_property
    def p_marginal(self) -> xr.DataArray:
        return self.marginal_rv.pdf(self.marginal)

    @cached_property
    def logp_marginal(self) -> xr.DataArray:
        return self.marginal_rv.logpdf(self.marginal)

    @property
    def coords(self) -> xr.Dataset:
        return xr.merge(
            (rv_coords(self.cond_rv), rv_coords(self.marginal_rv)),
            compat="broadcast_equals",
            combine_attrs="drop_conflicts",
        ).drop_vars(self.coord)

    def pdf(self, x: Union[float, xr.DataArray], apply_kwargs=None):
        p_x: xr.DataArray = self.cond_rv.pdf(x, apply_kwargs=apply_kwargs)
        return (p_x * self.p_marginal).integrate(self.coord)

    def cdf(self, x: Union[float, xr.DataArray], apply_kwargs=None):
        cdf_x: xr.DataArray = self.cond_rv.cdf(x, apply_kwargs=apply_kwargs)
        return (cdf_x * self.p_marginal).integrate(self.coord)

    def sf(self, x: Union[float, xr.DataArray], apply_kwargs=None):
        sf_x: xr.DataArray = self.cond_rv.sf(x, apply_kwargs=apply_kwargs)
        return (sf_x * self.p_marginal).integrate(self.coord)

    def ppf(self, q: Union[float, xr.DataArray], apply_kwargs=None):
        ppf_q = self.cond_rv.ppf(q, apply_kwargs=apply_kwargs)
        x_0: xr.DataArray = ppf_q.weighted(self.p_marginal).mean(self.coord)
        shape_ds = rv_to_ds(self.cond_rv)
        shape_params = list(shape_ds.values())
        return xr.apply_ufunc(
            _compound_ppf_root,
            q,
            x_0,
            ppf_q.min(self.coord),
            ppf_q.max(self.coord),
            self.marginal,
            self.p_marginal,
            *shape_params,
            input_core_dims=[[], [], [], [], [self.coord], [self.coord]] + [[self.coord]] * len(shape_params),
            kwargs=dict(dist=self.cond_rv.dist, shape_keys=shape_ds.keys()),
        )

    def isf(self, q: Union[float, xr.DataArray], apply_kwargs=None):
        isf_q = self.cond_rv.isf(q, apply_kwargs=apply_kwargs)
        x_0: xr.DataArray = isf_q.weighted(self.p_marginal).mean(self.coord)
        shape_ds = rv_to_ds(self.cond_rv)
        shape_params = list(shape_ds.values())
        return xr.apply_ufunc(
            _compound_isf_root,
            q,
            x_0,
            isf_q.min(self.coord),
            isf_q.max(self.coord),
            self.marginal,
            self.p_marginal,
            *shape_params,
            input_core_dims=[[], [], [], [], [self.coord], [self.coord]] + [[self.coord]] * len(shape_params),
            kwargs=dict(dist=self.cond_rv.dist, shape_keys=shape_ds.keys()),
        )

    def median(self, apply_kwargs=None):
        return self.ppf(0.5, apply_kwargs=apply_kwargs)

    def logpdf(self, x: Union[float, xr.DataArray], apply_kwargs=None):
        logp_x: xr.DataArray = self.cond_rv.logpdf(x, apply_kwargs=apply_kwargs)
        return xr_log_integrate(logp_x + self.logp_marginal, self.coord)

    def logcdf(self, x: Union[float, xr.DataArray], apply_kwargs=None):
        logcdf_x: xr.DataArray = self.cond_rv.logcdf(x, apply_kwargs=apply_kwargs)
        return xr_log_integrate(logcdf_x + self.logp_marginal, self.coord)

    def logsf(self, x: Union[float, xr.DataArray], apply_kwargs=None):
        logsf_x: xr.DataArray = self.cond_rv.logsf(x, apply_kwargs=apply_kwargs)
        return xr_log_integrate(logsf_x + self.logp_marginal, self.coord)


@wraps(xr.Dataset.isel)
def rv_isel(
    rv: Union[xr_stats.XrDiscreteRV, xr_stats.XrContinuousRV, XrCompoundRV],
    /,
    indexers: Optional[Mapping[Any, Any]] = None,
    drop: bool = False,
    missing_dims="raise",
    **indexers_kwargs: Any,
):
    indexers = xr_utils.either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
    kwargs = dict(drop=drop, missing_dims=missing_dims)
    if isinstance(rv, XrCompoundRV):
        if rv.coord in indexers:
            raise ValueError("Marginal coordinate can't be modified")
        cond_ds = rv_to_ds(rv.cond_rv)
        marginal_ds = rv_to_ds(rv.marginal_rv)
        dims = set(cond_ds.dims) | set(marginal_ds.dims)
        indexers = xr_utils.drop_dims_from_indexers(indexers, dims, missing_dims)
        cond_indexers = xr_utils.drop_dims_from_indexers(indexers, cond_ds.dims, "ignore")
        marginal_indexers = xr_utils.drop_dims_from_indexers(indexers, marginal_ds.dims, "ignore")
        cond_rv = map_rv_ds(rv.cond_rv, lambda ds: ds.isel(cond_indexers, **kwargs))
        marginal_rv = map_rv_ds(rv.marginal_rv, lambda ds: ds.isel(marginal_indexers, **kwargs))
        return XrCompoundRV(cond_rv, marginal_rv, rv.coord)
    else:
        return map_rv_ds(rv, lambda ds: ds.isel(indexers, **kwargs))


@wraps(xr.Dataset.sel)
def rv_sel(
    rv: Union[xr_stats.XrDiscreteRV, xr_stats.XrContinuousRV, XrCompoundRV],
    /,
    indexers: Optional[Mapping[Any, Any]] = None,
    method: xr_types.ReindexMethodOptions = None,
    tolerance: Optional[Union[int, float, Iterable[Union[int, float]]]] = None,
    drop: bool = False,
    **indexers_kwargs: Any,
):
    indexers = xr_utils.either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
    kwargs = dict(method=method, tolerance=tolerance, drop=drop)
    if isinstance(rv, XrCompoundRV):
        if rv.coord in indexers:
            raise ValueError("Marginal coordinate can't be modified")
        cond_ds = rv_to_ds(rv.cond_rv)
        marginal_ds = rv_to_ds(rv.marginal_rv)
        dims = set(cond_ds.dims) | set(marginal_ds.dims)
        indexers = xr_utils.drop_dims_from_indexers(indexers, dims, "raise")
        cond_indexers = xr_utils.drop_dims_from_indexers(indexers, cond_ds.dims, "ignore")
        marginal_indexers = xr_utils.drop_dims_from_indexers(indexers, marginal_ds.dims, "ignore")
        cond_rv = map_rv_ds(rv.cond_rv, lambda ds: ds.sel(cond_indexers, **kwargs))
        marginal_rv = map_rv_ds(rv.marginal_rv, lambda ds: ds.sel(marginal_indexers, **kwargs))
        return XrCompoundRV(cond_rv, marginal_rv, rv.coord)
    else:
        return map_rv_ds(rv, lambda ds: ds.sel(indexers, **kwargs))


@wraps(xr.Dataset.interp)
def rv_interp(
    rv: Union[xr_stats.XrDiscreteRV, xr_stats.XrContinuousRV, XrCompoundRV],
    /,
    coords: Optional[Mapping[Any, Any]] = None,
    method: xr_types.InterpOptions = "linear",
    assume_sorted: bool = False,
    kwargs: Optional[Mapping[str, Any]] = None,
    method_non_numeric: xr_types.ReindexMethodOptions = "nearest",
    **coords_kwargs: Any,
):
    coords = xr_utils.either_dict_or_kwargs(coords, coords_kwargs, "interp")
    interp_kwargs = dict(
        method=method,
        assume_sorted=assume_sorted,
        kwargs=kwargs,
        method_non_numeric=method_non_numeric,
    )
    if isinstance(rv, XrCompoundRV):
        if rv.coord in coords:
            raise ValueError("Marginal coordinate can't be modified")
        cond_ds = rv_to_ds(rv.cond_rv)
        marginal_ds = rv_to_ds(rv.marginal_rv)
        all_coords = set(cond_ds.coords) | set(marginal_ds.coords)

        invalid = set(coords.keys()) - all_coords
        if invalid:
            raise ValueError(f"Coordinates {invalid} do not exist. Expected one or more of {all_coords}")

        cond_coords = {k: v for k, v in coords.items() if k in cond_ds.coords}
        marginal_coords = {k: v for k, v in coords.items() if k in marginal_ds.coords}
        cond_rv = map_rv_ds(rv.cond_rv, lambda ds: ds.interp(cond_coords, **interp_kwargs))
        marginal_rv = map_rv_ds(rv.marginal_rv, lambda ds: ds.interp(marginal_coords, **interp_kwargs))
        return XrCompoundRV(cond_rv, marginal_rv, rv.coord)
    else:
        return map_rv_ds(rv, lambda ds: ds.interp(coords, **interp_kwargs))
