from functools import reduce

import numpy as np
import scipy.special as sc
import scipy.stats as stats
from scipy.stats._distn_infrastructure import _ShapeInfo

from .log_ext_ufunc import log1mexp, logsubexp

# LN_2 = np.log(2)
LN_2 = 0.693147180559945309417232121458176568


def bcs_logpdf(distr, y, mu, sigma, nu, *shape_params, **shape_kwds):
    z = sc.boxcox(y / mu, nu) / sigma
    return (
        (nu - 1) * np.log(y)
        - nu * np.log(mu)
        - np.log(sigma)
        + distr.logpdf(z, *shape_params, **shape_kwds)
        - distr.logcdf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
    )


def bcs_cdf(distr, y, mu, sigma, nu, *shape_params, **shape_kwds):
    z = sc.boxcox(y / mu, nu) / sigma
    z_cdf = distr.cdf(z, *shape_params, **shape_kwds)
    return np.where(
        nu > 0,
        z_cdf - distr.cdf(-1 / (sigma * abs(nu)), *shape_params, **shape_kwds),
        z_cdf,
    ) / distr.cdf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)


def bcs_sf(distr, y, mu, sigma, nu, *shape_params, **shape_kwds):
    z = sc.boxcox(y / mu, nu) / sigma
    z_sf = distr.sf(z, *shape_params, **shape_kwds)
    ub_sf = distr.sf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
    ub_cdf = distr.cdf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
    tz_sf = z_sf - ub_sf
    return (
        np.where(
            nu > 0,
            z_sf,
            tz_sf,
        )
        / ub_cdf
    )


def bcs_ppf(distr, p, mu, sigma, nu, *shape_params, **shape_kwds):
    ub_cdf = distr.cdf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
    z = np.where(
        nu <= 0,
        distr.ppf(p * ub_cdf, *shape_params, **shape_kwds),
        distr.ppf(1 - (1 - p) * ub_cdf, *shape_params, **shape_kwds),
    )
    return sc.inv_boxcox(z * sigma, nu) * mu


# def bcs_isf(distr, p, mu, sigma, nu, *shape_params, **shape_kwds):
#     ub_cdf = distr.cdf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
#     z = np.where(
#         nu <= 0,
#         distr.ppf((1 - p) * ub_cdf, *shape_params, **shape_kwds),
#         distr.ppf(1 - p * ub_cdf, *shape_params, **shape_kwds),
#     )
#     return inv_boxcox(z * sigma, nu) * mu


def bcs_logcdf(distr, y, mu, sigma, nu, *shape_params, **shape_kwds):
    z = sc.boxcox(y / mu, nu) / sigma
    z_logcdf = distr.logcdf(z, *shape_params, **shape_kwds)
    # lb_logcdf = distr.logcdf(-1 / (sigma * nu), *shape_params, **shape_kwds)
    ub_logcdf = distr.logcdf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
    return (
        np.where(
            nu > 0,
            np.logaddexp(z_logcdf, log1mexp(-ub_logcdf)),
            z_logcdf,
        )
        - ub_logcdf
    )


def bcs_logsf(distr, y, mu, sigma, nu, *shape_params, **shape_kwds):
    z = sc.boxcox(y / mu, nu) / sigma
    z_logsf = distr.logsf(z, *shape_params, **shape_kwds)
    ub_logsf = distr.logsf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
    ub_logcdf = distr.logcdf(1 / (sigma * abs(nu)), *shape_params, **shape_kwds)
    tz_logsf = logsubexp(z_logsf, ub_logsf)
    return (
        np.where(
            nu > 0,
            z_logsf,
            tz_logsf,
        )
        - ub_logcdf
    )


# Have to use approx, as doing the numerical integration isn't vectorized & takes an extremely long time
def approx_bcs_munp(distr, n, mu, sigma, nu, *shape_params, max_distr_moment=6, **shape_kwds):
    r"""$$\mathop{\mathbb{E}}\left[\left(\mu \left(1 + \sigma \nu X\right)^{1/\nu}\right)^n\right] = \mu^n \sum_{k=0}^{\infty} \binom{n / \nu}{k} \left(\nu  \sigma \right)^k \mathop{\mathbb{E}}\left[X^k\right]$$"""
    mom = 1
    alpha = n / nu
    c = nu * sigma
    # TODO: Stop assuming `distr` is symmetric, so as to work with truncated distributions
    # TODO: Take advantage of `k ∈ ℕ` and iteratively calc `binom(alpha, k)` and `c**k` via conversion to horner form
    for k in range(2, 1 + max_distr_moment, 2):
        mom += distr.moment(k, *shape_params, **shape_kwds) * c**k * sc.binom(alpha, k)
    return mom * mu**n


# Have to use approx, as doing the numerical integration isn't vectorized & takes an extremely long time
def approx_bcs_stats(distr, mu, sigma, nu, *shape_params, use_6th_moment=False, **shape_kwds):
    r"""This uses the same expansions from `approx_bcs_munp`, but does an additional expansion at $\sigma \to 0$ to linearize"""
    mean = mu * (1 + (1 - nu) * sigma**2 / 2)
    E_Z_4 = distr.moment(4, *shape_params, **shape_kwds)
    cv = sigma * (1 + (nu - 1) * sigma**2 * ((11 * nu - 7) * E_Z_4 - (3 * nu - 15)) / 24)
    skew = 3 * (1 - nu) * sigma * (E_Z_4 - 1) / 2
    # FIXME: fails for BCCG, but works for BCPE
    kurtosis = E_Z_4 - 3
    if use_6th_moment:
        # TODO: This works for both, but the un-patched BCPE has to do numerical int, so leaving optional for now
        E_Z_6 = distr.moment(6, *shape_params, **shape_kwds)
        kurtosis = (
            kurtosis
            - sigma**2
            * (nu - 1)
            * (E_Z_4**2 * (11 * nu - 7) + 15 * E_Z_4 * (nu - 1) + E_Z_6 * (13 - 17 * nu) - 9 * nu + 9)
            / 6
        )
    return mean, (mean * cv) ** 2, skew, kurtosis


class _BCS_gen(stats.rv_continuous):
    """
    Family of Box-Cox Symmetric Distributions

    References
    ----------
    [Box-Cox symmetric distributions and applications to nutritional data](https://arxiv.org/pdf/1604.02221.pdf)
    """

    _distr: stats.rv_continuous

    def __init__(self, *args, **kwargs):
        kwargs = {
            "a": 0,
            "shapes": " ".join([s.name for s in self._shape_info()]),
            **kwargs,
        }
        super().__init__(*args, **kwargs)

    def _shape_info(self):
        return [
            _ShapeInfo("mu", False, (0, np.inf), (False, False)),
            _ShapeInfo("sigma", False, (0, np.inf), (False, False)),
            _ShapeInfo("nu", False, (-np.inf, np.inf), (False, False)),
            *self._distr._shape_info(),
        ]

    def _scale_normalizer(self, *shape_params):
        return None

    def _get_shape_kwds(self, *shape_params):
        sn = self._scale_normalizer(*shape_params)
        return {} if sn is None else {"scale": sn}

    def _logpdf(self, x, mu, sigma, nu, *shape_params):
        return bcs_logpdf(
            self._distr,
            x,
            mu,
            sigma,
            nu,
            *shape_params,
            **self._get_shape_kwds(*shape_params),
        )

    def _pdf(self, x, mu, sigma, nu, *shape_params):
        return np.exp(self._logpdf(x, mu, sigma, nu, *shape_params))

    def _cdf(self, x, mu, sigma, nu, *shape_params):
        return bcs_cdf(
            self._distr,
            x,
            mu,
            sigma,
            nu,
            *shape_params,
            **self._get_shape_kwds(*shape_params),
        )

    def _sf(self, x, mu, sigma, nu, *shape_params):
        return bcs_sf(
            self._distr,
            x,
            mu,
            sigma,
            nu,
            *shape_params,
            **self._get_shape_kwds(*shape_params),
        )

    def _logcdf(self, x, mu, sigma, nu, *shape_params):
        return bcs_logcdf(
            self._distr,
            x,
            mu,
            sigma,
            nu,
            *shape_params,
            **self._get_shape_kwds(*shape_params),
        )

    def _logsf(self, x, mu, sigma, nu, *shape_params):
        return bcs_logsf(
            self._distr,
            x,
            mu,
            sigma,
            nu,
            *shape_params,
            **self._get_shape_kwds(*shape_params),
        )

    def _ppf(self, q, mu, sigma, nu, *shape_params):
        return bcs_ppf(
            self._distr,
            q,
            mu,
            sigma,
            nu,
            *shape_params,
            **self._get_shape_kwds(*shape_params),
        )

    # def _isf(self, q, mu, sigma, nu, *shape_params):
    #     return bcs_isf(self._distr, q, mu, sigma, nu, *shape_params, **self._get_shape_kwds(*shape_params))

    def _munp(self, n, mu, sigma, nu, *shape_params):
        return approx_bcs_munp(
            self._distr,
            n,
            mu,
            sigma,
            nu,
            *shape_params,
            max_distr_moment=6,
            **self._get_shape_kwds(*shape_params),
        )

    # def _stats(self, mu, sigma, nu, *shape_params):
    #     return approx_bcs_stats(self._distr, mu, sigma, nu, *shape_params, use_6th_moment=True, **self._get_shape_kwds(*shape_params))

    def _argcheck(self, mu, sigma, nu, *shape_params):
        return reduce(np.logical_and, [mu > 0, sigma > 0, np.isfinite(nu)]) & self._distr._argcheck(*shape_params)


class BCCG_gen(_BCS_gen):
    _distr = stats.norm


BCCG = BCCG_gen(name="BCCG")


class BCPE_gen(_BCS_gen):
    """
    Box–Cox Power Exponential distribution

    References
    ----------
    [Smooth centile curves for skew and kurtotic data modelled using the Box–Cox power exponential distribution](https://doi.org/10.1002/sim.1861)
    """

    _distr = stats.gennorm

    def _scale_normalizer(self, beta):
        return np.exp(0.5 * (sc.gammaln(1 / beta) - sc.gammaln(3 / beta)))


BCPE = BCPE_gen(name="BCPE")


# FIXME: upstream
if stats._continuous_distns.gennorm_gen._munp is stats.rv_continuous._munp:

    def _gennorm_munp(self, n, beta):
        if int(n) % 2 == 0:
            return np.exp(sc.gammaln((n + 1) / beta) - sc.gammaln(1 / beta))
        else:
            return 0.0

    stats._continuous_distns.gennorm_gen._munp = _gennorm_munp


# FIXME: verify & upstream
if stats._continuous_distns.gennorm_gen._logcdf is stats.rv_continuous._logcdf:

    def approx_gamma_logsf(x, s):
        return (s - 1) * np.log(x) - x - sc.gammaln(s)

    def _gennorm_logcdf(self, x, beta):
        s = abs(x) ** beta
        s_logsf = stats.gamma.logsf(s, 1 / beta)
        s_logsf = np.where(np.isfinite(s_logsf), s_logsf, approx_gamma_logsf(s, 1 / beta))
        return np.where(
            x >= 0,
            log1mexp(-s_logsf + LN_2),
            s_logsf - LN_2,
        )

    def _gennorm_logsf(self, x, beta):
        s = abs(x) ** beta
        s_logsf = stats.gamma.logsf(s, 1 / beta)
        s_logsf = np.where(np.isfinite(s_logsf), s_logsf, approx_gamma_logsf(s, 1 / beta))
        return np.where(
            x >= 0,
            s_logsf - LN_2,
            log1mexp(-s_logsf + LN_2),
        )

    stats._continuous_distns.gennorm_gen._logcdf = _gennorm_logcdf
    stats._continuous_distns.gennorm_gen._logsf = _gennorm_logsf
