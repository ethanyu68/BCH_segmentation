# import tensorflow.compat.v2 as tf
# from tensorflow_probability.python.bijectors import bijector
# from tensorflow_probability.python.internal import assert_util
# from tensorflow_probability.python.internal import parameter_properties
# from tensorflow_probability.python.internal import tensor_util

from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
from tensorflow_probability.substrates.jax.bijectors import (
    identity as identity_bijector,
)
from tensorflow_probability.substrates.jax.bijectors import (
    softplus as softplus_bijector,
)
from tensorflow_probability.substrates.jax.distributions import (
    GeneralizedNormal,
    MixtureSameFamily,
    Normal,
    distribution,
)
from tensorflow_probability.substrates.jax.internal import (
    dtype_util,
    parameter_properties,
)
from tensorflow_probability.substrates.jax.internal import prefer_static as ps
from tensorflow_probability.substrates.jax.internal import samplers, tensor_util


def boxcox(x, lmbda):
    return tf.where(
        lmbda == 0,
        tf.math.log(x),
        (tf.math.pow(x, lmbda) - 1) / lmbda,
        # tf.math.expm1(tf.math.log(x) * lmbda) / lmbda,
    )


def inv_boxcox(y, lmbda):
    return tf.where(
        lmbda == 0,
        tf.exp(y),
        tf.math.pow(1 + lmbda * y, tf.math.reciprocal(lmbda)),
        # tf.exp(tf.math.log1p(y * lmbda) / lmbda),
    )


def same_family_mixture_quantile(self: MixtureSameFamily, value, **find_root_kwargs):
    from tensorflow_probability.substrates.jax.math import find_root_chandrupatla

    _, components_distribution = self._get_distributions_with_broadcast_batch_shape()
    component_quantiles = components_distribution.quantile(value)  # [B, k, E]
    event_ndims = self._event_ndims()
    lb = component_quantiles.min(axis=-1 - event_ndims)  # [B, E]
    ub = component_quantiles.max(axis=-1 - event_ndims)  # [B, E]

    res = find_root_chandrupatla(lambda x: self.cdf(x) - value, lb, ub, **find_root_kwargs)  # [B]
    return res.estimated_root  # [B, E]


MAX_DISTR_MOMENT = 6
force_probs_to_zero_outside_support = True


class BoxCoxSymmetric(distribution.AutoCompositeTensorDistribution):
    def __init__(
        self,
        loc,
        scale,
        nu,
        std_distr: distribution.AutoCompositeTensorDistribution,
        validate_args=False,
        allow_nan_stats=True,
        parameters=None,
        name=None,
    ):
        parameters = dict(locals()) if parameters is None else parameters
        if name is None:
            name = std_distr.name or ""
            name = f"BoxCoxSymmetric{name}"
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([loc, scale, nu], dtype_hint=tf.float32)
            self._loc = tensor_util.convert_nonref_to_tensor(loc, dtype=dtype, name="loc")
            self._scale = tensor_util.convert_nonref_to_tensor(scale, dtype=dtype, name="scale")
            self._nu = tensor_util.convert_nonref_to_tensor(nu, dtype=dtype, name="nu")
            self._std_distr = std_distr
            super().__init__(
                dtype=dtype,
                reparameterization_type=self._std_distr.reparameterization_type,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name,
            )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            loc=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))
            ),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))
            ),
            nu=parameter_properties.ParameterProperties(),
            std_distr=parameter_properties.BatchedComponentProperties(),
        )

    @property
    def loc(self):
        """Distribution parameter for the median."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the relative dispersion."""
        return self._scale

    @property
    def nu(self):
        """Distribution parameter for the skew."""
        return self._nu

    @property
    def std_distr(self):
        """Standard base distribution, p(x)."""
        return self._std_distr

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _default_event_space_bijector(self):
        return softplus_bijector.Softplus(validate_args=self.validate_args)

    def _sample_n(self, n, seed=None):
        batch_shape = self._batch_shape_tensor()
        shape = ps.concat([[n], batch_shape], 0)
        probs = samplers.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=self.dtype, seed=seed)
        return self._quantile(probs)

    def _log_prob(self, y):
        z = boxcox(y / self.loc, self.nu) / self.scale
        lp = (
            (self.nu - 1) * tf.math.log(y)
            - self.nu * tf.math.log(self.loc)
            - tf.math.log(self.scale)
            + self.std_distr.log_prob(z)
            - self.std_distr.log_cdf(1 / (self.scale * abs(self.nu)))
        )
        if force_probs_to_zero_outside_support:
            return tf.where(y > 0, lp, -float("inf"))
        return lp

    def _cdf(self, y):
        z = boxcox(y / self.loc, self.nu) / self.scale
        z_cdf = self.std_distr.cdf(z)
        lb_cdf = self.std_distr.cdf(-1 / (self.scale * abs(self.nu)))
        ub_cdf = self.std_distr.cdf(1 / (self.scale * abs(self.nu)))
        return tf.where(self.nu > 0, z_cdf - lb_cdf, z_cdf) / ub_cdf

    def _survival_function(self, y):
        z = boxcox(y / self.loc, self.nu) / self.scale
        z_sf = self.std_distr.survival_function(z)
        ub_sf = self.std_distr.survival_function(1 / (self.scale * abs(self.nu)))
        ub_cdf = self.std_distr.cdf(1 / (self.scale * abs(self.nu)))
        tz_sf = z_sf - ub_sf
        return tf.where(self.nu > 0, z_sf, tz_sf) / ub_cdf

    def _quantile(self, p):
        ub_cdf = self.std_distr.cdf(1 / (self.scale * abs(self.nu)))
        z = tf.where(
            self.nu <= 0,
            self.std_distr.quantile(p * ub_cdf),
            self.std_distr.quantile(1 - (1 - p) * ub_cdf),
        )
        return inv_boxcox(z * self.scale, self.nu) * self.loc

    def _standard_moment(self, n):
        raise NotImplementedError

    def _munp(self, n):
        mom = 1
        alpha = n / self.nu
        c = self.nu * self.scale
        # TODO: Stop assuming `distr` is symmetric, so as to work with truncated distributions
        # TODO: Take advantage of `k ∈ ℕ` and iteratively calc `binom(alpha, k)` and `c**k` via conversion to horner form
        coef = c * alpha
        for k in range(2, 1 + MAX_DISTR_MOMENT):
            coef = coef * c * ((alpha + (1 - k)) / k)
            if k % 2 == 0:
                mom = mom + (self._standard_moment(k) * coef)
        return mom * tf.math.pow(self.loc, n)

    def _mean(self):
        return self._munp(1)

    def _variance(self):
        return self._munp(2) - tf.math.square(self._munp(1))


class BoxCoxColeGreen(BoxCoxSymmetric):
    def __init__(
        self,
        loc,
        scale,
        nu,
        validate_args=False,
        allow_nan_stats=True,
        name="BoxCoxColeGreen",
    ):
        parameters = dict(locals())
        dtype = dtype_util.common_dtype([loc, scale, nu], dtype_hint=tf.float32)
        return super().__init__(
            loc,
            scale,
            nu,
            Normal(tf.zeros([], dtype), tf.ones([], dtype)),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        parent = super()._parameter_properties(dtype, num_classes)
        subset = {k: p for k, p in parent.items() if k in ("loc", "scale", "nu")}
        return subset

    def _standard_moment(self, n):
        from scipy.special import factorial2

        if n % 2 == 0:
            return factorial2(n - 1)
        else:
            raise NotImplementedError


class BoxCoxPowerExponential(BoxCoxSymmetric):
    def __init__(
        self,
        loc,
        scale,
        nu,
        power,
        validate_args=False,
        allow_nan_stats=True,
        name="BoxCoxPowerExponential",
    ):
        parameters = dict(locals())
        dtype = dtype_util.common_dtype([loc, scale, nu, power], dtype_hint=tf.float32)
        std_scale = tf.exp((tf.math.lgamma(1 / power) - tf.math.lgamma(3 / power)) / 2)
        self._power = tensor_util.convert_nonref_to_tensor(power, dtype=dtype, name="power")

        return super().__init__(
            loc,
            scale,
            nu,
            GeneralizedNormal(tf.zeros([], dtype), tf.cast(std_scale, dtype), power),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        parent = super()._parameter_properties(dtype, num_classes)
        subset = {k: p for k, p in parent.items() if k in ("loc", "scale", "nu")}
        inner = GeneralizedNormal._parameter_properties(dtype, num_classes)
        return {**subset, "power": inner["power"]}

    @property
    def power(self):
        """Distribution parameter for the shape."""
        return self._power

    def _standard_moment(self, n):
        if n % 2 == 0:
            return (
                tf.math.pow(self.std_distr.scale, n)
                * tf.math.gamma((n + 1) / self.power)
                / tf.math.gamma(1 / self.power)
            )
        else:
            raise NotImplementedError
