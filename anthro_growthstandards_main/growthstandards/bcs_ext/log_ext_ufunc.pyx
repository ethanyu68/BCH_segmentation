# cython: language_level=3, infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import cython.cimports.libc.math as np
# import cython.cimports.numpy.math as np


@cython.ufunc
@cython.cfunc
@cython.nogil
def log1mexp(x: cython.floating) -> cython.floating:
    """log(1 - exp(-x))

    References
    ----------
    [Maechler, Martin (2012). Accurately Computing log(1-exp(-|a|)).](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
    """
    # LN_2 = np.log(2)
    LN_2: cython.double = 0.693147180559945309417232121458176568

    if x > LN_2:
        return np.log1p(-np.exp(-x))
    else:
        return np.log(-np.expm1(-x))


@cython.ufunc
@cython.cfunc
@cython.nogil
def log1pexp(x: cython.floating) -> cython.floating:
    """log(1 + exp(x))

    References
    ----------
    [Maechler, Martin (2012). Accurately Computing log(1-exp(-|a|)).](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
    """
    if x <= -37:
        return np.exp(x)
    elif x <= 18:
        return np.log1p(np.exp(x))
    elif x <= 33.3:
        return x + np.exp(-x)
    else:
        return x


@cython.ufunc
@cython.cfunc
@cython.nogil
def logsubexp(x1: cython.floating, x2: cython.floating) -> cython.floating:
    """log(exp(x1) - exp(x2))

    References
    ----------
    [Maechler, Martin (2012). Accurately Computing log(1-exp(-|a|)).](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
    """
    # return x1 + log1mexp(x1 - x2)
    # LN_2 = np.log(2)
    LN_2: cython.double = 0.693147180559945309417232121458176568

    x = x1 - x2
    if x > LN_2:
        return x1 + np.log1p(-np.exp(-x))
    else:
        return x1 + np.log(-np.expm1(-x))
