import numpy as np
from scipy.special import gamma
from mpmath import gammainc, mittag_leffler, exp as mp_exp

###############################################################################
# Gamma kernel kGamma
###############################################################################

def kGamma(params):
    al = params["al"]
    lam = params["lam"]
    nu = params["nu"]

    def f(tau):
        return (nu / gamma(al)) * tau**(al - 1) * np.exp(-lam * tau)

    return np.vectorize(f)


###############################################################################
# K00(tau) = ∫₀ᵗ kGamma(s)² ds
###############################################################################

def K00(params):
    al = params["al"]
    lam = params["lam"]
    nu = params["nu"]

    H2 = 2 * al - 1
    prefactor = (nu / gamma(al)) ** 2

    def f(tau):
        if tau <= 0:
            return 0.0
        if lam > 0:
            # gamma(H2) - Gamma(H2, 2 λ τ)
            diff_gamma = gamma(H2) - float(gammainc(H2, 0, 2 * lam * tau))
            return prefactor * diff_gamma / (2 * lam) ** H2
        else:
            return prefactor * tau**H2 / H2

    return np.vectorize(f)


###############################################################################
# Kjj(j, tau) = ∫₀ᵗ kGamma(s + j τ)² ds
###############################################################################

def Kjj(params):
    kG = kGamma(params)

    def f(j, tau):
        from scipy.integrate import quad
        if tau <= 0:
            return 0.0
        integrand = lambda s: kG(s + j * tau) ** 2
        val, _ = quad(integrand, 0, tau)
        return val

    return np.vectorize(f, excluded=None)


def K11(params):
    return lambda tau: Kjj(params)(1, tau)


###############################################################################
# K0(tau) = ∫₀ᵗ kGamma(s) ds
###############################################################################

def K0(params):
    al = params["al"]
    lam = params["lam"]
    nu = params["nu"]

    def f(tau):
        if tau <= 0:
            return 0.0

        if lam > 0:
            prefactor = (nu / gamma(al)) / lam**al
            bkt = gamma(al) - float(gammainc(al, 0, lam * tau))
            return prefactor * bkt
        else:
            return (nu / gamma(al + 1)) * tau**al

    return np.vectorize(f)


###############################################################################
# K01(t) = ∫₀ᵗ kGamma(s) kGamma(s + t) ds
###############################################################################

def K01(params):
    kG = kGamma(params)

    def f(t):
        from scipy.integrate import quad
        if t <= 0:
            return 0.0
        integrand = lambda s: kG(s) * kG(s + t)
        val, _ = quad(integrand, 0, t)
        return val

    return f


###############################################################################
# Kj(j)(dt) = ∫₀ᵗ kGamma(s + j dt) ds
###############################################################################

def Kj(params):
    al = params["al"]
    lam = params["lam"]
    nu = params["nu"]

    def g(j):
        def f(dt):
            if dt <= 0:
                return 0.0

            if lam > 0:
                prefactor = nu / (lam**al * gamma(al))
                bkt = float(gammainc(al, j * lam * dt, (j + 1) * lam * dt))
                return prefactor * bkt
            else:
                return (nu / gamma(al + 1)) * dt**al * ((j + 1)**al - j**al)

        return np.vectorize(f)

    return g


def K1(params):
    return Kj(params)(1)


###############################################################################
# bigK(tau) = resolvent of kGamma²
###############################################################################

def bigK(params):
    al = params["al"]
    lam = params["lam"]
    nu = params["nu"]

    H2 = 2 * al - 1
    nuHat2 = nu**2 * gamma(H2) / gamma(al)**2

    def f(tau):
        if tau <= 0:
            return 0.0

        # mlf(z, a, b) = Mittag-Leffler E_{a,b}(z)
        z = nuHat2 * tau**H2
        return nuHat2 * np.exp(-2 * lam * tau) * tau**(H2 - 1) * float(mittag_leffler(z, H2, H2))

    return f


###############################################################################
# bigK0(tau) = ∫₀ᵗ bigK(s) ds
###############################################################################

def bigK0(params):
    bigKp = bigK(params)

    def f(tau):
        if tau <= 0:
            return 0.0
        from scipy.integrate import quad
        val, _ = quad(bigKp, 0, tau)
        return val

    return np.vectorize(f)
