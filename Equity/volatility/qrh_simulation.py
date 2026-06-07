import numpy as np
from y_from_xi import yFromXi
from gamma_kernel import K00, K0, bigK0


def QRH_sim(params, xi, params0=None):
    """
    Returns a simulator function equivalent to the R closure QRH.sim.
    """

    c = params["c"]
    H = params["al"] - 0.5

    # External model functions (must be supplied by caller)
    ey = yFromXi(params0, xi)
    bigK0p = bigK0(params)
    K00p = K00(params)
    K0p = K0(params)

    def simulator(paths, steps, expiries, output="all", delvix=1/12, nvix=10):
        Z_eps = np.random.randn(steps, paths)
        Z_chi = np.random.randn(steps, paths)
        v0 = xi(0)
        ey0 = ey(0)

        def sim(expiry):
            dt = expiry / steps

            K0del = K0p(dt)
            K00del = K00p(dt)
            bigK0del = bigK0p(dt)

            tj = np.arange(1, steps + 1) * dt
            yj = ey(tj)

            K00j = np.concatenate(([0.0], K00p(tj)))
            bstar = np.sqrt(np.diff(K00j) / dt)

            chi = np.zeros((steps, paths))
            v = np.full(paths, v0)
            Y = np.full(paths, ey0)
            yhat = np.full(paths, yj[0])

            rho_uchi = K0del / np.sqrt(K00del * dt)
            beta_uchi = K0del / dt

            X = np.zeros(paths)
            w = np.zeros(paths)

            for j in range(steps):
                alp = 1 / (2 * H + 1)

                varu = bigK0del * (alp * yhat**2 + (1 - alp) * Y**2 + c)
                vbar = varu / K00del

                sig_chi = np.sqrt(vbar * dt)
                sig_eps = np.sqrt(vbar * K00del * (1 - rho_uchi**2))

                chi_j = sig_chi * Z_chi[j]
                eps_j = sig_eps * Z_eps[j]
                chi[j] = chi_j

                u = beta_uchi * chi_j + eps_j
                Y_new = yhat + u
                vf = Y_new**2 + c

                dw = (v + vf) * 0.5 * dt
                w += dw
                X -= dw * 0.5 + chi_j

                if j < steps - 1:
                    btilde = bstar[1:j+2][::-1]
                    yhat = yj[j+1] + chi[:j+1].T @ btilde

                v = vf
                Y = Y_new

            vix = None
            if output in ("vix", "all"):
                vix2 = 0.0
                ds = delvix / nvix

                for k in range(1, nvix + 1):
                    tk = expiry + k * ds
                    Ku = np.concatenate(([K00p(tk)], K00p(tk - tj)))
                    ck = np.sqrt(-np.diff(Ku) / dt)

                    dyTu = chi.T @ ck
                    yTu = ey(tk) + dyTu

                    weight = 1 + bigK0p((nvix - k) * ds)
                    vix2 += (yTu**2 + c) * weight / nvix

                vix2 += v * (1 + bigK0p(delvix)) / (2 * nvix)
                vix2 -= (yTu**2 + c) / (2 * nvix)
                vix = np.sqrt(vix2)

            if output == "v":
                return v
            if output == "X":
                return X
            if output == "w":
                return w
            if output == "vix":
                return vix

            return dict(v=v, X=X, w=w, vix=vix)

        if output != "all":
            return np.vstack([sim(e) for e in expiries])
        else:
            return [sim(e) for e in expiries]

    return simulator
