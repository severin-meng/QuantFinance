r"""
merton jump-diffusion model to price european options

Merton Model: (left-limit, right-continuous)
Same as black-scholes with a jump term - per time step, there are dN jumps such that S -> S * V, where log(V) ~ N(m, v)

Under P:
dS = S * mu * dt  +  S * vol * dW  +  S * [exp( Normal(mean_jmp, var_jmp) ) - 1] * dN

dW is a brownian motion, e.g. sqrt(dt) * Normal(0, 1)
dN is a poissonian with intensity lambda * dt

Recall Ito, for a function f(t, S):
f(t, S_t) - f(0, S_0) = int_0^t d_u { f(u, s) + mu * S_ * d_s f(u, s) + 1/2 * vol**2 * S_**2 * d_ss f(u, s) du }
                        + vol * int_0^t S_ * d_s f(u, s) dW_u
                        + \Sum_{0 <= r <= t} [f(r, S_r) - f(r-, S_r-)] * 1_{N(r) - N(r-) = 1}

the first term is the Ito drift
the second term is the Ito diffusion
the third term is the cumulative jump contribution

Then, with f(t, S) = log(S), we have that f(S+) - f(S-) = log(S*V) - log(S) = N(m, v)

log(S_t / S_0 ) = (mu - 0.5 * vol**2) * t + vol * W_t + \Sum_{jumps} Normal(m, v)
S_t = S_0 * exp[ (mu - 0.5*vol**2)t + vol*N*sqrt(dt) + sum_n Normals(m,v)  ]


what is the drift of this process under the risk-neutral measure?
\E[S_t] = S_0 * exp[ (mu - 0.5*vol**2)t] + S_0 \E[exp(vol * N * sqrt(t))] + S_0 \E[exp(sum_n Normals(m, v))]

from Black-Scholes we know \E[exp(vol*N*sqrt(t))] = 0.5 * vol**2 * t:

What about \E[exp(sum_n Normals(m, v)]?
what about a single jump
\E [exp (Normals(m, v))] = int_R p(x) exp(x) dx; p(x) = 1 / sqrt(2 * pi * v) * exp(-(x-m)**2/(2*v))
-> \E [exp (Normals(m, v))] = 1 / sqrt(2 * pi * v) * int_R exp(- (x**2 - 2mx + m**2 - 2*v*x) / (2*v))
complete the square
x**2 - 2mx - 2*v*x + m**2 = (x - (m+v))**2  - v**2 - 2*m*v
exp(- (x**2 - 2mx + m**2 - 2*v*x) / (2*v)) = exp(-(x - (m+v))**2 / (2*v)) * exp((v**2+2mv) / (2 * v))
Integrated, we get exp(m + v/2)

So, we have \E[dS] = S * mu * dt + S * [exp(m + v/2) - 1] * lambda * dt
we want: \E[dS] = r * dt
we need: mu = r - lambda * {exp(m + v/2) -1}
So we have, compare to black-scholes risk-neutral drift of mu=r, an extra compensation term

(S -> V * S  <=> log(S) -> log(V) + log(S))
Under Q, we have:
dS/S = [r - lambda * (exp(m + v/2) - 1)] dt + vol dW * (V-1)dN
dlog(S) =  [r - lambda * (exp(m + v/2) - 1)] dt - vol**2/2 dt + vol dW  + log(V) dN

"""



import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use('ggplot')
import numpy as np
from black_scholes import bs_call, bs_vega, bs_call_iv
from math import factorial


def simulate_merthon_paths(spot, maturity, rate, vol, intensity, mean_jump, std_jump, steps, nbr_paths):
    # shape: [time, path]
    size = (steps, nbr_paths)
    dt = maturity / steps
    rng = np.random.default_rng()
    nbr_jumps = rng.poisson(lam=intensity * dt, size=size)
    jump_size = rng.normal(loc=0, scale=1, size=size)
    log_jumps = nbr_jumps * mean_jump + np.sqrt(nbr_jumps) * std_jump * jump_size  # Z = n*mu + sqrt(n) * std_dev * N
    log_diffusion_steps = (rate - vol**2/2 - intensity * (np.exp(mean_jump + std_jump**2 / 2) - 1)) * dt + vol * np.sqrt(dt) * rng.normal(size=size)

    log_increments = log_jumps + log_diffusion_steps
    log_paths = np.cumsum(log_increments, axis=0)
    paths = np.exp(log_paths) * spot
    return paths


def plot_merton_paths():
    spot = 100
    maturity = 1
    rate = 0.02
    vol = 0.2

    intensity = 0.05
    mean_jump = -0.15
    std_jump = 0.1

    steps = 300
    nbr_paths = 3

    merton_paths = simulate_merthon_paths(spot, maturity, rate, vol, intensity, mean_jump, std_jump, steps, nbr_paths)
    plt.figure()
    plt.plot(merton_paths)
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.title("Asset Paths")
    plt.show()

# bs_call(spot, strike, vol, rate, tte)
def merton_call_price(spot, maturity, rate, vol, intensity, mean_jump, std_jump, strike, summands=10, method='internet', iv_tol=1.e-5):
    intrinsic = max(spot - strike, 0)
    if method == 'internet':
        expected_jump = np.exp(mean_jump + std_jump**2 / 2)
        lam_m = intensity * expected_jump
        exp_lam_m_t = np.exp(-lam_m * maturity)

        summand = lambda n: exp_lam_m_t * (lam_m * maturity) ** n / factorial(n) * bs_call(
            spot,
            strike,
            np.sqrt(vol**2 + n * std_jump**2 / maturity),
            rate - intensity * (expected_jump - 1) + n * np.log(expected_jump) / maturity,
            maturity)
    elif method == 'savine':
        # for zero rates one can adjust the formula so the black-scholes function is also called with zero rate
        lam_t = intensity * maturity
        exp_lam_t = np.exp(-lam_t)

        compensation = intensity * (np.exp(mean_jump + std_jump ** 2 / 2) - 1)
        summand = lambda n: exp_lam_t * lam_t ** n / factorial(n) * bs_call(
            spot * np.exp(n * (mean_jump + std_jump ** 2 / 2) - compensation * maturity),
            strike,
            np.sqrt(vol**2 + n * std_jump**2 / maturity),
            0,
            maturity)

    else:
        raise NotImplementedError(f"Merton method {method} unknown.")

    estimated_iv = np.sqrt(vol**2 + intensity * (mean_jump**2 + std_jump**2))
    estimated_vega = bs_vega(spot, strike, estimated_iv, rate, maturity)
    safety_factor = 0.1
    abs_tol = max(1.e-13, estimated_vega * iv_tol * safety_factor)
    rel_tol = 1.e-12

    call_price = 0
    for n in range(summands):
        term = summand(n)
        call_price += term
        tolerance = max(abs_tol, rel_tol * call_price)
        # if call_price > intrinsic and term < tolerance:
        #     print(f"K={strike}, T={maturity}, N={n}")
        #     break
    return call_price


def merton_call_price_opt(spot, maturity, rate, vol, intensity, mean_jump, std_jump, strike, iv_tol=1.e-5):
    # compute merton call price using truncated series
    # start at poisson mode counter, move outwards
    intrinsic = max(spot - strike, 0)

    lam_t = intensity * maturity
    exp_lam_t = np.exp(-lam_t)
    m_v2 = mean_jump + std_jump ** 2 / 2
    compensation_mat = intensity * (np.exp(m_v2) - 1) * maturity

    def vol_n(n):
        return np.sqrt(vol**2 + n * std_jump**2 / maturity)

    def spot_n(n):
        return spot * np.exp(n * m_v2 - compensation_mat)

    def prob_n(n):
        return exp_lam_t * lam_t ** n / factorial(n)

    n_mode = int(lam_t)
    print(f"Starting at n_mode={n_mode}")
    prob = prob_n(n_mode)
    call_prc = bs_call(spot_n(n_mode), strike, vol_n(n_mode), rate, maturity)
    term = call_prc * prob

    partial_sum = 0.0
    cumulative_proba = 0.0

    n_left = n_mode - 1
    n_right = n_mode + 1
    if n_mode > 0:
        prob_left = prob * n_mode / lam_t
    else:
        prob_left = 0.0
    prob_right = prob * lam_t / n_right

    partial_sum += term
    cumulative_proba += prob

    if n_left >= 0:
        call_left = bs_call(spot_n(n_left), strike, vol_n(n_left), rate, maturity)
        term_left = prob_left * call_left
    else:
        term_left = -1
    call_right = bs_call(spot_n(n_right), strike, vol_n(n_right), rate, maturity)
    term_right = prob_right * call_right


    # error tolerance
    estimated_iv = np.sqrt(vol**2 + intensity * (mean_jump**2 + std_jump**2))
    estimated_vega = bs_vega(spot, strike, estimated_iv, rate, maturity)
    safety_factor = 0.1
    abs_tol = max(1.e-13, estimated_vega * iv_tol * safety_factor)
    rel_tol = 1.e-12

    print(f"Absolute tolerance is: {abs_tol}")

    def tol_term(partial):
        return max(abs_tol, rel_tol * partial)

    # probability tolerance
    eps_poisson = 1.e-12

    # max number iterations
    n_limit = 100

    for i in range(n_limit):
        remaining_proba = 1 - cumulative_proba
        next_term = max(term_left, term_right)

        if next_term <= tol_term(partial_sum) and remaining_proba <= eps_poisson:
            print(f"Stopping at after {i} iterations due to tolerance: next_term = {next_term}, remaining_proba={remaining_proba}")
            break
        pick_left = term_left > term_right and term_left > 0

        if pick_left:
            # print(f"Picking left at i={i}, summand {n_left}, {term_left}")
            partial_sum += term_left
            cumulative_proba += prob_left
            if n_left >= 1:
                prob_left *= n_left / lam_t
                n_left -= 1
                call_left = bs_call(spot_n(n_left), strike, vol_n(n_left), rate, maturity)
                term_left = call_left * prob_left
            else:
                print("Left Max")
                term_left = -1
        else:
            # print(f"Picking right at i={i}, summand {n_right}, {term_right}")
            partial_sum += term_right
            cumulative_proba += prob_right
            n_right += 1
            prob_right *= lam_t / n_right
            call_right = bs_call(spot_n(n_right), strike, vol_n(n_right), rate, maturity)
            term_right = call_right * prob_right

        if cumulative_proba + 1.e-18 >= 1.0:
            print(f"stopping at cumulative probability = 1 after {i} iterations")
            break

    return partial_sum


def benchmark_merton_call():
    spot = 100
    maturity = 1.0
    rate = 0.
    vol = 0.2

    intensity = 1.0
    mean_jump = 0
    std_jump = 0.4

    strike = 100
    summands = 10
    price = merton_call_price(spot, maturity, rate, vol, intensity, mean_jump, std_jump, strike, summands=summands, method='internet')
    print(f"Merton analytical: {price}")

    price = merton_call_price(spot, maturity, rate, vol, intensity, mean_jump, std_jump, strike, summands=summands, method='savine')
    print(f"Merton analytical: {price}")

    # monte carlo
    avg_monte = 0
    nbr_paths = 10_000_000
    time_steps = 1
    n_reruns = 1
    for i in range(n_reruns):
        paths = simulate_merthon_paths(spot, maturity, rate, vol, intensity, mean_jump, std_jump, time_steps, nbr_paths)
        merton_monte = np.mean(np.exp(-rate * maturity) * np.maximum(paths[-1, :] - strike, 0))
        print(f"Merton MC: {merton_monte}")
        avg_monte += merton_monte / n_reruns
    print(f"Average Merton MC: {avg_monte}")


def plot_merton_call():
    spot = 100
    maturity = 1.0
    rate = 0.0
    vol = 0.15

    intensity = 5.0
    mean_jump = -0.16251892949777494
    std_jump = 0.10

    strike = 100
    summands = 100

    strikes = np.linspace(50, 200, 16)
    maturities = np.linspace(1, 5, 5)
    prices = []
    for maturity in maturities:
        price_per_strk = []
        for strike in strikes:
            price = merton_call_price(spot, maturity, rate, vol, intensity, mean_jump, std_jump, strike, summands=summands, method='savine')
            # price = bs_call(spot, strike, vol, rate, maturity)
            print(f"Merton for K={strike}, T={maturity} is: {price}")
            price_per_strk.append(price)
        prices.append(price_per_strk)
    plt.figure()
    for counter, t in enumerate(maturities):
        intrinsics = np.maximum(spot - strikes, 0)
        plt.plot(strikes, intrinsics)
        plt.plot(strikes, prices[counter])
        plt.title(f"TTE = {t}")
        plt.xlabel("K")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()


def print_merton():
    spot = 100
    maturity = 5.0
    rate = 0.0
    vol = 0.15

    mean_jump = -0.15
    intensity = 0.05
    std_jump = 0.1

    strike = 50
    strikes = np.linspace(50, 200, 16)
    maturities = np.linspace(1, 5, 5)
    prices = []
    for maturity in maturities:
        price_per_strk = []
        for strike in strikes:
            price = merton_call_price_opt(spot, maturity, rate, vol, intensity, mean_jump, std_jump, strike)
            print(f"Merton for K={strike}, T={maturity} is: {price}")
            price_per_strk.append(price)
        prices.append(price_per_strk)
    plt.figure()
    for counter, t in enumerate(maturities):
        intrinsics = np.maximum(spot - strikes, 0)
        plt.plot(strikes, intrinsics)
        plt.plot(strikes, prices[counter])
        plt.title(f"TTE = {t}")
        plt.xlabel("K")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()


def plot_merton_iv_2d():
    spot = 100
    maturity = 5.0
    rate = 0.0
    vol = 0.15

    mean_jump = -0.15
    intensity = 0.5
    std_jump = 0.2

    strike_min = 40
    strike_max = 200
    mat_min = 0.4
    mat_max = 5.5

    def IV(strk, mat):
        merton_prc = merton_call_price(spot, mat, rate, vol, intensity, mean_jump, std_jump, strk, summands=30, method='internet')
        iv = bs_call_iv(spot, strk, merton_prc, rate, mat)
        return iv

    num_strikes = 30
    num_mats = 30

    strike_range = np.linspace(strike_min, strike_max, num_strikes)
    mat_range = np.linspace(mat_min, mat_max, num_mats)
    iv_surface = np.zeros((num_strikes, num_mats))

    strikes, maturities = np.meshgrid(strike_range, mat_range)

    for i in range(num_strikes):
        for j in range(num_mats):
            iv_surface[i, j] = IV(strikes[i, j], maturities[i, j])

    # plot risk view
    rv_strike = np.linspace(50, 180, 14)
    rv_mat = np.linspace(1, 5, 5)
    riskview_k, riskview_t = np.meshgrid(rv_strike, rv_mat)
    riskview_iv = np.zeros((5, 14))
    for i in range(5):
        for j in range(14):
            riskview_iv[i, j] = IV(riskview_k[i, j], riskview_t[i, j]) + 1.e-3

    # 600k paths, 3y call 120
    vega_matrix = np.array([[-0.0460348, -0.221728, -0.194649, -0.21263, 0.501785, 0.0810492, 0.150104, -0.171543, 0.133114, 0.233465, -0.371074, -0.120994, -0.409019, -0.00176027, ],
                            [-0.00111691, -0.0152889, 0.000852723, -0.152864, -0.0726684, 0.150958, 0.515676, -0.090548, 0.249351, 0.0255403, 0.264953, 0.242036, -2.37569, 1.31905, ],
                            [-0.000149354, 0.000235125, -0.0214169, -0.0372974, -0.156937, -0.0859814, 2.81429, 60.2662, 3.83625, -0.721873, 0.325535, -1.47745, -0.882827, 0.927791, ],
                            [0, 0, 0, 0, 1.86145e-09, 1.07888e-06, 5.58932e-05, 0.000257102, 5.08467e-05, 5.33631e-07, -6.61242e-07, -6.28416e-07, 2.06511e-07, -4.07059e-06, ],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


    # 300k paths, 2y call 105
    vega_matrix = np.array([[-0.0398405, -0.285854, -0.368455, 0.146603, 0.355723, 0.406365, 0.260059, 0.0560151, -0.148932, 0.238754, -1.29628, 0.0539962, -0.0547512, 0.0415828, ],
[0.00242438, -0.00856379, -0.019011, -0.00502648, -0.256297, 28.8798, 27.2184, -0.627725, 0.525972, 1.37219, -1.61495, 0.266487, -0.43941, 0.499073, ],
[0, 0, 0, 1.20093e-08, 4.99568e-06, 0.000152957, 0.000148638, 5.37888e-06, -1.14105e-06, -1.86467e-06, -1.48343e-08, 2.32549e-06, 9.4301e-07, -5.77093e-07, ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # 300k paths 2.5y call 85
    vega_matrix = np.array([
        [-0.151846, -0.70378, -0.646578, 1.00849, 0.531877, -0.0428076, 0.217497, 0.07512, -0.168478, 0.572574,
         -1.27777, 0.216125, -0.113374, 0.0454884, ],
        [0.0509157, -0.291069, -0.489891, 13.811, 13.1965, 0.703092, -1.08595, -0.086781, -1.20526, 2.53638, -1.12951,
         0.414592, -2.14503, 1.69093, ],
        [-0.00159746, -0.00851664, -0.0896952, 12.559, 12.8425, 0.0898832, -0.297002, -0.154269, -0.000794352,
         0.00620517, 0.475337, -0.549427, 0.0751299, -0.161215, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]])

    # 300k paths 3y call 120 upout barrier 150
    # smoothing = 0.001
    vega_matrix = np.array([[-0.315772, 0.487014, 1.1524, -1.11235, -0.108645, -2.39211, 4.09419, 2.78941, -7.07862, -0.433601, 1.65066, -0.0949417, 0, 0, ],
[-0.203359, 0.386217, -0.727601, 0.13253, 2.55086, -3.57928, 4.28159, 4.22499, -14.8259, -8.06827, 17.6968, -0.816512, 0, 0, ],
[-0.000229982, 0.00627085, -0.0338144, 0.0441019, -0.15966, 0.767123, 2.89145, 53.3942, -22.2273, -87.948, 47.0891, -1.60974, 0, 0, ],
[0, 0, 0, 0, 1.63501e-09, 1.0487e-06, 4.88057e-05, 0.000208151, 2.03665e-05, -0.000194246, -9.52665e-05, -7.2784e-11, 0, 0, ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]])

    # 600k paths 3y call 120 upout barrier 150, relaxed merton params
    # smoothing = 0.001, 120 time steps, 300 spot steps
    vega_matrix = np.array([[-5.14498e-07, -0.00206793, 0.0101515, -0.66982, 0.551905, -1.90183, 2.04829, 1.51506, -0.368275, -0.0184726, 0.00852424, -0.000486764, 0, 0, ],
[-1.72493e-07, 0.00145105, 0.0385751, -0.210948, -1.215, 3.12047, -1.5027, -4.25991, -7.61992, 2.00109, 9.51451, -0.27779, 0, 0, ],
[0, -1.3373e-05, 0.000422833, -0.00243162, -0.115687, -0.0401766, 3.20692, 51.9101, -10.3248, -73.4048, 22.8791, -1.11336, 0, 0, ],
[0, 0, 0, 0, 0, 5.24237e-10, 9.31928e-06, 0.000119283, 7.29981e-06, -4.61661e-05, -7.16117e-05, -5.24097e-12, 0, 0, ],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]])

    # 1m paths 3y call 120 upout barrier 120, relaxed merton params
    # smoothing = 0.001, 240 time steps, 300 spot steps,
    vega_matrix = np.array([
        [-0.000128218, -0.0079029, 0.034563, -0.210307, 0.566788, -0.536413, 0.237895, -1.06534, -0.0253534, 0.173004,
         0.366464, -0.00295543, 0, 0, ],
        [-3.14663e-05, -0.000723477, 0.0157918, 0.0361545, -0.266102, 1.64231, -0.880909, -2.13758, -4.50251, -0.219733,
         8.8065, -1.46362, 0, 0, ],
        [0, -5.79628e-05, 6.79342e-05, 0.00519381, 0.00188361, 0.0498922, 2.08295, 52.8797, -14.4442, -61.7788, 19.0537,
         -4.21678, 0, 0, ],
        [0, 0, 0, 0, 0, 1.04489e-09, 9.61223e-06, 0.000136373, 7.83103e-06, -4.85249e-05, -9.17566e-05, 2.87952e-07, 0,
         0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]])

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")
    # surf = ax.plot_surface(strikes, maturities, iv_surface, cmap='viridis', edgecolor="none", alpha=0.7)
    surf = ax.plot_surface(strikes, maturities, iv_surface, color='lightgray', alpha=0.5)

    # 3d bars
    x = riskview_k.flatten()
    y = riskview_t.flatten()
    z = riskview_iv.flatten()
    dy = 1.2* 5./60
    dx = 1.2* 160./60  # width of the bars
    dz = vega_matrix.flatten() * 1/600
    norm = plt.Normalize(dz.min(), dz.max())
    colors = cm.viridis(norm(dz))
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, zsort='average')
    mappable = cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.5, label='Vega')

    """
    sc = ax.scatter(riskview_k, riskview_t, riskview_iv, c=vega_matrix, s=3 + 3*abs(vega_matrix), cmap="coolwarm", edgecolor="k", zorder=10, linewidth=0.5, alpha=1.0, depthshade=False)
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label="Vega")
    """
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("IV")
    # ax.set_title("Implied Volatility Surface")
    ax.set_title("Vega - 3Y Up-Out-Call (Strike 120, Barrier 150)")
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    plot_merton_paths()
    # benchmark_merton_call()
    # plot_merton_call()
    # print_merton()
    # plot_merton_iv_2d()
    # iv = bs_call_iv(100, 120, 7.8331, 0, 3)
    # print(bs_call(100, 120, iv, 0, 3))
    # print(bs_vega(100, 120, iv, 0, 3))
