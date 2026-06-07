import numpy as np
import matplotlib.pyplot as plt

# setup: arrays x and y
# target: interpolate for point x' using various interpolation schemes

# piecewise linear is easiest, and is local, but only C0
# cubic hermite spline is still mostly local but C1
# cubic hermite spline matches first derivatives at grid points. I don't have these, so I will use a proxy.


def get_slopes(x_base, y_base, slope_type="cpp"):
    slopes = np.empty_like(y_base)
    if slope_type == "pchip":
        deltas = np.diff(y_base) / np.diff(x_base)
        slopes[0] = deltas[0]
        slopes[-1] = deltas[-1]
        slopes[0] = 0
        slopes[-1] = 0
        for i in range(1, len(slopes) - 1):
            if deltas[i - 1] * deltas[i] < 0:
                slopes[i] = 0
            else:
                slopes[i] = 0.5 * (deltas[i] + deltas[i - 1])
        for i in range(len(slopes) - 1):
            if deltas[i] == 0:
                slopes[i] = slopes[i + 1] = 0
                i += 1
                continue
            alpha = slopes[i] / deltas[i]
            beta = slopes[i + 1] / deltas[i]
            if alpha < 0:
                slopes[i] = 0
            if beta < 0:
                slopes[i + 1] = 0
            if alpha > 3:
                slopes[i] = 3 * deltas[i]
            if beta > 3:
                slopes[i + 1] = 3 * deltas[i]
            i += 1
    else:
        slopes[0] = 0
        slopes[-1] = 0
        i = 0
        y_it = 0
        x_it = 0
        delta1 = (y_base[y_it+1]-y_base[y_it]) / (x_base[x_it+1]-x_base[x_it])
        delta0 = delta1
        delta2 = (y_base[y_it+2] - y_base[y_it+1]) / (x_base[x_it+2] - x_base[x_it+1])
        s0 = 0
        s1 = 0 if delta1*delta2<0 else 0.5*(delta1+delta2)
        new_s1 = "zero"
        while i < len(slopes) -1:
            if delta1 != 0:
                a = s0 / delta1
                b = s1 / delta1
                if a<0:
                    slopes[i] = 0
                elif a > 3:
                    slopes[i] = 3 * delta1
                else:
                    if i == 0:
                        slopes[i] = 0
                    else:
                        if new_s1 == "zero":
                            slopes[i] = 0
                        elif new_s1 == "delta":
                            prev_delta = (y_base[y_it]-y_base[y_it-1]) / (x_base[x_it]-x_base[x_it-1])
                            prev_delta = delta0
                            slopes[i] = 3 * prev_delta
                            # slopes[i] = s0
                        elif new_s1 == "avg":
                            prev_delta = (y_base[y_it] - y_base[y_it-1]) / (x_base[x_it] - x_base[x_it - 1])
                            prev_delta = delta0
                            slopes[i] = 0 if delta1 * prev_delta < 0 else 0.5 * (delta1 + prev_delta)
                            # slopes[i] = s0
                        else:
                            slopes[i] = s0  # 0 if delta1 * delta2 <0 else 0.5 * (delta2 * delta1)
                            raise Exception("bad")
                if b < 0:
                    s1 = 0
                    new_s1 = "zero"
                elif b > 3:
                    new_s1 = "delta"
                    s1 = 3 * delta1
                else:
                    # no change to s1 - it means we get whatever we had before for s1.
                    new_s1 = "avg"
                    pass
            else:
                new_s1 = "zero"
                s1 = 0
                slopes[i] = 0
            i += 1
            if i < len(slopes) - 1:
                y_it += 1
                x_it += 1
                s0 = s1
                delta0 = delta1
                delta1 = delta2

            if i < len(slopes) - 2:
                # delta2 = (y_base[y_it+2] - y_base[y_it+1]) / (x_base[x_it+2] - x_base[x_it+1])
                delta2 = (y_base[y_it + 2] - y_base[y_it + 1]) / (x_base[x_it + 2] - x_base[x_it + 1])
                s1 = 0 if delta1 * delta2 < 0 else 0.5 * (delta1 + delta2)
    return slopes


def pchip(x_base, y_base, x_new, slope_type='threepoint'):
    # piecewise cubic hermite - not necessarily monotone!
    # estimate slopes on grid
    # use slope = 0 on first and last point? Or use method below with y_{i-1}=y_i for left boundary?
    # use m_i = 0.5 * [(y_i-y_{i-1})/(x_i-x_{i-1}) + (y_{i+1}-y_i)/(x_{i+1}-x_i)]
    # this is the average of left and right linear interpolation
    y_new = np.empty_like(x_new)
    slopes = np.empty_like(y_base)
    if slope_type == 'threepoint':
        # choice of non-monotone slopes
        slopes[0] = 0
        slopes[-1] = 0
        for idx in range(1, len(x_base) - 1):
            slopes[idx] = 0.5 * ((y_base[idx]-y_base[idx-1])/(x_base[idx]-x_base[idx-1]) + (y_base[idx+1]-y_base[idx])/(x_base[idx+1]-x_base[idx]))
    elif slope_type == 'pchip':
        deltas = np.diff(y_base) / np.diff(x_base)
        slopes[0] = deltas[0]
        slopes[-1] = deltas[-1]
        slopes[0] = 0
        slopes[-1] = 0
        for i in range(1, len(slopes)-1):
            if deltas[i-1] * deltas[i] < 0:
                slopes[i] = 0
            else:
                slopes[i] = 0.5 * (deltas[i] + deltas[i-1])
        for i in range(len(slopes)-1):
            if deltas[i] == 0:
                slopes[i] = slopes[i+1] = 0
                i += 1
                continue
            alpha = slopes[i] / deltas[i]
            beta = slopes[i+1] / deltas[i]
            if alpha < 0:
                slopes[i] = 0
            if beta < 0:
                slopes[i+1] = 0
            if alpha > 3:
                slopes[i] = 3 * deltas[i]
            if beta > 3:
                slopes[i+1] = 3 * deltas[i]
            i += 1
    elif slope_type == "other":
        deltas = np.diff(y_base) / np.diff(x_base)
        slopes[0] = deltas[0]
        slopes[-1] = deltas[-1]
        slopes[0] = 0
        slopes[-1] = 0
        i = 0
        s0 = 0
        s1 = 0
        if len(slopes) > 2:
            s1 = 0 if deltas[0]*deltas[1] < 0 else 0.5 * (deltas[1] + deltas[0])
            # s1 = max(0, deltas[0]*deltas[1])
            # slopes[1] = 0 if deltas[0]*deltas[1] < 0 else 0.5 * (deltas[1] + deltas[0])
        # if len(slopes) > 3:
            # slopes[2] = 0 if deltas[1]*deltas[2] < 0 else 0.5 * (deltas[2] + deltas[1])
            # s2 = 0 if deltas[1]*deltas[2] < 0 else 0.5 * (deltas[2] + deltas[1])
        while i < len(slopes)-1:
            # if i + 3 < len(slopes):
                # slopes[i+2] = 0 if deltas[i+2]*deltas[i+1] < 0 else 0.5 * (deltas[i+2] + deltas[i+1])
                # s2 = 0 if deltas[i+2]*deltas[i+1] < 0 else 0.5 * (deltas[i+2] + deltas[i+1])
            if deltas[i] == 0:
                slopes[i] = 0
                s1 = 0
                # slopes[i] = slopes[i+1] = 0
                i += 1
            else:
                # a = slopes[i] / deltas[i]
                a = s0 / deltas[i]
                b = s1 / deltas[i]
                # b = slopes[i+1] / deltas[i]
                if a < 0: slopes[i] = 0
                elif a > 3: slopes[i] = 3 * deltas[i]
                else: slopes[i] = s0
                if b < 0: s1 = 0 # slopes[i+1] = 0
                if b > 3: s1 = 3 * deltas[i] # slopes[i+1] = 3 * deltas[i]
                i += 1
            s0 = s1
            if i + 2 < len(slopes):
                # slopes[i+2] = 0 if deltas[i+2]*deltas[i+1] < 0 else 0.5 * (deltas[i+2] + deltas[i+1])
                s1 = 0 if deltas[i+1]*deltas[i] < 0 else 0.5 * (deltas[i+1] + deltas[i])
            # s1 = s2
    else:
        print("In CPP part")
        slopes[0] = 0
        slopes[-1] = 0
        d1 = (y_base[1]-y_base[0]) / (x_base[1]-x_base[0])
        d2 = (y_base[2] - y_base[1]) / (x_base[2] - x_base[1])
        s1 = 0 if d1*d2<0 else 0.5*(d1+d2)
        if d1 != 0:
            b = s1 / d1
            if b < 0:
                s1 = 0
                new_s1 = "zero"
            elif b > 3:
                new_s1 = "delta"
                s1 = 3 * d1
            else:
                # no change to s1 - it means we get whatever we had before for s1.
                new_s1 = "avg"
        else:
            new_s1 = "zero"
            s1 = 0
        for i in range(1, len(slopes)-1):
            s0 = s1
            delta0 = (y_base[i]-y_base[i-1]) / (x_base[i]-x_base[i-1])
            delta1 = (y_base[i+1]-y_base[i]) / (x_base[i+1]-x_base[i])
            if i < len(slopes) - 2:
                delta2 = (y_base[i + 2] - y_base[i + 1]) / (x_base[i + 2] - x_base[i + 1])
                s1 = 0 if delta1 * delta2 < 0 else 0.5 * (delta1 + delta2)

            if delta1 != 0:
                a = s0 / delta1
                b = s1 / delta1
                if a<0:
                    slopes[i] = 0
                elif a > 3:
                    slopes[i] = 3 * delta1
                else:
                    if new_s1 == "zero":
                        slopes[i] = 0
                    elif new_s1 == "delta":
                        slopes[i] = 3 * delta0
                    elif new_s1 == "avg":
                        slopes[i] = 0 if delta1 * delta0 < 0 else 0.5 * (delta1 + delta0)
                if b < 0:
                    s1 = 0
                    new_s1 = "zero"
                elif b > 3:
                    new_s1 = "delta"
                    s1 = 3 * delta1
                else:
                    # no change to s1 - it means we get whatever we had before for s1.
                    new_s1 = "avg"
            else:
                new_s1 = "zero"
                s1 = 0
                slopes[i] = 0

    for idx in range(len(x_new)):
        x = x_new[idx]
        # constant extrapolation
        if x <= x_base[0]:
            y_new[idx] = y_base[0]
        elif x >= x_base[-1]:
            y_new[idx] = y_base[-1]
        else:
            right_idx = np.searchsorted(x_base, x)
            # by construction, this left_idx is between 1 and len(x_base)-1
            left_idx = right_idx  - 1
            interval = (x_base[right_idx] - x_base[left_idx])
            t = (x-x_base[left_idx]) / interval
            y_0 = y_base[left_idx]
            y_1 = y_base[right_idx]
            m0 = slopes[left_idx]
            m1 = slopes[right_idx]
            y_new[idx] = h00(t) * y_0 + h10(t) * interval * m0 + h01(t) * y_1 + h11(t) * interval * m1
    return y_new

def h00(t):
    return 2 * t**3 - 3 * t**2 + 1
def h10(t):
    return t**3 - 2 * t**2 + t
def h01(t):
    return -2 * t**3 + 3 * t**2
def h11(t):
    return t**3 - t**2

    # p(t) = h00(t) p0 + h10(t)*(x_{k+1}-x_k)* m0 + h01(t) p1 + h11(t) *(x_{k+1}-x_k)* m1
    # t = (x-x_k) / (x_{k+1}-x_k)
    # h00(t) = 2*t**3 - 3*t**2 + 1 = (1+2t)(1-t)**2
    # h10(t) = t**3 - 2*t**2 + 1 = t * (1-t)**2
    # h01(t) = -2t**3 + 3 t**2 = t**2 * (3-2t)
    # h11(t) = t**3 - t**2 = t**2 *(t - 1)
    # maybe this can all be optimized a little
    # can reuse t**3 and t**2 for example, or


def lin_interp(x_base, y_base, x_new):
    y_new = np.empty_like(x_new)
    for idx in range(len(x_new)):
        x = x_new[idx]
        i = np.searchsorted(x_base, x)


def test_slope_computation():
    spots = np.array([-0.69314718056, -0.683843191559, -0.674539202558, -0.665235213558, -0.655931224557, -0.646627235556, -0.637323246555, -0.628019257554, -0.618715268554, -0.609411279553, -0.600107290552, -0.590803301551, -0.58149931255,-0.572195323549, -0.562891334549, -0.553587345548, -0.544283356547, -0.534979367546, -0.525675378545, -0.516371389545,-0.507067400544, -0.497763411543, -0.488459422542, -0.479155433541, -0.469851444541, -0.46054745554, -0.451243466539,-0.441939477538, -0.432635488537, -0.423331499537, -0.414027510536, -0.404723521535, -0.395419532534, -0.386115543533,-0.376811554533, -0.367507565532, -0.358203576531, -0.34889958753, -0.339595598529, -0.330291609529, -0.320987620528,-0.311683631527,-0.302379642526, -0.293075653525, -0.283771664525, -0.274467675524, -0.265163686523, -0.255859697522,-0.246555708521,-0.237251719521, -0.22794773052, -0.218643741519, -0.209339752518, -0.200035763517, -0.190731774516,-0.181427785516, -0.172123796515, -0.162819807514, -0.153515818513, -0.144211829512, -0.134907840512, -0.125603851511,-0.11629986251, -0.106995873509, -0.0976918845084, -0.0883878955076, -0.0790839065068, -0.069779917506, -0.0604759285052,-0.0511719395044,-0.0418679505036, -0.0325639615028, -0.023259972502, -0.0139559835012, -0.0046519945004, 0.0046519945004,0.0139559835012, 0.023259972502, 0.0325639615028, 0.0418679505036, 0.0511719395044, 0.0604759285052, 0.069779917506,0.0790839065068,0.0883878955076, 0.0976918845084, 0.106995873509, 0.11629986251, 0.125603851511, 0.134907840512,0.144211829512, 0.153515818513, 0.162819807514, 0.172123796515, 0.181427785516, 0.190731774516, 0.200035763517, 0.209339752518,0.218643741519,0.22794773052, 0.237251719521, 0.246555708521, 0.255859697522, 0.265163686523, 0.274467675524, 0.283771664525,0.293075653525, 0.302379642526, 0.311683631527, 0.320987620528, 0.330291609529, 0.339595598529, 0.34889958753,0.358203576531, 0.367507565532, 0.376811554533, 0.386115543533, 0.395419532534, 0.404723521535, 0.414027510536, 0.423331499537,0.432635488537, 0.441939477538, 0.451243466539, 0.46054745554, 0.469851444541, 0.479155433541, 0.488459422542,0.497763411543, 0.507067400544, 0.516371389545, 0.525675378545, 0.534979367546, 0.544283356547, 0.553587345548, 0.562891334549,0.572195323549, 0.58149931255, 0.590803301551, 0.600107290552, 0.609411279553, 0.618715268554, 0.628019257554,0.637323246555, 0.646627235556, 0.655931224557, 0.665235213558, 0.674539202558, 0.683843191559, 0.69314718056, ])
    vols = np.array([0.400199505436, 0.397947678383, 0.395685525101, 0.3934129634, 0.391129850655, 0.388836089721,0.386531546541, 0.384216146787, 0.381889756101, 0.379552280853, 0.377203564005, 0.374843541561, 0.372472060157, 0.370089005513,0.367694292475, 0.365287776437, 0.362869346038, 0.360438903896, 0.357996321677, 0.355541478068, 0.353074253475,0.350594578378, 0.348102297542, 0.345597302236, 0.343079494645, 0.340548761992, 0.338005022206, 0.335448137902, 0.332878059348,0.330294618697, 0.327697791121, 0.325087460008, 0.322463537473, 0.319825970183, 0.317174674416, 0.314509593016,0.311830666075, 0.309137849859, 0.306431111251, 0.303710409806, 0.300975758963, 0.298227129951, 0.295464548827, 0.292688036754,0.28989764509, 0.287093436819, 0.28427550159, 0.281443938336, 0.278598874226, 0.275740480713, 0.27286893468,0.269984460112, 0.267087306961, 0.264177761711, 0.261256165469, 0.258322886544, 0.25537835292, 0.252423040261, 0.249457487805,0.246482277609,0.243498079121, 0.240505630232, 0.237505727779, 0.234499273747, 0.231487251499, 0.228470743744,0.225450926944, 0.22242909804, 0.219406674356, 0.216385192484, 0.213366325023, 0.210351889697, 0.207343854837, 0.204344343127,0.201355648944, 0.198380236679, 0.195420752486, 0.192480026587, 0.189561083893, 0.186667139553, 0.183801605492,0.180968086294, 0.178170380469, 0.175412465645, 0.172698496544, 0.170032785144, 0.167419786754, 0.164864071328, 0.16237029566,0.159943176404, 0.157587437495, 0.155307779315, 0.153108819934, 0.150995046844, 0.148970755773, 0.147039997175,0.145206512583, 0.143473676795, 0.14184444608, 0.140321299741, 0.1389062097, 0.137600592332, 0.136405287295, 0.135320549319,0.134346033679, 0.133480823424, 0.132723428255, 0.132071843185, 0.131523560066, 0.13107565101, 0.130724787099, 0.130467326289,0.130299366502, 0.130216795743, 0.130215361104, 0.130290718769, 0.130438510773, 0.130654356105, 0.130933957385,0.131273091515, 0.131667637584, 0.132113642112, 0.132607297408, 0.133144946748, 0.133723159642, 0.134338643189, 0.134988322039,0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039,0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039,0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039, 0.134988322039,0.134988322039, ])
    slopes = np.array([0, -0.242583064894, -0.243697352969, -0.244823722679, -0.245962977721, -0.247114657638, -0.248277536324, -0.249451629831, -0.250637975451, -0.251837792163, -0.253049487257, -0.254272863362, -0.255510622774, -0.256759099841, -0.258019924361, -0.259294504571, -0.260580302744, -0.261878230982, -0.263189575314, -0.264513866112, -0.265848319965, -0.267194852275, -0.268555570197, -0.269927387949, -0.271310523002, -0.272704129302, -0.274109529255, -0.275524984885, -0.276952133338, -0.27838963626, -0.279834740146, -0.281290833823, -0.28275451663, -0.284225564799, -0.285704183774, -0.287189093837, -0.28867957375, -0.290174183585, -0.291672746638, -0.29317276099, -0.294673599367, -0.296174583616, -0.297673030147,-0.29916757945, -0.300655983932, -0.302136185837, -0.30360625332, -0.305064170004, -0.306506038574, -0.307929187428, -0.309330793504, -0.310706930032, -0.312054238283, -0.313367819518, -0.314643276484, -0.315875940313, -0.317060041788, -0.318189602046, -0.319258903455, -0.320260948504, -0.321187362535, -0.322031299727, -0.322783941628, -0.323435264155, -0.323975555165, -0.324394437403, -0.324680398019, -0.32482049299, -0.324801843335, -0.324610730542, -0.324232046409, -0.323649898214, -0.322847897254, -0.32180852172, -0.320513408151, -0.318943651857, -0.317079593044, -0.314900877071, -0.312386817839, -0.309516617032, -0.306269346307, -0.302624230449, -0.298561221903, -0.294061177653, -0.289106129645, -0.283679924269, -0.277768697699, -0.27136162208, -0.264450813672, -0.257032664422, -0.249108048638, -0.240682655626, -0.231767926149, -0.222381182951, -0.212545912793, -0.20229189815, -0.191655449038, -0.18067876597, -0.169409973153, -0.157901969753, -0.146211878012, -0.134400546109, -0.122530401363, -0.110665092982, -0.0988675875949, -0.0871994487656, -0.0757191478893, -0.0644813847498, -0.0535357563062, -0.0429263709723, -0.0326916078959, -0.0228622689063, -0.0134636093208, -0.000462588228113, 0, 0.0119921502837, 0.0195420123541, 0.0266254942822, 0.0332510824106, 0.039428260205, 0.0451715170985, 0.0504976856818, 0.0554227136211, 0.0599668719312, 0.0641497126478, 0.0679903209999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

    comp_slopes = get_slopes(spots, vols)
    print(comp_slopes)
    print(comp_slopes.shape)
    print(slopes.shape)
    diff = np.abs(slopes - comp_slopes)
    print("diff: ", np.max(diff))
    plt.figure()
    plt.scatter(spots, vols)
    plt.show()
    plt.figure()
    plt.plot(spots, slopes, label="cpp")
    plt.plot(spots, comp_slopes, label="py")
    plt.legend()
    plt.show()


def debug_slopes():
    plot=False
    x_base = np.array([-3, -2, -1, 0.0, 1, 2, 3, 4])
    for i in range(1000):
        rng = np.random.default_rng()
        y_base = rng.random(x_base.shape)
        # y_base = np.array([-1, -1, -1, 0.5, 1, 2, 2])
        x_interp = np.linspace(x_base[0], x_base[-1], 300)
        # y_interp_3p = pchip(x_base, y_base, x_interp, slope_type='threepoint')
        y_interp_m = pchip(x_base, y_base, x_interp, slope_type='pchip')
        # y_interp_w = pchip(x_base, y_base, x_interp, slope_type='other')
        y_interp_cpp = pchip(x_base, y_base, x_interp, slope_type='CPP')

        sum_of_squares = np.sum(np.abs(y_interp_m - y_interp_cpp)**2)
        if sum_of_squares > 0:
            print(i)
            print(sum_of_squares)
            plot=True
        if plot:
            plt.figure()
            # plt.plot(x_interp, y_interp_3p, label="3-point")
            plt.plot(x_interp, y_interp_m, label="Monotone PCHIP")
            plt.plot(x_interp, y_interp_cpp, label="CPP")
            plt.scatter(x_base, y_base)
            plt.legend()
            plt.show()
            plot=False


if __name__ == "__main__":
    # debug_slopes()
    test_slope_computation()
