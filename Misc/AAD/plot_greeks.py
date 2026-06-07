import numpy as np
import matplotlib.pyplot as plt


class SSVI:
    def __init__(self, spot, vol, rho, eta=1.0, gamma=0.5, kappa=1.0, vol_0=0.4, phi="Heston"):
        self.spot = spot
        self.vol = vol
        self.rho = rho
        self.eta = eta
        self.gamma = gamma
        self.kappa = kappa
        assert (kappa >= 0)
        self.vol_0 = vol_0
        assert phi in ("Heston", "PowerLaw", "PowerLawFix")
        self.phi_type = phi
        if phi == "Heston":
            assert gamma >= 0.25 * (1 + abs(rho))
        elif phi == "PowerLawFix":
            assert eta * (1 + abs(rho)) <= 2

    def phi(self, theta):
        if self.phi_type == "Heston":
            gt = self.gamma * theta
            return 1 / gt * (1 - (1 - np.exp(-gt))/gt)
        elif self.phi_type == "PowerLawFix":
            return self.eta / (np.power(theta, self.gamma) * np.power(1 + theta, 1 - self.gamma))
        else:
            return self.eta / np.power(theta, self.gamma)

    def vol_func(self, maturity):
        # return self.vol + (self.vol_0 - self.vol) * np.exp(-self.kappa * maturity)
        return self.vol

    def implied_vol(self, strike, maturity):
        theta = self.vol_func(maturity)**2 * maturity
        phi_val_k = self.phi(theta) * np.log(strike/self.spot)
        total_var = theta / 2 * (1 + self.rho * phi_val_k + np.sqrt(
            (phi_val_k + self.rho)**2 + 1 - self.rho**2))
        return np.sqrt(total_var/maturity)

    def deriv_t(self, spot, time):
        eps = 0.0001
        return (self.implied_vol(spot, time + eps)**2*(time+eps) - self.implied_vol(spot, time-eps)**2*(time-eps)) / (2 * eps)

    def deriv_xx(self, spot, time):
        x = np.log(spot / self.spot)
        theta = self.vol_func(time) ** 2 * time
        p = self.phi(theta)
        rho = self.rho
        return 0.5*theta*p*p*(1.-rho*rho)/ ((p*p*x*x + 2.*p*rho*x + 1.)*np.sqrt(p*p*x*x+2.*p*rho*x+1.))

    def deriv_x(self, spot, time):
        x = np.log(spot/self.spot)
        theta = self.vol_func(time) ** 2 * time
        p = self.phi(theta)
        rho = self.rho
        return 0.5*theta*p*(p*x+rho*np.sqrt(p*p*x*x+2.*p*rho*x+1.)+rho)/np.sqrt(p*p*x*x+2.*p*rho*x+1.)

    def g(self, spot, time):
        x = np.log(spot/self.spot)
        w = self.implied_vol(spot, time)**2*time
        w1 = self.deriv_x(spot, time)
        w2 = self.deriv_xx(spot, time)
        return (1. - 0.5 * x * w1 / w) * (1. - 0.5 * x * w1 / w) - 0.25 * w1 * w1 * (0.25 + 1. / w) + 0.5 * w2

    def local_vol(self, spot, time):
        return np.sqrt(self.deriv_t(spot, time) / self.g(spot, time))


def plot_vol():
    spot = 1.0
    vol = 0.2
    rho = -0.7
    eta = 0.7
    gamma = 0.4

    surface = SSVI(spot, vol, rho, eta, gamma, phi="PowerLawFix")
    strikes = np.linspace(0.5, 2.0, 200)
    times = np.linspace(0.05, 2, 30)
    s, t = np.meshgrid(strikes, times)
    vols = np.empty((len(times), len(strikes)))
    for i in range(len(times)):
        vols[i, :] = surface.implied_vol(strikes, times[i])

    fig = plt.figure()
    fig.suptitle("Implied Vol")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(s, t, vols)
    ax1.set_title("PowerLawFix SSVI")
    ax1.set_xlabel("strike")
    ax1.set_ylabel("maturity")
    ax1.set_zlabel("vol")

    spot = 1.0
    vol = 0.2
    rho = -0.7
    eta = 0.9
    gamma = 0.4

    surface = SSVI(spot, vol, rho, eta, gamma, phi="PowerLawFix")
    strikes = np.linspace(0.5, 2.0, 200)
    times = np.linspace(0.05, 2, 30)
    s, t = np.meshgrid(strikes, times)
    vols = np.empty((len(times), len(strikes)))
    for i in range(len(times)):
        vols[i, :] = surface.implied_vol(strikes, times[i])

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(s, t, vols)
    ax2.set_title("PowerLawFix 2 SSVI")
    ax2.set_xlabel("strike")
    ax2.set_ylabel("maturity")
    ax2.set_zlabel("vol")
    plt.show()


# plot market data: spots, ivs
# plot results:
# three dupire delta (bar plot),
# three ss delta (bar plot),
# three vega matrices (separate matrix plots)

def plot_sensis(ss_delta, d_delta, corr_delta, vegas1, vegas2, vegas3):
    layout = [
        ['A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        ['D', 'D', 'D', 'D', 'E', 'E', 'E', 'E']
    ]

    fig, axd = plt.subplot_mosaic(layout, figsize=(12, 8), constrained_layout=True)

    asset_names = ['S1', 'S2', 'S3']
    # Access axes using the keys defined in layout
    axd['A'].set_title('Spot Delta')
    x = np.arange(3)  # [0, 1, 2]
    width = 0.35
    # Plot the bars with offsets
    rects_ss = axd['A'].bar(x - width / 2, ss_delta, width, label='Sticky Strike', color='blue')
    rects_d = axd['A'].bar(x + width / 2, d_delta, width, label='Const LV', color='orange')
    axd['A'].set_xticks(x)
    axd['A'].set_xticklabels(asset_names)
    axd['A'].set_ylim(-0.01, 0.45)
    axd['A'].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.06),  # (x, y) coordinates
        ncol=1,  # Puts legend items side-by-side
        frameon=False  # Optional: removes the box border
    )

    axd['B'].set_title('Correlation Delta')
    axd['B'].matshow(corr_delta, vmin=0.0, vmax=0.1)
    for (i, j), z in np.ndenumerate(corr_delta):
        if z == 0:
            continue
        axd['B'].text(j, i, '{:0.3f}'.format(z), ha='center', va='center', color="white")
    axd['B'].set_xticks(range(len(asset_names)), asset_names)
    axd['B'].set_yticks(range(len(asset_names)), asset_names)


    axd['C'].set_title("Vega S1")
    axd['C'].set_xlabel("Strike")
    axd['C'].set_ylabel("Maturity")
    strikes = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    mats = np.array([0.5, 1, 1.5, 2, 2.5, ])
    axd['C'].set_yticks(range(len(mats)), labels=mats)
    axd['C'].set_xticks(range(len(strikes)), labels=strikes)
    axd['C'].imshow(vegas1, vmin=-0.14, vmax=0.10)
    for i in range(len(mats)):
        for j in range(len(strikes)):
            color = "white"
            axd['C'].text(j, i, round(vegas1[i, j], 3), ha="center", va="center", color=color)

    axd['D'].set_title("Vega S2")
    axd['D'].set_xlabel("Strike")
    axd['D'].set_ylabel("Maturity")
    axd['D'].set_yticks(range(len(mats)), labels=mats)
    axd['D'].set_xticks(range(len(strikes)), labels=strikes)
    axd['D'].imshow(vegas2, vmin=-0.14, vmax=0.10)
    for i in range(len(mats)):
        for j in range(len(strikes)):
            color = "white"
            axd['D'].text(j, i, round(vegas2[i, j], 3), ha="center", va="center", color=color)


    axd['E'].set_title("Vega S3")
    axd['E'].set_xlabel("Strike")
    axd['E'].set_ylabel("Maturity")
    axd['E'].set_yticks(range(len(mats)), labels=mats)
    axd['E'].set_xticks(range(len(strikes)), labels=strikes)
    axd['E'].imshow(vegas3, vmin=-0.14, vmax=0.10)
    for i in range(len(mats)):
        for j in range(len(strikes)):
            color = "white"
            axd['E'].text(j, i, round(vegas3[i, j], 3), ha="center", va="center", color=color)

    plt.show()


def plot_mkt_data():
    layout = [
        ['A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        ['D', 'D', 'D', 'D', 'E', 'E', 'E', 'E']
    ]

    fig, axd = plt.subplot_mosaic(layout, figsize=(12, 8), constrained_layout=True,
                                  per_subplot_kw={'C': {"projection": "3d"}, 'D': {"projection": "3d"}, 'E': {"projection": "3d"}})

    asset_names = ['S1', 'S2', 'S3']
    # Access axes using the keys defined in layout
    axd['A'].set_title('Spot Prices')
    spots = np.array([1.0, 0.9, 1.1])
    axd['A'].bar(asset_names, spots)

    axd['B'].set_title('Correlation')
    default_corr = np.array([[1.0, 0.8, 0.2],
                               [0.8, 1.0, 0.3],
                               [0.2, 0.3, 1.0]])
    axd['B'].matshow(default_corr, vmin=-1, vmax=1)
    for (i, j), z in np.ndenumerate(default_corr):
        axd['B'].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    axd['B'].set_xticks(range(len(asset_names)), asset_names)
    axd['B'].set_yticks(range(len(asset_names)), asset_names)


    axd['C'].set_title("Vola S1")
    axd['C'].set_xlabel("Strike")
    axd['C'].set_ylabel("Maturity")
    surface1 = SSVI(spots[0], 0.2, -0.7, 0.9, 0.4, phi="PowerLawFix")
    strikes = np.linspace(0.5, 2.0, 200)
    times = np.linspace(0.05, 2, 30)
    s, t = np.meshgrid(strikes, times)
    vols1 = np.empty((len(times), len(strikes)))
    for i in range(len(times)):
        vols1[i, :] = surface1.implied_vol(strikes, times[i])
    axd['C'].plot_surface(s, t, vols1)
    axd['C'].set_xlim(0.5, 2)
    axd['C'].set_ylim(0.05, 2)
    axd['C'].set_zlim(0.1, 0.7)
    axd['C'].set_box_aspect(None, zoom=1.2)
    axd['C'].view_init(azim=-55, elev=18)

    axd['D'].set_title("Vola S2")
    axd['D'].set_xlabel("Strike")
    axd['D'].set_ylabel("Maturity")
    surface2 = SSVI(spots[1], 0.25, -0.5, 0.8, 0.5, phi="PowerLawFix")
    strikes = np.linspace(0.5, 2.0, 200)
    times = np.linspace(0.05, 2, 30)
    s, t = np.meshgrid(strikes, times)
    vols2 = np.empty((len(times), len(strikes)))
    for i in range(len(times)):
        vols2[i, :] = surface2.implied_vol(strikes, times[i])
    axd['D'].plot_surface(s, t, vols2)
    axd['D'].set_xlim(0.5, 2)
    axd['D'].set_ylim(0.05, 2)
    axd['D'].set_zlim(0.1, 0.7)
    axd['D'].set_box_aspect(None, zoom=1.2)
    axd['D'].view_init(azim=-55, elev=18)

    axd['E'].set_title("Vola S3")
    axd['E'].set_xlabel("Strike")
    axd['E'].set_ylabel("Maturity")
    surface3 = SSVI(spots[2], 0.3, -0.4, 0.4, 0.6, phi="PowerLawFix")
    strikes = np.linspace(0.5, 2.0, 200)
    times = np.linspace(0.05, 2, 30)
    s, t = np.meshgrid(strikes, times)
    vols3 = np.empty((len(times), len(strikes)))
    for i in range(len(times)):
        vols3[i, :] = surface3.implied_vol(strikes, times[i])
    axd['E'].plot_surface(s, t, vols3)
    axd['E'].set_xlim(0.5, 2)
    axd['E'].set_ylim(0.05, 2)
    axd['E'].set_zlim(0.1, 0.7)
    axd['E'].set_box_aspect(None, zoom=1.2)
    axd['E'].view_init(azim=-55, elev=18)

    plt.show()


if __name__ == "__main__":
    # aad sensis
    aad_ss_delta = np.array([0.0967199855751, 0.299443740954, 0.15625835882])
    aad_d_delta = np.array([0.152329312281, 0.407222433401, 0.209725239369])
    aad_corr_delta = np.array([[0, 0.0529466467103, 0.00518724753814],
                               [0.0529466467103, 0.0, 0.0369812599145],
                               [0.00518724753814, 0.0369812599145, 0.0]])
    aad_vega1 = np.array([
                    [0.000415176347728, 0.000944530480327, 0.000826498327492, 0.0123079317407, -0.0226967275115, -0.00362109780861, -0.00403911361849, -0.00120610922521, 0.000272180874249, 1.87795540514e-05, 0, 0, 0, 0, ],
                    [-0.0036162443969, 0.00285120570371, 0.00719713848798, -0.00673450390316, -0.00669429218444, -0.0065988950486, 0.00200148721078, 0.00148407879615, 0.000654083873174, 0.000152953761075, 0, 0, 0, 0, ],
                    [-0.00984949913036, 0.00391244526693, 0.0228473916521, -0.00820168202804, -0.0478514983388, 0.00790486356959, 0.00800182352838, 0.00139809711564, -0.000229343538753, 4.63230268449e-05, 0, 0, 0, 0, ],
                    [-0.0150983616831, -0.00118011361576, 0.00074960109947, -0.072439764928, -0.0155623821613, -0.0206387777407, 0.00399857058143, 8.68488695803e-05, -2.04960421012e-05, 8.3125587485e-09, 0, 0, 0, 0, ],
                    [-8.96521087992e-08, -5.58755686784e-08, -3.73841538146e-07, -3.96653088812e-07, -1.63092622756e-07, -2.08172806644e-07, -1.14432864757e-08, -2.66268150573e-314, 0, 0, 0, 0, 0, 0, ]])
    aad_vega2 = np.array([
                    [-0.000393384378037, 0.00191866973285, 0.000435346208421, -0.00766331789281, -0.0123311154032, -0.0306347957525, 0.0103151667323, 0.000821400027909, 0.00103423576243, 0.000165221148692, -2.32987685333e-07, 0, 0, 0, ],
                    [-0.0173985372233, 0.0117956759571, 0.019547206597, -0.00962195792143, -0.0292583394538, -0.0327343273887, 0.0243657393544, -0.00250188705129, 0.00101659737243, 0.000181605325511, -9.87290333785e-06, 0, 0, 0, ],
                    [-0.0310259605732, 0.0322117704086, 0.00978325242508, -0.0173660453072, -0.0213015219413, -0.023028089011, 0.0240306630756, 0.0019309509976, 0.000325755267143, 3.9967053582e-05, -0.000140200142664, 0, 0, 0, ],
                    [-0.029414849151, -0.00107524000308, -0.0204909200316, -0.13249089449, -0.0319920253616, -0.0421099035424, 0.00685679366898, 0.000940703059731, 9.791248018e-05, -7.22549754164e-07, 5.76687304983e-07, 0, 0, 0, ],
                    [-2.33374957273e-07, -5.18505658995e-08, -9.81449078247e-07, -8.59099999755e-07, -1.98417124953e-07, -5.23287743949e-07,-3.89084400236e-08, -7.95989056191e-13, 0, 0, 0, 0, 0, 0, ],])
    aad_vega3 = np.array([
                    [-0.00150652760691, 0.00237141718035, 0.00288551318889, 0.01008105443, 0.0091372460198, -0.0521863904993, 0.00607468240222, -0.00231494315196, -0.0056423484865, 0.000110208619229, 0.00451043310882, -0.000871478546685, -0.000784079613164, 0.000210300318215, ],
                    [-0.0112294976016, 0.0103505120457, -0.00583088766588, 0.0282515242705, -0.0453611567423, -0.0142509655026, 0.0139062350344, -0.00461380522588, 0.00304834166277, -0.00312656369101, 0.00747972926596, -0.00779457749052, 0.00391106732921, -0.000143194147335, ],
                    [-0.0130395482023, 0.0111414617176, 0.00984429992339, 0.0125780379908, -0.0699398082463, -0.00554614806887, 0.0186639578374, 0.000156202471937, -0.00241357194406, 0.00407922728021, 0.000263482092819, -0.000674700688267, -0.000144417136935,0.000155039352076, ],
                    [-0.0222968983789, 0.00116649202165, -0.00980465311415, -0.104953180716, -0.0526105315267, -0.0482118780179, 0.0169177568928, 0.00174547576719, 0.000368667750731, 0.000270555875884, 0.000117983004263, 9.95854006831e-06, -9.44882926893e-06,-3.94902478498e-11, ],
                    [-1.83780395196e-07, -1.77138769801e-07, -6.08994953674e-07, -1.05093432101e-06, -4.32222710243e-07, -4.6316974151e-07,-5.84769934143e-08, -1.21150153411e-09, -3.35563263513e-154, -2.48758153406e-242, 0, 0, 0, 0, ],])

    # bump sensis
    bump_ss_delta = np.array([0.0977231129351, 0.298112118324, 0.156301083187])
    bump_d_delta = np.array([0.152356738337, 0.407535055702, 0.209863376677])
    bump_corr_delta = np.array([[0, 0.05255119229, 0.00505312749444],
                                     [0.05255119229, 0.0, 0.0370413105925],
                                     [0.00505312749444, 0.0370413105925, 0.0]])
    bump_vega1 = np.array([
                    [0.000979626557651, -0.000118759269085, -0.000177956449576, 0.0130573821533, -0.022011226708,-0.00349086296536, -0.00386334675362, -0.00137917381959, 0.000244711428898, 1.97971306015e-05, 1.0237699577e-06,1.0237699577e-06, 1.0237699577e-06, 1.0237699577e-06, ],
                    [-0.00822109891274, 0.00598102301019, 0.00742844223911, -0.00669849267165, -0.00667933528486, -0.00642501846171, 0.00197099269128, 0.00148293858304, 0.000654409071377, 0.000153757584442, 1.0237699577e-06,1.0237699577e-06, 1.0237699577e-06,1.0237699577e-06, ],
                    [-0.00985065368164, 0.00382319575998, 0.0228419956727, -0.00825547554761, -0.0476537735161, 0.00789512851762, 0.00801879946888, 0.00139883025163, -0.000228357854848, 4.73390326938e-05, 1.0237699577e-06,1.0237699577e-06, 1.0237699577e-06, 1.0237699577e-06, ],
                    [-0.0150994161285, -0.00120779432011, 0.000737928740335, -0.0724026117949, -0.015722898572, -0.0206382798851, 0.003998300524, 8.76428818408e-05, -1.94604110604e-05, 1.03207442592e-06, 1.0237699577e-06, 1.0237699577e-06,1.0237699577e-06, 1.0237699577e-06, ],
                    [9.33908506084e-07, 9.67692592724e-07, 6.50168807681e-07, 6.27153884381e-07, 8.60755910992e-07,8.15469913817e-07, 1.01227914939e-06, 1.0237699577e-06, 1.0237699577e-06, 1.0237699577e-06, 1.0237699577e-06,1.0237699577e-06, 1.0237699577e-06, 1.0237699577e-06, ],])
    bump_vega2 = np.array([
                    [-0.0042841702852, 0.00399103337045, 0.00034286079531, -0.00695595034905, -0.0123890765313,-0.0301275046799, 0.0104913071231, -0.000168093916564, 0.00108595816783, 0.000141060296777, 1.97570293459e-05,1.76838543808e-06, 1.76838543808e-06,1.76838543808e-06, ],
                    [-0.0173991557606, 0.0116366353775, 0.0195774106038, -0.00973477654043, -0.0293442172228, -0.032618773238, 0.0241820657387, -0.00250985101591, 0.00104032182868, 9.0510565709e-05, 4.43915237902e-05, 1.76838543808e-06,1.76838543808e-06, 1.76838543808e-06, ],
                    [-0.0310415001925, 0.0321383967217, 0.00974047957847, -0.0175234003419, -0.0215490383648, -0.0230444329063, 0.0243092684094, 0.00192916994601, 0.000327020643898, 4.17585632739e-05, -0.000138155964446, 1.76838543808e-06,1.76838543808e-06, 1.76838543808e-06, ],
                    [-0.0294447807581, -0.00114271162532, -0.0205858786684, -0.132691819144, -0.0320045264068, -0.042109866194, 0.00685475237683, 0.000941561872914, 9.96651094987e-05, 1.04500852416e-06, 2.34475772132e-06,1.76838543808e-06, 1.76838543808e-06,1.76838543808e-06, ],
                    [1.53508317169e-06, 1.71677116967e-06, 7.87248044531e-07, 9.09738950838e-07, 1.56996637912e-06, 1.24503740651e-06, 1.72950542776e-06, 1.76838543808e-06, 1.76838543808e-06, 1.76838543808e-06, 1.76838543808e-06,1.76838543808e-06, 1.76838543808e-06, 1.76838543808e-06, ]])
    bump_vega3 =     np.array([
                    [-0.00195304235007, 0.00158102101411, 0.00349898695573, 0.00846444124791, 0.00775189412661,-0.0504455745731, 0.00430034416965, -0.00236901350759, -0.00567917352035, -0.000250337750529, 0.00492730585222,-0.00187640453175, -0.000158739843492, 0.000121394494457, ],
                    [-0.0112382029682, 0.0103397030382, -0.00582111403524, 0.0273813661744, -0.0460222896681, -0.0134836848131, 0.0138171236896, -0.00469413180548, 0.00306592589094, -0.00317347619294, 0.00743967422112, -0.00779631887937,0.00392497208068, -0.000192367333174, ],
                    [-0.0130371848317, 0.0111577632911, 0.00987895508731, 0.0122762376376, -0.0703931119839, -0.00572650575847, 0.018562858306, 0.000401135147321, -0.00241849350546, 0.00406889014704, 0.000260144106257, -0.000683947132263,-0.000147155165831, 0.000148233614272, ],
                    [-0.0223069147154, 0.00117684380019, -0.00997965680094, -0.105190717037, -0.0528202678485, -0.0484742398998, 0.0169002990602, 0.00173663050607, 0.000363537566717, 0.000268723943098, 0.000115941800605, 7.92101939595e-06,-1.14541265361e-05,-2.00657268579e-06, ],
                    [-2.19025908521e-06, -2.18379758721e-06, -2.61530797019e-06, -3.05723224514e-06, -2.43853826021e-06, -2.46960230044e-06,-2.06494821242e-06, -2.00773841996e-06, -2.00651717464e-06, -2.00651717464e-06, -2.00651717464e-06,-2.00651717464e-06,-2.00651717464e-06, -2.00651717464e-06, ],])

    plot_mkt_data()
    plot_sensis(ss_delta=aad_ss_delta,
                d_delta=aad_d_delta,
                corr_delta=aad_corr_delta,
                vegas1=aad_vega1,
                vegas2=aad_vega2,
                vegas3=aad_vega3)
    plot_sensis(ss_delta=bump_ss_delta,
                d_delta=bump_d_delta,
                corr_delta=bump_corr_delta,
                vegas1=bump_vega1,
                vegas2=bump_vega2,
                vegas3=bump_vega3)
    # plot diff of bumping vs aad
    plot_sensis(ss_delta=aad_ss_delta-bump_ss_delta,
                d_delta=aad_d_delta-bump_d_delta,
                corr_delta=aad_corr_delta-bump_corr_delta,
                vegas1=aad_vega1-bump_vega1,
                vegas2=aad_vega2-bump_vega2,
                vegas3=aad_vega3-bump_vega3
                )