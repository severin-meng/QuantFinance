import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


model_abbrev = {"Dupire": "LV", "Dupire-Multi": "LVM"}
variance_params = {"path", "smooth", "bump"}
bias_params = {"steps"}


def get_df(model, product, param, K, lvgrid="equidist", disc="euler", ipol="linear", vol="ssvi"):
    model_short = model_abbrev.get(model)
    dir = rf"C:\Quantitative Finance\Scrambling_PathConstruction\convBandData\{model}\{product}"
    if model == "Dupire":
        filename = rf"convergence_{param}_{model_short}-{lvgrid}-{disc}-{ipol}-{vol}_smpls-{K}.csv"
    elif model == "Dupire-Multi":
        filename = rf"convergence_{param}_{model_short}-{lvgrid}-{disc}-{ipol}-{vol}_smpls-{K}.csv"
    fullpath = os.path.join(dir, filename)
    df = pd.read_csv(fullpath)
    return df


def plot_single(model, prod, K, param, measure, std_devs=2, **mdlParams):
    """
    mdlParams are lvgrid="equidist", disc="euler", ipol="linear", vol="ssvi"
    """
    df = get_df(model, prod, param.lower(), K, **mdlParams)

    # measure = "Sticky-Strike Gamma"
    # measure = "Value"
    # measure = "Sticky-Strike Delta"
    # measure = "Vega_3_4"
    # measure = "Dupire Delta"
    # measure = "Vega_2_7"

    plt.figure()
    plt.title(f"{prod}: {measure} vs {param} paramter")
    plt.xlabel(param)
    plt.ylabel(measure)
    # Value, Sticky-Strike Delta, Vega_3_4
    rqmc_mask = df["Type"].str.startswith("RQMC_")

    rqmc_std = df[rqmc_mask].groupby(["Config", "Measure", param.capitalize()])["Result"].std().reset_index(name="Std")
    rqmc_mean = df[rqmc_mask].groupby(["Config", "Measure", param.capitalize()])["Result"].mean().reset_index(name="Avg")
    rqmc_stat = pd.merge(rqmc_std, rqmc_mean, on=["Config", "Measure", param.capitalize()]).sort_values(by=[param.capitalize()], ascending=False)

    rqmc_stat = rqmc_stat[rqmc_stat["Measure"] == measure]
    x_vals = rqmc_stat[param.capitalize()]
    if (param.lower() == "paths"):
        x_vals *= K
    if param in variance_params:
        plt.semilogx(x_vals, rqmc_stat["Avg"], label="RQMC Avg")
        plt.semilogx(x_vals, rqmc_stat["Avg"] + std_devs*rqmc_stat["Std"]/np.sqrt(K), color="blue", alpha=0.4)
        plt.semilogx(x_vals, rqmc_stat["Avg"] - std_devs*rqmc_stat["Std"]/np.sqrt(K), color="blue", alpha=0.4)
    else:
        plt.plot(x_vals, rqmc_stat["Avg"], label="RQMC Avg")
        plt.plot(x_vals, rqmc_stat["Avg"] + std_devs*rqmc_stat["Std"]/np.sqrt(K), color="blue", alpha=0.4)
        plt.plot(x_vals, rqmc_stat["Avg"] - std_devs*rqmc_stat["Std"]/np.sqrt(K), color="blue", alpha=0.4)
    # Shade ±1 std
    plt.fill_between(x_vals, rqmc_stat["Avg"] - std_devs*rqmc_stat["Std"]/np.sqrt(K), rqmc_stat["Avg"] + std_devs*rqmc_stat["Std"]/np.sqrt(K), color="blue", alpha=0.2, label=f"±{std_devs} std")

    qmc_vals = df[(df["Type"] == "QMC") & (df["Measure"] == measure)].sort_values(by=[param.capitalize()], ascending=False)
    x_vals = qmc_vals[param.capitalize()]
    if param in variance_params:
        plt.semilogx(x_vals, qmc_vals["Result"], label="QMC")
    else:
        plt.plot(x_vals, qmc_vals["Result"], label="QMC")
    plt.scatter(x_vals, qmc_vals["Result"])
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    model = "Dupire"
    K = 32
    product = "Accumulator"  # product = Digital
    # plot_vs_smoothing(model, product, K)
    # plot_vs_paths(model, product, K)
    plot_single(model, product, K, "steps", "Vanna_2_7")
    # plot_vs_bump(model, product, K)
