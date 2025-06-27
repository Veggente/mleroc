import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import config
from mleroc.unbalance import BHTSim
from mleroc.unbalance import Gamma
from mleroc.unbalance import structure_estimator


def tail_estimation_plots(bht_sim: BHTSim):
    """Generates tail estimation plots for the various estimators of ROC
    curves for the binormal BHT problem.

    Used in IT Trans paper draft May 2025 "Maximum Likelihood Estimation of
    Optimal Receiver Operating Characteristic Curves From Likelihood Ratio
    Observations".
    """
    plt.style.use("ggplot")
    # Enable LaTeX rendering (optional, requires LaTeX installation).
    plt.rcParams["text.usetex"] = True

    for alpha in config.config["tail-levy"]["alpha"]:
        print(f"alpha={alpha}")
        # Number of likelihood ratio samples used by each ROC estimator
        # variate; 100 is good.
        for n_samples in config.config["tail-levy"]["n_samples"][
            bht_sim._distribution.name()
        ]:
            print(f"n_samples={n_samples}")
            # Number of Levy distance samples used for each curve. 1000 is good,
            # 10000 is better could take an hour per figure.
            n_levy_samples = config.config["tail-levy"]["n_levy_samples"]
            data = np.zeros([n_levy_samples, 6])
            rng = np.random.default_rng(42)
            for i in tqdm(range(n_levy_samples)):
                data[i, :] = n_samples * np.square(
                    bht_sim.levy_single(
                        [int(alpha * n_samples), n_samples - int(alpha * n_samples)],
                        rng=rng,
                    )
                )
            sorted_data = np.sort(data, axis=0)
            cdf_values = np.log(
                1 / 2 - np.arange(n_levy_samples) / (2 * n_levy_samples)
            )

            plt.figure()
            plt.rcParams.update({"font.size": 16})
            # Cuts off bottom part of y range where there is little data.
            plt.ylim([-np.log(n_levy_samples / 2) + 2, 0])
            for est in [
                structure_estimator(est)
                for est in config.config["tail-levy"]["estimators"]
            ]:
                plt.plot(
                    sorted_data[:, est.index],
                    cdf_values,
                    est.style,
                    label=est.name,
                    color=f"C{est.color}",
                    markevery=0.1,
                )

            plt.xlabel(r"$\gamma$")
            plt.ylabel(r"$\psi(\gamma)$")
            plt.grid(True)
            plt.legend(loc="best")
            plt.savefig(
                config.config["paths"]["plots_dir"]
                + "tail_estimation-num_sim"
                + str(n_samples)
                + "-num_L_samp"
                + str(n_levy_samples)
                + "-alpha"
                + str(alpha)
                + f"-{bht_sim._distribution.name()}"
                + ".pdf",
                bbox_inches="tight",
            )


if __name__ == "__main__":
    tail_estimation_plots(BHTSim())
    tail_estimation_plots(BHTSim(distribution=Gamma()))
