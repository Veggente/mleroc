import matplotlib.pyplot as plt
import numpy as np

from config import config

plt.style.use("ggplot")

# Enable LaTeX rendering (optional, requires LaTeX installation).
plt.rcParams["text.usetex"] = True

line_width = 3


alpha = np.linspace(0, 1, 100)
y1 = 2 * np.minimum(alpha, 1 - alpha)
y2 = 2 * np.maximum(alpha, 1 - alpha) ** 2
y3 = 0.5 * np.ones_like(alpha)

plt.rcParams.update({"font.size": 16})

plt.plot(alpha, y1, "--", linewidth=line_width, label="empirical")
plt.plot(alpha, y2, ":", linewidth=line_width, label="split")
plt.plot(alpha, y3, "-", linewidth=line_width, label="fused")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$c$")
plt.legend()

# Adjust layout to prevent cropping.
plt.tight_layout()

plt.savefig(
    config.config["paths"]["plots_dir"] + "c_values_for_estimators.png",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
