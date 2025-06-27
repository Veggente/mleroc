![mleroc](/logo/mleroc-72ppi.png)
# mleroc
mleroc (pronounced melee rock) is an ML estimator of optimal ROC curves.  This
project is associated with the paper "Maximum likelihood estimation of optimal
receiver operating characteristic curves from likelihood ratio observations"
available at https://arxiv.org/abs/2202.01956.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
```

Install the package:

```bash
pip install -e ".[dev]"
```

## Configuration

The project uses a TOML-based configuration system that automatically sets up
your personal configuration file on first use.

### Automatic Setup

When you first import the configuration, it will automatically:
1. Look for a config file in `~/.config/causal_recovery/config.toml`.
2. If not found, copy the template from the package to create your personal
   config.
3. Set appropriate file permissions (600) for security.

### Using Configuration in Code

```python
from config import config

# Access configuration values.
plots_dir = config.config["paths"]["plots_dir"]
data_dir = config.config["paths"]["data_dir"]
n_sims = config.config["average-levy"]["n_sims"]

# The config object provides direct access to the TOML data.
print(f"Plots will be saved to: {plots_dir}")
```

### Customizing Configuration

1. **Edit your config file**:
   ```bash
   # Open in your preferred editor.
   nano ~/.config/causal_recovery/config.toml
   # or
   code ~/.config/causal_recovery/config.toml
   ```

2. **Modify paths**: Change the `plots_dir` and `data_dir` to your preferred
   locations.

3. **Adjust experiment parameters**: Modify `n_sims` and sample sizes as
   needed.

4. **Customize estimators**: Add or modify estimator configurations.

## Usage

Use the following scripts to reproduce the figures in the paper ["Maximum Likelihood Estimation of Optimal Receiver Operating Characteristic Curves From Likelihood Ratio Observations" by Hajek and Kang](https://arxiv.org/abs/2202.01956).

- `c_values_for_estimators.py`:
  - Fig. 2. c vs. alpha in upper bounds
- `unbalance.py`:
  - Fig. 3–5, mean Levy distances of estimators from true ROC
- `tail_estimation.py`:
  - Fig. 6.  Estimates of psi_n(gamma) vs. gamma for binormal gaussian
- `tail_estimation2.py`:
  - Fig. 7.  Estimates of psi_n(gamma) vs. gamma for H0: exponential distribution
