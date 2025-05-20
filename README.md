![mleroc](/logo/mleroc-72ppi.png)
# mleroc
mleroc (pronounced melee rock) is a maximum likelihood (ML) estimator of
optimal ROC curves.

## Installation
Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

## Usage
Use the following command to reproduce the figures in the paper ["Maximum
Likelihood Estimation of Optimal Receiver Operating Characteristic Curves From
Likelihood Ratio Observations" by Hajek and
Kang](https://arxiv.org/abs/2202.01956).
```bash
unbalance
```
This is equivalent to running
```bash
python -c "from mleroc.unbalance import main; main()"
```
