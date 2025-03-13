# Bayesian Optimization with Model-based Priors

> A Python 3.12 project to guide sample acquisition with models of user performance or preference.

This project provides a Python implementation of a prior injection technique for Bayesian optimization that uses models of user performance or preference to guide sample acquisition.

## Installation

To install the project, follow these steps:

1. Clone the repository (incl. the `colabo` submodule): `git clone --recursive https://github.com/christophajohns/model-based-prior.git`
2. Navigate to the project directory: `cd model-based-prior`
3. Create a virtual environment from `environment.yml`: `conda env create -f environment.yml`
4. Activate the virtual environment: `conda activate model-based-prior`
5. Install the local package: `pip install -e .`
6. Copy the `.env.example` file into a new file `.env`
7. Set the variables in `.env`

You can also install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

And then install the local package:

```bash
pip install -e .
```

If `colabo` receives updates, pull them using:

```bash
git submodule update --remote
```

## Usage

To try out the prior-guided Bayesian optimization, run the following command:

```bash
python main.py
```

## Replication

You can replicate the experiments and figures used in our paper.

### Run the experiments

To replicate the experiments from our paper, run the following command:

```bash
python scripts/synthetic_evaluation.py
```

Alternatively, to brute force run the script, use:

```bash
. ./scripts/run_experiment.sh
```

This will create and populate an SQLite database with the optimization results at `DATA_DIR/experiments.db`.

**Note:** You will need to download the an image from the [AVA dataset](https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment) (ID: 43405) and make it available at `AVA_FLOWERS_DIR/43405.jpg` to use the `ImageSimilarity` objective. Since this is image is protected by copyright, we do not include it here.

**Note:** Due to the way that randomness works in BoTorch, the results can differ from those reported in the paper.

**Note:** You may want to modify `scripts/synthetic_evaluation.py` to change which experiments are run and how.

**Note:** The `scripts/synthetic_evaluation.py` script has a variable `SMOKE_TEST` that can be set to `SMOKE_TEST=TRUE` to run a smaller test of the script's functionality and quickly catch issues.

### Create the plots

To create the plots used in the paper, you can run:

```bash
python scripts/create_plots.py
```

This will pull data from the previously created database and visualize it using matplotlib.

## Contributing

Contributions to the project are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request on the GitHub repository.

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
