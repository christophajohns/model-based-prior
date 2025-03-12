# Bayesian Optimization with Model-based Priors

> A Python 3.12 project to guide sample acquisition with models of user performance or preference.

This project provides a Python implementation of a prior injection technique for Bayesian optimization that uses models of user performance or preference to guide sample acquisition.

## Installation

To install the project, follow these steps:

1. Clone the repository: `git clone christophajohns/model-based-prior`
2. Navigate to the project directory: `cd model-based-prior`
3. Create a virtual environment from `environment.yml`: `conda env create -f environment.yml`
4. Activate the virtual environment: `conda activate model-based-prior`
5. Install the local package: `pip install -e .`

You can also install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

And then install the local package:

```bash
pip install -e .
```

## Usage

To try out the prior-guided Bayesian optimization, run the following command:

```bash
python main.py
```

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
