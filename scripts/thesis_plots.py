import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from modelbasedprior.benchmarking.database import Database
from modelbasedprior.visualization.illustrative_plots import (
    prior_temperature_plot,
    pibo_acquisition_plot,
    colabo_acquisition_plot,
    sphere_plot,
    shekel_plot,
    scatter_quality_plot,
    pibo_normalization_plot,
    colabo_robustness_plot,
)

load_dotenv(dotenv_path="../.env")

db = Database(os.path.join(os.getenv('DATA_DIR'), 'experiments.db'))
plots_dir = os.getenv('PLOTS_DIR')

plot_func = colabo_acquisition_plot
filename = 'colabo_acquisition.pdf'

fig, ax = plot_func()
fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
fig.tight_layout()

# plt.show()