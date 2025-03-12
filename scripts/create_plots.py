import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd

from modelbasedprior.benchmarking.database import Database
from modelbasedprior.visualization.illustrative_plots import (
    sphere_plot,
    shekel_plot,
    scatter_quality_plot,
    prior_temperature_plot,
    initial_samples_plot,
)
from modelbasedprior.visualization.results_plots import (
    regret_sphere_plot,
    regret_shekel_plot,
    regret_image_similarity_plot,
    regret_scatterplot_quality_plot,
    regret_mr_layout_quality_plot,
    regret_by_technique_and_temperature,
    regret_by_technique_and_temperature_df,
    get_only_max_paths_for_colabo_df,
)

load_dotenv(dotenv_path="../.env")

db = Database(os.path.join(os.getenv('DATA_DIR'), 'experiments.db'))
plots_dir = os.getenv('PLOTS_DIR')

# Illustrative plots
for plot_func, filename in [
    # (sphere_plot, 'sphere.png'),
    # (shekel_plot, 'shekel.png'),
    # (scatter_quality_plot, 'scatter_quality.png'),
    # (prior_temperature_plot, 'prior_temperature.png'),
    # (initial_samples_plot, 'initial_samples.png'),
]:
    fig, ax = plot_func()
    fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    fig.tight_layout()

# Results plots
for plot_func, filename in [
    # (regret_sphere_plot, 'regret_sphere.png'),
    (regret_shekel_plot, 'regret_shekel.png'),
    # (regret_image_similarity_plot, 'regret_image_similarity.png'),
    # (regret_scatterplot_quality_plot, 'regret_scatterplot_quality.png'),
    # (regret_mr_layout_quality_plot, 'regret_mr_layout_quality.png'),
]:
    fig, ax = plot_func(db)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=300)

for optimization_type, objective_type in [
    # ('BO', 'Sphere'),
    # ('BO', 'SphereNoisy'),
    # ('PBO', 'Sphere'),
    ('BO', 'Shekel'),
    # ('BO', 'ImageSimilarity'),
    # ('PBO', 'ImageSimilarity'),
    # ('BO', 'ScatterPlotQuality'),
    # ('BO', 'MRLayoutQuality'),
]:
    df = regret_by_technique_and_temperature_df(db, optimization_type, objective_type)
    if objective_type == 'Shekel':
        df_colabo = get_only_max_paths_for_colabo_df(db, optimization_type, objective_type)
        # Drop all ColaBO rows
        df = df[~df['label'].str.contains('ColaBO', na=False)]
        df = pd.concat([df, df_colabo])
    fig, axs = regret_by_technique_and_temperature(df, optimization_type, objective_type)
    fig.tight_layout()
    fig.savefig(os.path.join(
        plots_dir,
        f'regret_{optimization_type}_{objective_type}.png'
    ), dpi=300)

plt.show()