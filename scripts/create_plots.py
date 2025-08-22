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
    regret_image_tuning_plot,
)

load_dotenv(dotenv_path="../.env")

db = Database(os.path.join(os.getenv('DATA_DIR'), 'experiments.db'))
plots_dir = os.getenv('PLOTS_DIR')

# Illustrative plots
for plot_func, filename in [
    # (sphere_plot, 'sphere.png'),
    # (shekel_plot, 'shekel.png'),
    # (scatter_quality_plot, 'scatter_quality.png'),
    (prior_temperature_plot, 'prior_temperature.png'),
    # (initial_samples_plot, 'initial_samples.png'),
]:
    fig, ax = plot_func()
    fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    fig.tight_layout()

# Results plots
for plot_func, filename in [
    # (regret_sphere_plot, 'regret_sphere.png'),
    # (regret_shekel_plot, 'regret_shekel.png'),
    # (regret_image_similarity_plot, 'regret_image_similarity.png'),
    # (regret_scatterplot_quality_plot, 'regret_scatterplot_quality.png'),
    # (regret_mr_layout_quality_plot, 'regret_mr_layout_quality.png'),
]:
    fig, ax = plot_func(db)
    if 'shekel' in filename:
        ax.set_title(None)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=300)

    if 'sphere' in filename:
        for o_type in [
            ('BO', 'Sphere', 'Sphere BO'),
            ('BO', 'SphereNoisy', 'Sphere BO (Noisy)'),
            ('PBO', 'Sphere', 'Sphere PBO'),
        ]:
            fig, ax = plot_func(db, optimization_and_objective_types=[o_type], ax_size=(4.5,3.5))
            for a in ax:
                a.set_title(None)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, o_type[1] + o_type[0] + '_' + filename), dpi=300)

for optimization_type, objective_type in [
    # ('BO', 'Sphere'),
    # ('BO', 'SphereNoisy'),
    # ('PBO', 'Sphere'),
    # ('BO', 'Shekel'),
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

image_tuning_results_dir = os.getenv("IMAGE_TUNING_SAVE_DIR", "./image_tuning_results")
for task in [
    # "Aesthetics",
    # "Reference"
]:
    fig_img, ax_img = regret_image_tuning_plot(
        base_save_dir=image_tuning_results_dir,
        task=task,
        optimal_value=10.0, # Max rating
        # num_initial_samples=4, # Can often be inferred
        # max_iterations_to_plot=30 # Set if needed
    )
    ax_img.set_title(f"Image Tuning Regret ({task})")
    fig_img.tight_layout()
    fig_img.savefig(os.path.join(
        plots_dir,
        f'regret_pilot_{task}.png'
    ), dpi=300)

plt.show()