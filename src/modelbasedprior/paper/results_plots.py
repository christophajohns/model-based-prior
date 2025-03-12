"""Plotting functions for the paper."""

from typing import Tuple, List, Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from modelbasedprior.benchmarking.database import Database
from modelbasedprior.objectives.sphere import Sphere

load_dotenv()

COLORS = [
    "blue",
    "red",
    "gold",
    "green",
    "purple",
    "orange",
    "brown",
    "pink",
    "gray",
    "cyan",
    "magenta",
    "lime",
    "teal",
    "lavender",
    "maroon",
    "navy",
    "olive",
    "silver",
    "black",
]

def plot_regret(ax: plt.Axes, df: pd.DataFrame, num_initial_samples: int = 4, shaded_area: Literal['min_max', 'iqr'] | None = 'iqr') -> None:
    """Plot the regret for each bias level."""
    df = df.sort_values(['label', 'iteration'])

    max_iterations = df['iteration'].max()

    # Plot the regret for each bias level
    for label_idx, label in enumerate(df['label'].unique()):
        color = COLORS[label_idx]

        df_trace = df[df['label'] == label]
        ax.plot(df_trace['iteration'] - 1, df_trace['median_log10_regret'], label=label, color=color, alpha=0.9)

        if shaded_area == 'min_max':
            # Add the min and max regret as a shaded area
            ax.fill_between(df_trace['iteration'] - 1, df_trace['min_log10_regret'], df_trace['max_log10_regret'],
                            color=color, alpha=0.2)
            
        elif shaded_area == 'iqr':
            # Alternatively, plot the IQR as a shaded area
            ax.fill_between(df_trace['iteration'] - 1, df_trace['first_quartile_log10_regret'], df_trace['third_quartile_log10_regret'],
                            color=color, alpha=0.2)

    # Add vertical lines for the initial samples
    ax.axvline(x=num_initial_samples - 0.5, color='darkgrey', linestyle='--', alpha=0.5)

    # Label the axes
    ax.set_xlabel(r'$\text{Iteration } i$')
    ax.set_ylabel(r'$\text{Log10 Regret }\log_{10}(f(\hat{x}_i) - f(x^*))$')

    # Remove the box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set the x-axis ticks
    ax.set_xticks(np.arange(0, max_iterations, 5))

    # Limit the number of y-axis ticks to 5
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Add a grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add a legend
    ax.legend(loc='upper right')

def get_regret_data(
        db: Database,
        optimization_type: str,
        objective_type: str,
        prior_types_injection_method_and_descriptions: List[Tuple[str, str, str]] = [
            ('None', 'None', 'Uniform prior'),
            ('Unbiased', 'ColaBO', 'ColaBO, Unbiased'),
            ('UnbiasedCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.1$'),
            ('UnbiasedMoreCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.01$'),
            ('UnbiasedUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 10.0$'),
            ('UnbiasedMoreUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 100.0$'),
            ('Biased', 'ColaBO', r'ColaBO, Biased, $T = 1.0$'),
            ('BiasedCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.1$'),
            ('BiasedMoreCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.01$'),
            ('BiasedUncertain', 'ColaBO', r'ColaBO, Biased, $T = 10.0$'),
            ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, Biased, $T = 100.0$'),
            ('Unbiased', 'piBO', r'$\pi$BO, Unbiased'),
            ('UnbiasedCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.1$'),
            ('UnbiasedMoreCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.01$'),
            ('UnbiasedUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 10.0$'),
            ('UnbiasedMoreUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 100.0$'),
            ('Biased', 'piBO', r'$\pi$BO, Biased, $T = 1.0$'),
            ('BiasedCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.1$'),
            ('BiasedMoreCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.01$'),
            ('BiasedUncertain', 'piBO', r'$\pi$BO, Biased, $T = 10.0$'),
            ('BiasedMoreUncertain', 'piBO', r'$\pi$BO, Biased, $T = 100.0$'),
        ],
        **kwargs
    ) -> pd.DataFrame:
    """Get the regret data for the given configuration."""
    data = []
    for prior_type, prior_injection_method, description in prior_types_injection_method_and_descriptions:
        experiment_configs = db.get_experiment_configs(
            optimization_type=optimization_type,
            objective_type=objective_type,
            prior_type=prior_type,
            prior_injection_method=prior_injection_method,
            **kwargs,
        )
        if len(experiment_configs) == 0:
            continue
        experiment_results = [
            db.get_bo_experiment_results(experiment.id) if optimization_type in ['BO', 'PriorSampling']
            else db.get_pbo_experiment_results(experiment.id)
            for experiment in experiment_configs
        ]
        max_iterations = max([len(result) for result in experiment_results])
        optimal_values = [db.get_objective(config.objective_id).optimal_value for config in experiment_configs]

        # Loop over the iterations and calculate the regret
        for iteration in range(1, max_iterations + 1):
            regrets = []
            for optimal_value, result in zip(optimal_values, experiment_results):
                if iteration <= len(result):
                    regrets.append(np.log10(max(1e-10, optimal_value - result[iteration - 1].best_rating)))
            if regrets:
                data.append({
                    'iteration': iteration,
                    'min_log10_regret': min(regrets),
                    'first_quartile_log10_regret': np.percentile(regrets, 25),
                    'median_log10_regret': np.median(regrets),
                    'third_quartile_log10_regret': np.percentile(regrets, 75),
                    'max_log10_regret': max(regrets),
                    'label': description,
                })

    return pd.DataFrame(data)

def regret_sphere_plot(db: Database) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret sphere plot."""
    prior_types_injection_method_and_descriptions = [
        ('None', 'None', 'MC-LogEI'),
        ('Biased', 'ColaBO', 'ColaBO-MC-LogEI'),
        ('Biased', 'piBO', r'$\pi$BO-MC-LogEI'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'Prior Sampling'),
    ]
    # prior_types_injection_method_and_descriptions = [
    #     ('None', 'None', 'Uniform prior'),
    #     # ('BiasedMoreCertain', 'ColaBO', r'ColaBO, $\Delta = (-0.5,-0.5)^T, T = 0.01$'),
    #     # ('BiasedCertain', 'ColaBO', r'ColaBO, $\Delta = (-0.5,-0.5)^T, T = 0.1$'),
    #     ('Biased', 'ColaBO', r'ColaBO, $\Delta = (-0.5,-0.5)^T, T = 1.0$'),
    #     # ('BiasedUncertain', 'ColaBO', r'ColaBO, $\Delta = (-0.5,-0.5)^T, T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, $\Delta = (-0.5,-0.5)^T, T = 100.0$'),
    #     ('Biased', 'piBO', r'$\pi$BO, Biased, $\Delta = (-0.5,-0.5)^T, T = 1.0$'),
    #     # ('BiasedCertain', 'piBO', r'$\pi$BO, Biased, $\Delta = (-0.5,-0.5)^T, T = 0.1$'),
    #     # ('BiasedMoreCertain', 'piBO', r'$\pi$BO, Biased, $\Delta = (-0.5,-0.5)^T, T = 0.01$'),
    #     # ('BiasedUncertain', 'piBO', r'$\pi$BO, Biased, $\Delta = (-0.5,-0.5)^T, T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'piBO', r'$\pi$BO, Biased, $\Delta = (-0.5,-0.5)^T, T = 100.0$'),
    # ]
    # prior_sampling_injection_methods_and_descriptions = [
    #     ('Biased', 'None', r'Prior Sampling, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'None', r'Prior Sampling, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'None', r'Prior Sampling, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'None', r'Prior Sampling, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'None', r'Prior Sampling, Biased, $T = 100.0$'),
    # ]

    dfs = []
    for optimization_type, objective_type in [
        ('BO', 'Sphere'),
        ('BO', 'SphereNoisy'),
        ('PBO', 'Sphere'),
    ]:
        df_bayesian = get_regret_data(
            db=db,
            optimization_type=optimization_type,
            objective_type=objective_type,
            prior_types_injection_method_and_descriptions=prior_types_injection_method_and_descriptions,
        )
        df_prior_sampling = get_regret_data(
            db=db,
            optimization_type='PriorSampling',
            objective_type=objective_type,
            prior_types_injection_method_and_descriptions=prior_sampling_injection_methods_and_descriptions,
        )
        df = pd.concat([df_bayesian, df_prior_sampling])
        dfs.append(df)

    # Drop all iterations after 40
    max_iterations_to_plot = 40
    num_initial_samples = 4
    for i, df in enumerate(dfs):
        dfs[i] = df[df['iteration'] <= (max_iterations_to_plot + num_initial_samples)]

    # Create a figure
    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4), sharex=True)

    # Plot the regret for each bias level
    for ax, df, title in zip(axes, dfs, ['Sphere BO', 'Sphere BO (Noisy)', 'Sphere PBO']):
        plot_regret(ax, df, num_initial_samples)
        ax.set_title(title)

    # Remove the y-axis label from all plots except the first one
    for ax in axes[1:]:
        ax.set_ylabel('')

    return fig, axes

def regret_shekel_plot(db: Database) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret shekel plot."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    prior_types_injection_method_and_descriptions = [
        ('None', 'None', 'MC-LogEI'),
        ('Biased', 'piBO', r'$\pi$BO-MC-LogEI'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'Prior Sampling'),
    ]
    # prior_types_injection_method_and_descriptions = [
    #     ('None', 'None', 'Uniform prior'),
    #     # ('Unbiased', 'ColaBO', 'ColaBO, Unbiased'),
    #     # ('UnbiasedCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 100.0$'),
    #     ('Biased', 'ColaBO', r'ColaBO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'ColaBO', r'ColaBO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, Biased, $T = 100.0$'),
    #     # ('Unbiased', 'piBO', r'$\pi$BO, Unbiased'),
    #     # ('UnbiasedCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 100.0$'),
    #     ('Biased', 'piBO', r'$\pi$BO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'piBO', r'$\pi$BO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'piBO', r'$\pi$BO, Biased, $T = 100.0$'),
    # ]
    # prior_sampling_injection_methods_and_descriptions = [
    #     # ('Unbiased', 'None', 'Prior Sampling, Unbiased'),
    #     # ('UnbiasedCertain', 'None', r'Prior Sampling, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'None', r'PriorSampling, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'None', r'PriorSampling, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'None', r'PriorSampling, Unbiased, $T = 100.0$'),
    #     ('Biased', 'None', r'Prior Sampling, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'None', r'Prior Sampling, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'None', r'Prior Sampling, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'None', r'Prior Sampling, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'None', r'Prior Sampling, Biased, $T = 100.0$'),
    # ]

    df_bo_colabo = get_regret_data(
        db=db,
        optimization_type='BO',
        objective_type='Shekel',
        prior_types_injection_method_and_descriptions=[
            ('Biased', 'ColaBO', 'ColaBO-MC-LogEI'),
        ],
        num_paths=65536,
    )
    df_bo = get_regret_data(
        db=db,
        optimization_type='BO',
        objective_type='Shekel',
        prior_types_injection_method_and_descriptions=prior_types_injection_method_and_descriptions,
    )
    df_prior_sampling = get_regret_data(
        db=db,
        optimization_type='PriorSampling',
        objective_type='Shekel',
        prior_types_injection_method_and_descriptions=prior_sampling_injection_methods_and_descriptions,
    )
    df = pd.concat([df_bo, df_bo_colabo, df_prior_sampling])

    # Plot the regret for each bias level
    num_initial_samples = 4
    plot_regret(ax, df, num_initial_samples)
    ax.set_title('Shekel BO')

    return fig, ax

def regret_image_similarity_plot(db: Database) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret image similarity plot."""
    num_initial_samples = 4
    optimization_types = ['BO', 'PBO']
    prior_types_injection_method_and_descriptions = [
        ('None', 'None', 'MC-LogEI'),
        ('Biased', 'ColaBO', 'ColaBO-MC-LogEI'),
        ('Biased', 'piBO', r'$\pi$BO-MC-LogEI'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'Prior Sampling'),
    ]
    # prior_types_injection_method_and_descriptions = [
    #     ('None', 'None', 'Uniform prior'),
    #     # ('Unbiased', 'ColaBO', 'ColaBO, Unbiased'),
    #     # ('UnbiasedCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 100.0$'),
    #     ('Biased', 'ColaBO', r'ColaBO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'ColaBO', r'ColaBO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, Biased, $T = 100.0$'),
    #     # ('Unbiased', 'piBO', r'$\pi$BO, Unbiased'),
    #     # ('UnbiasedCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 100.0$'),
    #     ('Biased', 'piBO', r'$\pi$BO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'piBO', r'$\pi$BO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'piBO', r'$\pi$BO, Biased, $T = 100.0$'),
    # ]
    # prior_sampling_injection_methods_and_descriptions = [
    #     # ('Unbiased', 'None', 'Prior Sampling, Unbiased'),
    #     # ('UnbiasedCertain', 'None', r'Prior Sampling, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'None', r'PriorSampling, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'None', r'PriorSampling, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'None', r'PriorSampling, Unbiased, $T = 100.0$'),
    #     ('Biased', 'None', r'Prior Sampling, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'None', r'Prior Sampling, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'None', r'Prior Sampling, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'None', r'Prior Sampling, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'None', r'Prior Sampling, Biased, $T = 100.0$'),
    # ]
    fig, ax = plt.subplots(1, len(optimization_types), figsize=(5 * len(optimization_types), 4))

    for i, optimization_type in enumerate(optimization_types):
        df_bo = get_regret_data(
            db=db,
            optimization_type=optimization_type,
            objective_type='ImageSimilarity',
            prior_types_injection_method_and_descriptions=prior_types_injection_method_and_descriptions,
        )
        df_prior_sampling = get_regret_data(
            db=db,
            optimization_type='PriorSampling',
            objective_type='ImageSimilarity',
            prior_types_injection_method_and_descriptions=prior_sampling_injection_methods_and_descriptions,
        )
        df = pd.concat([df_bo, df_prior_sampling])

        # Plot the regret for each bias level
        plot_regret(ax[i], df, num_initial_samples)
        ax[i].set_title(f'Image Similarity {optimization_type}')

    return fig, ax

def regret_scatterplot_quality_plot(db: Database) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret scatterplot quality plot."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    prior_types_injection_method_and_descriptions = [
        ('None', 'None', 'MC-LogEI'),
        ('Biased', 'ColaBO', 'ColaBO-MC-LogEI'),
        ('Biased', 'piBO', r'$\pi$BO-MC-LogEI'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'Prior Sampling'),
    ]
    # prior_types_injection_method_and_descriptions = [
    #     ('None', 'None', 'Uniform prior'),
    #     ('Biased', 'ColaBO', r'ColaBO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'ColaBO', r'ColaBO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, Biased, $T = 100.0$'),
    #     ('Biased', 'piBO', r'$\pi$BO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'piBO', r'$\pi$BO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'piBO', r'$\pi$BO, Biased, $T = 100.0$'),
    # ]
    # prior_sampling_injection_methods_and_descriptions = [
    #     ('Biased', 'None', r'Prior Sampling, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'None', r'Prior Sampling, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'None', r'Prior Sampling, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'None', r'Prior Sampling, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'None', r'Prior Sampling, Biased, $T = 100.0$'),
    # ]

    df_bo = get_regret_data(
        db=db,
        optimization_type='BO',
        objective_type='ScatterPlotQuality',
        prior_types_injection_method_and_descriptions=prior_types_injection_method_and_descriptions,
    )
    df_prior_sampling = get_regret_data(
        db=db,
        optimization_type='PriorSampling',
        objective_type='ScatterPlotQuality',
        prior_types_injection_method_and_descriptions=prior_sampling_injection_methods_and_descriptions,
    )
    df = pd.concat([df_bo, df_prior_sampling])

    # Plot the regret for each bias level
    num_initial_samples = 4
    plot_regret(ax, df, num_initial_samples)
    ax.set_title('Scatter Plot Quality BO')

    return fig, ax

def regret_mr_layout_quality_plot(db: Database) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret MR layout quality plot."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    prior_types_injection_method_and_descriptions = [
        ('None', 'None', 'MC-LogEI'),
        ('Biased', 'ColaBO', 'ColaBO-MC-LogEI'),
        ('Biased', 'piBO', r'$\pi$BO-MC-LogEI'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'Prior Sampling'),
    ]
    # prior_types_injection_method_and_descriptions = [
    #     ('None', 'None', 'Uniform prior'),
    #     # ('Unbiased', 'ColaBO', 'ColaBO, Unbiased'),
    #     # ('UnbiasedCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'ColaBO', r'ColaBO, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'ColaBO', r'ColaBO, Unbiased, $T = 100.0$'),
    #     ('Biased', 'ColaBO', r'ColaBO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'ColaBO', r'ColaBO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, Biased, $T = 100.0$'),
    #     # ('Unbiased', 'piBO', r'$\pi$BO, Unbiased'),
    #     # ('UnbiasedCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'piBO', r'$\pi$BO, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'piBO', r'$\pi$BO, Unbiased, $T = 100.0$'),
    #     ('Biased', 'piBO', r'$\pi$BO, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'piBO', r'$\pi$BO, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'piBO', r'$\pi$BO, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'piBO', r'$\pi$BO, Biased, $T = 100.0$'),
    # ]
    # prior_sampling_injection_methods_and_descriptions = [
    #     # ('Unbiased', 'None', 'Prior Sampling, Unbiased'),
    #     # ('UnbiasedCertain', 'None', r'Prior Sampling, Unbiased, $T = 0.1$'),
    #     # ('UnbiasedMoreCertain', 'None', r'PriorSampling, Unbiased, $T = 0.01$'),
    #     # ('UnbiasedUncertain', 'None', r'PriorSampling, Unbiased, $T = 10.0$'),
    #     # ('UnbiasedMoreUncertain', 'None', r'PriorSampling, Unbiased, $T = 100.0$'),
    #     ('Biased', 'None', r'Prior Sampling, Biased, $T = 1.0$'),
    #     # ('BiasedCertain', 'None', r'Prior Sampling, Biased, $T = 0.1$'),
    #     # ('BiasedMoreCertain', 'None', r'Prior Sampling, Biased, $T = 0.01$'),
    #     # ('BiasedUncertain', 'None', r'Prior Sampling, Biased, $T = 10.0$'),
    #     # ('BiasedMoreUncertain', 'None', r'Prior Sampling, Biased, $T = 100.0$'),
    # ]

    df_bo = get_regret_data(
        db=db,
        optimization_type='BO',
        objective_type='MRLayoutQuality',
        prior_types_injection_method_and_descriptions=prior_types_injection_method_and_descriptions,
    )
    df_prior_sampling = get_regret_data(
        db=db,
        optimization_type='PriorSampling',
        objective_type='MRLayoutQuality',
        prior_types_injection_method_and_descriptions=prior_sampling_injection_methods_and_descriptions,
    )
    df = pd.concat([df_bo, df_prior_sampling])

    # Plot the regret for each bias level
    num_initial_samples = 4
    plot_regret(ax, df, num_initial_samples)
    ax.set_title('MR Layout Quality BO')

    return fig, ax

# def regret_by_technique_and_temperature(
#         db: Database,
#         optimization_type: str,
#         objective_type: str,
#         num_initial_samples: int = 4,
#         max_iterations_to_plot: int = 40,
#         **kwargs,
#     ) -> Tuple[plt.Figure, plt.Axes]:
#     """Create an illustration of the regret MR layout quality plot."""
#     fig, axs = plt.subplots(1, 4, figsize=(20, 4))

#     injection_method_comparison_ax, prior_sampling_ax, pibo_ax, colabo_ax = axs

#     # Injection method comparison
#     df_bo = get_regret_data(
#         db=db,
#         optimization_type=optimization_type,
#         objective_type=objective_type,
#         prior_types_injection_method_and_descriptions=[
#             ('None', 'None', 'MC-LogEI'),
#             ('Biased', 'piBO', r'$\pi$BO-MC-LogEI'),
#             ('Biased', 'ColaBO', 'ColaBO-MC-LogEI'),
#         ],
#         **kwargs
#     )
#     df_prior_sampling = get_regret_data(
#         db=db,
#         optimization_type='PriorSampling',
#         objective_type=objective_type,
#         prior_types_injection_method_and_descriptions=[
#             ('Biased', 'None', 'Prior Sampling'),
#         ],
#         **kwargs
#     )
#     df_injection = pd.concat([df_bo, df_prior_sampling])

#     # Drop all iterations after 40
#     df_injection = df_injection[df_injection['iteration'] <= max_iterations_to_plot]

#     plot_regret(injection_method_comparison_ax, df_injection, num_initial_samples)
#     injection_method_comparison_ax.set_title(f'{optimization_type} {objective_type}')

#     # Temperature comparison
#     for ax, prior_injection_method in zip([prior_sampling_ax, pibo_ax, colabo_ax], ['PriorSampling', 'piBO', 'ColaBO']):
        
#         # Hotfix to make PriorSampling plotting work
#         original_prior_injection_method = prior_injection_method
#         original_optimization_type = optimization_type
#         if prior_injection_method == 'PriorSampling':
#             optimization_type = 'PriorSampling'
#             prior_injection_method = 'None'

#         df = get_regret_data(
#             db=db,
#             optimization_type=optimization_type,
#             objective_type=objective_type,
#             prior_types_injection_method_and_descriptions=[
#                 ('BiasedMoreCertain', prior_injection_method, r'$T = 0.01$'),
#                 ('BiasedCertain', prior_injection_method, r'$T = 0.1$'),
#                 ('Biased', prior_injection_method, r'$T = 1.0$'),
#                 ('BiasedUncertain', prior_injection_method, r'$T = 10.0$'),
#                 ('BiasedMoreUncertain', prior_injection_method, r'$T = 100.0$'),
#             ],
#             **kwargs
#         )

#         optimization_type = original_optimization_type

#         # Drop all iterations after 40
#         df = df[df['iteration'] <= max_iterations_to_plot]

#         plot_regret(ax, df, num_initial_samples)
#         ax.set_title(f'{original_prior_injection_method if original_prior_injection_method != "piBO" else r"$\pi$BO"}')

#     # Remove y-axis label from all but first subfigure
#     for ax in axs[1:]:
#         ax.set_ylabel(None)

#     return fig, axs

def regret_by_technique_and_temperature(
        df: pd.DataFrame,
        optimization_type: str,
        objective_type: str,
        num_initial_samples: int = 4,
        max_iterations_to_plot: int = 40,
        prior_injection_techniques=['PriorSampling', 'piBO', 'ColaBO'],
    ) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret MR layout quality plot."""
    # Drop all iterations after 40
    df = df[df['iteration'] <= (max_iterations_to_plot + num_initial_samples)]
    df.rename(columns={'label': 'original_label'}, inplace=True)

    # Get all data where prior was biased and temperature was 1.0
    df_injection_comparison = df[(df['original_label'].str.contains('T = 1.0', na=False, regex=False) & df['original_label'].str.contains('Biased', na=False))
                                 | df['original_label'].str.contains('MC-LogEI', na=False, regex=False)]

    # Rename columns
    df_injection_comparison['label'] = df_injection_comparison['original_label'].replace({fr'{prior_injection_technique}, Biased, $T = 1.0$': fr'{prior_injection_technique}-MC-LogEI' for prior_injection_technique in prior_injection_techniques})
    df_injection_comparison['label'].replace({'PriorSampling-MC-LogEI': 'PriorSampling'}, inplace=True)
    df_injection_comparison['label'].replace({'piBO-MC-LogEI': r'$\pi$BO-MC-LogEI'}, inplace=True)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    injection_method_comparison_ax, prior_sampling_ax, pibo_ax, colabo_ax = axs

    plot_regret(injection_method_comparison_ax, df_injection_comparison, num_initial_samples)
    injection_method_comparison_ax.set_title(f'{optimization_type} {objective_type}')

    # Temperature comparison
    for ax, prior_injection_method in zip([prior_sampling_ax, pibo_ax, colabo_ax], prior_injection_techniques):

        df_method = df[df['original_label'].str.contains(prior_injection_method)]
        df_method['label'] = df_method['original_label'].str.replace(r'.*\$T = ([\d\.]+)\$', r'$T = \1$', regex=True)
        df_method['temperature'] = df_method['original_label'].str.extract(r'\$T = ([\d\.]+)\$')[0].astype(float)

        df_sorted = df_method.sort_values(by=['temperature', 'iteration'], ascending=True)

        plot_regret(ax, df_sorted, num_initial_samples)
        ax.set_title(f'{prior_injection_method if prior_injection_method != "piBO" else r"$\pi$BO"}')

    # Remove y-axis label from all but first subfigure
    for ax in axs[1:]:
        ax.set_ylabel(None)

    return fig, axs

def regret_by_technique_and_temperature_df(
        db: Database,
        optimization_type: str,
        objective_type: str,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret MR layout quality plot."""
    # Injection method comparison
    df_bo = get_regret_data(
        db=db,
        optimization_type=optimization_type,
        objective_type=objective_type,
        prior_types_injection_method_and_descriptions=[
            ('None', 'None', 'MC-LogEI'),
            ('Biased', 'ColaBO', r'ColaBO, Biased, $T = 1.0$'),
            ('BiasedCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.1$'),
            ('BiasedMoreCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.01$'),
            ('BiasedUncertain', 'ColaBO', r'ColaBO, Biased, $T = 10.0$'),
            ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, Biased, $T = 100.0$'),
            ('Biased', 'piBO', r'piBO, Biased, $T = 1.0$'),
            ('BiasedCertain', 'piBO', r'piBO, Biased, $T = 0.1$'),
            ('BiasedMoreCertain', 'piBO', r'piBO, Biased, $T = 0.01$'),
            ('BiasedUncertain', 'piBO', r'piBO, Biased, $T = 10.0$'),
            ('BiasedMoreUncertain', 'piBO', r'piBO, Biased, $T = 100.0$'),
        ],
        **kwargs
    )
    df_prior_sampling = get_regret_data(
        db=db,
        optimization_type='PriorSampling',
        objective_type=objective_type,
        prior_types_injection_method_and_descriptions=[
            ('Biased', 'None', r'PriorSampling, Biased, $T = 1.0$'),
            ('BiasedCertain', 'None', r'PriorSampling, Biased, $T = 0.1$'),
            ('BiasedMoreCertain', 'None', r'PriorSampling, Biased, $T = 0.01$'),
            ('BiasedUncertain', 'None', r'PriorSampling, Biased, $T = 10.0$'),
            ('BiasedMoreUncertain', 'None', r'PriorSampling, Biased, $T = 100.0$'),
        ],
        **kwargs
    )
    df = pd.concat([df_bo, df_prior_sampling])

    return df

def get_only_max_paths_for_colabo_df(
        db: Database,
        optimization_type: str,
        objective_type: str,
    ) -> pd.DataFrame:
    # Injection method comparison
    return get_regret_data(
        db=db,
        optimization_type=optimization_type,
        objective_type=objective_type,
        prior_types_injection_method_and_descriptions=[
            ('Biased', 'ColaBO', r'ColaBO, Biased, $T = 1.0$'),
            ('BiasedCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.1$'),
            ('BiasedMoreCertain', 'ColaBO', r'ColaBO, Biased, $T = 0.01$'),
            ('BiasedUncertain', 'ColaBO', r'ColaBO, Biased, $T = 10.0$'),
            ('BiasedMoreUncertain', 'ColaBO', r'ColaBO, Biased, $T = 100.0$'),
        ],
        num_paths=65536,
    )

def main():
    db = Database(os.getenv('DATA_DIR') + 'experiments.db')
    plots_dir = os.getenv('PLOTS_DIR')

    for plot_func, filename in [
        (regret_sphere_plot, 'regret_sphere.png'),
        (regret_shekel_plot, 'regret_shekel.png'),
        (regret_image_similarity_plot, 'regret_image_similarity.png'),
        (regret_scatterplot_quality_plot, 'regret_scatterplot_quality.png'),
        (regret_mr_layout_quality_plot, 'regret_mr_layout_quality.png'),
    ]:
        fig, ax = plot_func(db)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, filename), dpi=300)

    for optimization_type, objective_type in [
        ('BO', 'Sphere'),
        ('BO', 'SphereNoisy'),
        ('PBO', 'Sphere'),
        ('BO', 'Shekel'),
        ('BO', 'ImageSimilarity'),
        ('PBO', 'ImageSimilarity'),
        ('BO', 'ScatterPlotQuality'),
        ('BO', 'MRLayoutQuality'),
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

if __name__ == '__main__':
    main()