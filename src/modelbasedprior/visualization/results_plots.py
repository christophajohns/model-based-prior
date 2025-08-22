"""Plotting functions for the paper."""

import logging
import json
from pathlib import Path
import torch
from typing import Tuple, List, Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from modelbasedprior.benchmarking.database import Database
from modelbasedprior.objectives.sphere import Sphere

logger = logging.getLogger(__name__)

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
    ax.grid(True, which='both', linestyle='-', alpha=0.3)

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

def regret_sphere_plot(
        db: Database,
        prior_types_injection_method_and_descriptions: List[Tuple[str, str, str]] = [
            ('None', 'None', 'ConventionalBO'),
            ('Biased', 'ColaBO', 'ColaBO'),
            ('Biased', 'piBO', r'$\pi$BO'),
        ],
        prior_sampling_injection_methods_and_descriptions: List[Tuple[str, str, str]] = [
            ('Biased', 'None', 'PriorSampling'),
        ],
        optimization_and_objective_types: List[Tuple[str, str, str]] = [
            ('BO', 'Sphere', 'Sphere BO'),
            ('BO', 'SphereNoisy', 'Sphere BO (Noisy)'),
            ('PBO', 'Sphere', 'Sphere PBO'),
        ],
        ax_size: Tuple[float, float] = (3.5, 3),
    ) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret sphere plot."""
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
    for optimization_type, objective_type, _ in optimization_and_objective_types:
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
    fig, axes = plt.subplots(1, len(dfs), figsize=(ax_size[0] * len(dfs), ax_size[1]), sharex=True)
    if len(dfs) == 1: axes = [axes]  # Make sure that axes is iterable, even if only plotting a single optimization and objective combination

    # Plot the regret for each bias level
    titles = [title for _, _, title in optimization_and_objective_types]
    for ax, df, title in zip(axes, dfs, titles):
        plot_regret(ax, df, num_initial_samples)
        ax.set_title(title)

    # Remove the y-axis label from all plots except the first one
    for ax in axes[1:]:
        ax.set_ylabel('')

    return fig, axes

def regret_shekel_plot(db: Database) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret shekel plot."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    prior_types_injection_method_and_descriptions = [
        ('None', 'None', 'ConventionalBO'),
        ('Biased', 'piBO', r'$\pi$BO'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'PriorSampling'),
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
            ('Biased', 'ColaBO', 'ColaBO'),
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

def regret_image_similarity_plot(
        db: Database,
        prior_types_injection_method_and_descriptions: List[Tuple[str, str, str]] = [
            ('None', 'None', 'ConventionalBO'),
            ('Biased', 'ColaBO', 'ColaBO'),
            ('Biased', 'piBO', r'$\pi$BO'),
        ],
        prior_sampling_injection_methods_and_descriptions: List[Tuple[str, str, str]] = [
            ('Biased', 'None', 'PriorSampling'),
        ],
        optimization_types: List[str] = ['BO', 'PBO'],
    ) -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the regret image similarity plot."""
    num_initial_samples = 4
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
    if len(optimization_types) == 1: ax = [ax]  # Make sure that ax is iterable, even if only plotting a single optimization type

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
        ('None', 'None', 'ConventionalBO'),
        ('Biased', 'ColaBO', 'ColaBO'),
        ('Biased', 'piBO', r'$\pi$BO'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'PriorSampling'),
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
        ('None', 'None', 'ConventionalBO'),
        ('Biased', 'ColaBO', 'ColaBO'),
        ('Biased', 'piBO', r'$\pi$BO'),
    ]
    prior_sampling_injection_methods_and_descriptions = [
        ('Biased', 'None', 'PriorSampling'),
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

def load_image_tuning_data(base_save_dir: str, task: Literal["Aesthetics", "Reference"], optimal_value: float = 10.0) -> pd.DataFrame:
    """
    Loads image tuning results from the specified directory structure.

    Args:
        base_save_dir: The root directory containing participant folders.
                       Expected structure: <base_save_dir>/participant_<id>/<condition_id>/<image_id>/results_<image_id>.pt
        task: The image tuning task used to construct the <condition_id> (e.g., "Aesthetics" or "Reference").
        optimal_value: The theoretical maximum achievable score (e.g., 10.0 for ratings).

    Returns:
        A pandas DataFrame containing aggregated results with regret calculated.
        Columns include: participant_id, condition_id, optimization_method,
                         image_id, iteration, best_rating_so_far, regret, log10_regret.
    """
    all_results = []
    base_path = Path(base_save_dir)

    if not base_path.is_dir():
        logger.error(f"Base save directory not found or is not a directory: {base_save_dir}")
        return pd.DataFrame() # Return empty dataframe

    # Iterate through participant directories
    for participant_dir in base_path.glob("participant_*"):
        if not participant_dir.is_dir():
            continue
        try:
            participant_id = int(participant_dir.name.split('_')[-1])
        except ValueError:
            logger.warning(f"Could not parse participant ID from directory name: {participant_dir.name}")
            continue

        # Iterate through condition directories
        for condition_dir in participant_dir.iterdir():
            if not condition_dir.is_dir():
                continue
            condition_id = condition_dir.name # e.g., "ColaBO_Aesthetics"
            if not task in condition_id:
                continue

            # --- Load config to get optimization method ---
            # Option 1: Load from config_run.json (preferred if always present)
            config_path = condition_dir / "config_run.json"
            optimization_method = None
            num_initial_samples = 4 # Default, try to read from config
            if config_path.exists():
                 try:
                     with open(config_path, 'r') as f:
                         config_data = json.load(f)
                     optimization_method = config_data.get("optimization_method", "Unknown")
                     num_initial_samples = config_data.get("num_initial_samples", 4)
                 except Exception as e:
                     logger.warning(f"Failed to load or parse {config_path}: {e}")
            else:
                 logger.warning(f"Config file not found in {condition_dir}, cannot determine optimization method reliably.")
                 # As a fallback, try parsing from condition_id (less robust)
                 if "_" in condition_id:
                     optimization_method = condition_id.split('_')[0]
                 else:
                     optimization_method = condition_id # Or set to Unknown

            if not optimization_method or optimization_method == "Unknown":
                 logger.warning(f"Skipping condition {condition_id} for participant {participant_id} due to missing optimization method.")
                 continue


            # Iterate through image directories
            for image_dir in condition_dir.iterdir():
                if not image_dir.is_dir():
                    continue
                image_id = image_dir.name

                # Find the results file
                results_file = next(image_dir.glob("results_*.pt"), None)
                if results_file:
                    try:
                        data = torch.load(results_file, map_location='cpu', weights_only=True) # Load to CPU
                        best_y_trace = data.get('result_best_y_trace').squeeze()

                        if best_y_trace is not None and isinstance(best_y_trace, torch.Tensor):
                            for i, best_y_val in enumerate(best_y_trace.numpy()):
                                iteration = i + 1 # 1-based iteration index
                                regret = optimal_value - best_y_val
                                log10_regret = np.log10(max(regret, 1e-10)) # Avoid log(0) or log(-)

                                all_results.append({
                                    'participant_id': participant_id,
                                    'condition_id': condition_id,
                                    'optimization_method': optimization_method,
                                    'image_id': image_id,
                                    'iteration': iteration,
                                    'best_rating_so_far': best_y_val,
                                    'regret': regret,
                                    'log10_regret': log10_regret,
                                    'num_initial_samples': num_initial_samples # Store for later use if needed
                                })
                        else:
                            logger.warning(f"result_best_y_trace missing or not a Tensor in {results_file}")

                    except Exception as e:
                        logger.error(f"Failed to load or process {results_file}: {e}")
                else:
                     logger.warning(f"No results .pt file found in {image_dir}")

    if not all_results:
        logger.warning("No valid results data found.")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def prepare_regret_dataframe(
    df_raw: pd.DataFrame,
    grouping_col: str = 'optimization_method'
) -> pd.DataFrame:
    """
    Aggregates raw regret data into the format needed for plot_regret.

    Args:
        df_raw: DataFrame loaded by load_image_tuning_data.
        grouping_col: The column name to group by (e.g., 'optimization_method').
                      This will become the 'label' in the output DataFrame.

    Returns:
        A pandas DataFrame aggregated by iteration and the grouping column,
        containing median, quartiles, min, and max of log10_regret.
    """
    if df_raw.empty:
        return pd.DataFrame()
    if grouping_col not in df_raw.columns:
        logger.error(f"Grouping column '{grouping_col}' not found in DataFrame.")
        return pd.DataFrame()
    if 'iteration' not in df_raw.columns or 'log10_regret' not in df_raw.columns:
        logger.error("Required columns 'iteration' or 'log10_regret' missing.")
        return pd.DataFrame()

    # Group by the specified column and iteration, then aggregate log10_regret
    df_agg = df_raw.groupby([grouping_col, 'iteration'])['log10_regret'].agg([
        ('min_log10_regret', 'min'),
        ('first_quartile_log10_regret', lambda x: np.percentile(x, 25)),
        ('median_log10_regret', 'median'),
        ('third_quartile_log10_regret', lambda x: np.percentile(x, 75)),
        ('max_log10_regret', 'max')
    ]).reset_index()

    # Rename the grouping column to 'label' as expected by plot_regret
    df_agg = df_agg.rename(columns={grouping_col: 'label'})

    return df_agg


def regret_image_tuning_plot(
    base_save_dir: str,
    task: Literal["Aesthetics", "Reference"],
    optimal_value: float = 10.0,
    num_initial_samples: int | None = None, # If None, tries to infer from data
    max_iterations_to_plot: int | None = None,
    shaded_area: Literal['min_max', 'iqr'] | None = 'iqr',
    title: str = "Image Tuning Regret",
    figsize: Tuple[int, int] = (5, 4)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Loads image tuning data and plots the aggregated regret over iterations.

    Args:
        base_save_dir: The root directory containing participant folders.
        task: The image tuning task used to construct the <condition_id> (e.g., "Aesthetics" or "Reference").
        optimal_value: The theoretical maximum achievable score (e.g., 10.0).
        num_initial_samples: Number of initial samples used in BO (for plotting vertical line).
                             If None, it tries to infer from the loaded data (uses the mode).
        max_iterations_to_plot: Maximum iteration number to include in the plot.
                                If None, includes all iterations found.
        shaded_area: Type of shaded area to plot ('min_max', 'iqr', or None).
        title: The title for the plot.
        figsize: The size of the matplotlib figure.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """
    # Load the raw data
    df_raw = load_image_tuning_data(base_save_dir, task, optimal_value)

    if df_raw.empty:
        logger.error("Could not load any data. Cannot generate plot.")
        # Return empty figure/axes
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title} (No Data)")
        return fig, ax

    # Determine num_initial_samples if not provided
    if num_initial_samples is None:
        if 'num_initial_samples' in df_raw.columns:
            try:
                # Use the most frequent value found in the data
                num_initial_samples = df_raw['num_initial_samples'].mode()[0]
                logger.info(f"Inferred num_initial_samples = {num_initial_samples}")
            except IndexError:
                logger.warning("Could not infer num_initial_samples, defaulting to 4.")
                num_initial_samples = 4 # Fallback default
        else:
            logger.warning("num_initial_samples column missing, defaulting to 4.")
            num_initial_samples = 4

    # Filter iterations if needed
    if max_iterations_to_plot is not None:
        df_raw = df_raw[df_raw['iteration'] <= max_iterations_to_plot]
        logger.info(f"Plotting up to iteration {max_iterations_to_plot}")

    # Prepare the data for plotting (aggregate by optimization method)
    df_plot = prepare_regret_dataframe(df_raw, grouping_col='optimization_method')

    if df_plot.empty:
        logger.error("Data aggregation failed. Cannot generate plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title} (Aggregation Failed)")
        return fig, ax
    
    # Rename piBO
    df_plot["label"] = df_plot["label"].replace("piBO", r"$\pi$BO")

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_regret(
        ax=ax,
        df=df_plot,
        num_initial_samples=num_initial_samples,
        shaded_area=shaded_area
    )

    ax.set_title(title)
    fig.tight_layout()

    return fig, ax

def main():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    image_tuning_results_dir = os.getenv("IMAGE_TUNING_SAVE_DIR", "./image_tuning_results") # Get base dir
    if os.path.exists(image_tuning_results_dir):
        logger.info(f"Attempting to plot image tuning results from: {image_tuning_results_dir}")
        try:
            fig_img, ax_img = regret_image_tuning_plot(
                base_save_dir=image_tuning_results_dir,
                task="Reference",
                optimal_value=10.0, # Max rating
                # num_initial_samples=4, # Can often be inferred
                # max_iterations_to_plot=30 # Set if needed
            )
            fig_img.tight_layout()
            # fig_img.suptitle("Image Tuning Experiment Regret") # Add overall title if desired
            # fig_img.savefig(os.path.join(plots_dir, 'regret_image_tuning.png'), dpi=300) # Uncomment to save
        except Exception as e:
            logger.error(f"Failed to generate image tuning regret plot: {e}", exc_info=True)
    else:
        logger.warning(f"Image tuning results directory not found, skipping plot: {image_tuning_results_dir}")

    db = Database(os.path.join(os.getenv('DATA_DIR'), 'experiments.db'))
    # plots_dir = os.getenv('PLOTS_DIR')

    for plot_func, filename in [
        # (regret_sphere_plot, 'regret_sphere.png'),
        # (regret_shekel_plot, 'regret_shekel.png'),
        # (regret_image_similarity_plot, 'regret_image_similarity.png'),
        # (regret_scatterplot_quality_plot, 'regret_scatterplot_quality.png'),
        # (regret_mr_layout_quality_plot, 'regret_mr_layout_quality.png'),
    ]:
        fig, ax = plot_func(db)
        fig.tight_layout()
        # fig.savefig(os.path.join(plots_dir, filename), dpi=300)

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
        # fig.savefig(os.path.join(
        #     plots_dir,
        #     f'regret_{optimization_type}_{objective_type}.png'
        # ), dpi=300)
    
    plt.show()

if __name__ == '__main__':
    main()