"""Create the illustrative plots for the paper."""

from typing import Tuple
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
from botorch.utils.prior import normalize, unnormalize
from botorch.sampling.pathwise import MatheronPath, draw_matheron_paths
from botorch.sampling.pathwise_sampler import PathwiseSampler
from botorch.acquisition.prior_monte_carlo import qPriorExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from dotenv import load_dotenv
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from modelbasedprior.optimization.bo import generate_data_from_prior, init_and_fit_model, make_new_data
from modelbasedprior.objectives.sphere import Sphere
from modelbasedprior.objectives.shekel import Shekel2D, Shekel2DNoGlobal, ShekelNoGlobal, Shekel1D, Shekel1DNoGlobal
from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss
from modelbasedprior.objectives.scatterplot_quality import ScatterPlotQualityLoss
from modelbasedprior.objectives.mr_layout_quality import MRLayoutQualityLoss
from modelbasedprior.prior import ModelBasedPrior

load_dotenv()

# Helper functions
def detach(tensor):
    return tensor.detach().cpu().numpy()

def normalize_X(X, bounds):
    return normalize(X.unsqueeze(-1), bounds)

def sphere_plot() -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the Sphere objective
    and the prior belief as the shifted Sphere objective."""
    fig, ax = plt.subplots(figsize=(2,2))

    # Create a grid
    x_limits = (-0.75, 1.2)
    x1 = x2 = torch.linspace(*x_limits, 100)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    X = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    # Set contour levels
    levels = torch.linspace(0, 0.4, 5)

    # Create a contour plot of the Sphere objective
    sphere_function = Sphere(dim=2)
    Y = sphere_function(X).reshape(X1.shape)
    contours = ax.contour(X1, X2, Y, levels=levels, colors='black', zorder=2)
    # ax.clabel(contours, inline=True, fontsize=8)

    # Create a contour plot of the prior belief
    # as Sphere(x - 0.5)
    delta = 0.5
    shifted_X = X - delta
    shifted_Y = sphere_function(shifted_X).reshape(X1.shape)
    contours_prior = ax.contour(X1, X2, shifted_Y, levels=levels, colors='grey', linestyles='dashed', zorder=2)
    # ax.clabel(contours_prior, inline=True, fontsize=8)

    # Add the respective global minima
    optimum_objective = (0, 0)
    optimum_prior = (delta, delta)
    ax.plot(*optimum_objective, 'ko', label='Global minimum', markersize=2)
    ax.plot(*optimum_prior, 'ko', label='Global minimum shifted', markersize=2, fillstyle='none')

    # Label the global minima
    optimum_objective_text = (optimum_objective[0], optimum_objective[1] - 0.1)
    ax.text(*optimum_objective_text, r'$x^*$', verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='circle,pad=0.2', mutation_aspect=0.8, fc='white', ec='none'))
    optimum_prior_text = (optimum_prior[0], optimum_prior[1] + 0.05)
    ax.text(*optimum_prior_text, r'$x^*_{\text{prior}}$', verticalalignment='bottom', horizontalalignment='center',
            bbox=dict(boxstyle='circle,pad=0.2', mutation_aspect=0.5, fc='white', ec='none'))

    # Denote the distance between the minima
    # and label it as delta
    # ax.annotate('', xy=optimum_objective, xytext=optimum_prior,
    #             arrowprops=dict(arrowstyle='<->', lw=1), zorder=1)
    # ax.text(0.25, 0.25, r'$\Delta$', verticalalignment='center', horizontalalignment='center',
    #         bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=1)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set the axis labels
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    # Remove the box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

def shekel_plot() -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the Shekel objective
    and the prior belief as the Shekel objective
    without the global minimum."""
    # Create a figure with two subplots side by side
    # that share the axes
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True, sharex=True)

    shekel_function = Shekel2D(negate=True)

    # Create a grid
    x_limits, y_limits = shekel_function.bounds.T
    resolution = 100
    x1 = torch.linspace(*x_limits, resolution)
    x2 = torch.linspace(*y_limits, resolution)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    X = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    # Set contour levels
    levels = torch.linspace(0, 3, 6)
    contour_line_width = 0.5

    # Create a contour plot of the Shekel objective
    Y = shekel_function(X).reshape(X1.shape)
    contours = axes[0].contour(X1, X2, Y, levels=levels, colors='black', linewidths=contour_line_width)
    # axes[0].clabel(contours, inline=True, fontsize=8)

    # Create a contour plot of the prior belief
    # as Shekel(x) without the global minimum
    shekel_function_no_global = Shekel2DNoGlobal(negate=True)
    Y_no_global = shekel_function_no_global(X).reshape(X1.shape)
    contours_prior = axes[1].contour(X1, X2, Y_no_global, levels=levels, colors='grey', linestyles='dashed', linewidths=contour_line_width)
    # axes[1].clabel(contours_prior, inline=True, fontsize=8)

    # Add the global minimum
    optimum = shekel_function.optimizers[0]
    for ax in axes:
        ax.plot(*optimum, 'ro', label=r'x^*', markersize=4)

    # Label the global minimum
    def annotate_optimum(ax, offset = (-2, -2)):
        text_position = (optimum[0] + offset[0], optimum[1] + offset[1])
        ax.annotate(r'$x^*$', xy=optimum, xytext=text_position,
                    arrowprops=dict(facecolor='red', edgecolor='none', shrink=0.05, width=2, headwidth=10),
                    bbox=dict(boxstyle='circle,pad=0.2', mutation_aspect=0.7, fc='white', ec='none'),
                    verticalalignment='center', horizontalalignment='center')
    annotate_optimum(axes[0])
    annotate_optimum(axes[1])

    # Remove ticks
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Label the axes as x1 and x2
    axes[0].set_xlabel(r'$x_1$')
    axes[0].set_ylabel(r'$x_2$')
    axes[1].set_xlabel(r'$x_1$')
    axes[1].set_ylabel(r'$x_2$')

    # Remove the box
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return fig, axes

def scatter_quality_plot() -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the scatter plot quality
    objective (one example with high and one with low
    quality) and the effect of the overplotting weight
    to illustrate the prior."""
    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    for ax, label in zip(axes[0], ['High objective value', 'Low objective value']):
        ax.set_title(label)

    for ax, label in zip(axes[1], ['Low overplotting', 'High overplotting']):
        ax.set_title(label)

    # Load the Cars dataset
    df = sns.load_dataset('mpg')  # Load Cars dataset from seaborn
    x_data = torch.tensor(df['horsepower'].values, dtype=torch.float32)
    y_data = torch.tensor(df['mpg'].values, dtype=torch.float32)

    # Label the axes
    for ax in axes.flatten():
        ax.set_xlabel('horsepower')
        ax.set_ylabel('mpg')

    # Remove the box
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add the scatter plot
    good_marker_size = 12
    bad_marker_size = 64
    good_marker_opacity = 32
    bad_marker_opacity = 120
    axes[0,0].scatter(x_data, y_data, s=good_marker_size, c='black', alpha=good_marker_opacity / 255.0)
    axes[0,1].scatter(x_data, y_data, s=bad_marker_size, c='black', alpha=bad_marker_opacity / 255.0)
    axes[1,0].scatter(x_data, y_data, s=bad_marker_size, c='black', alpha=0.05)
    axes[1,1].scatter(x_data, y_data, s=bad_marker_size, c='black', alpha=0.9)

    # Evalute the ScatterPlotQualityLoss
    # scatter_quality_loss = ScatterPlotQualityLoss(x_data=x_data, y_data=y_data, negate=True)
    # ratings = [scatter_quality_loss(torch.tensor([marker_size, marker_opacity, 1.0]))
    #            for marker_size, marker_opacity in [(good_marker_size, good_marker_opacity),
    #                                                 (bad_marker_size, bad_marker_opacity)]]
    # print([rating.item() for rating in ratings])

    axes[0,1].set_box_aspect(2)

    fig.tight_layout(h_pad=1.5)

    return fig, axes

def prior_temperature_plot() -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates a plot of the prior over the optimum distribution for various prior functions and temperatures.
    This function creates a grid of subplots where each row corresponds to a different transformation function
    and each column corresponds to a different prior function. The plot shows how the prior 
    distribution changes with different temperatures.

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the figure and axes of the generated plot.
    """

    df = sns.load_dataset('mpg')  # Load Cars dataset from seaborn
    x_data = torch.tensor(df['horsepower'].values, dtype=torch.float32)
    y_data = torch.tensor(df['mpg'].values, dtype=torch.float32)

    sphere = Sphere(negate=True)
    shekel = ShekelNoGlobal(negate=True)
    image_similarity = ImageSimilarityLoss(resize(read_image(os.path.join(os.getenv("AVA_FLOWERS_DIR"), '43405.jpg')), 64), weight_psnr=0.2, weight_ssim=0.8, negate=True)
    scatterplot_quality = ScatterPlotQualityLoss(x_data=x_data, y_data=y_data, weight_overplotting=0, use_approximate_model=True, negate=True)
    mr_layout_quality = MRLayoutQualityLoss(negate=True)
    prior_funcs = [
        ('Sphere', lambda x: sphere(x - 0.5), sphere.bounds),
        ('Shekel', shekel, shekel.bounds),
        ('Image Simil.', image_similarity, image_similarity.bounds),
        ('MR Layout', mr_layout_quality, mr_layout_quality.bounds),
        ('Scatter Plot', scatterplot_quality, scatterplot_quality.bounds),
    ]

    def normalized_boltzmann(prior_func, temperature, bounds):
        prior = ModelBasedPrior(bounds=bounds, predict_func=prior_func, temperature=temperature, minimize=False, seed=42)
        return lambda x: torch.exp(prior.evaluate(normalize(x, bounds)))
    
    transformations = [
        # ('Softmax', normalized_boltzmann),
        ('Exponential', lambda prior_func, temperature, *args: lambda x: torch.exp(prior_func(x) / temperature)),
        # ('Sigmoid', lambda prior_func, temperature, *args: lambda x: 1 / (1 + torch.exp(-prior_func(x) / temperature))),
        # ('Tanh', lambda prior_func, temperature, *args: lambda x: (torch.tanh(prior_func(x) / temperature) + 1) / 2),
        # ('Exp-Tanh', lambda prior_func, temperature, *args: lambda x: torch.exp(torch.tanh(prior_func(x) / temperature))),
        # ('Softplus', lambda prior_func, temperature, *args: lambda x: torch.nn.functional.softplus(prior_func(x), beta=1/temperature)),
        # ('ArcTan', lambda prior_func, temperature, *args: lambda x: torch.atan(prior_func(x) / temperature) + torch.tensor([torch.pi / 2])),
        # ('Softsign', lambda prior_func, temperature, *args: lambda x: torch.nn.functional.softsign(prior_func(x) / temperature) + torch.tensor(1.)),
    ]
    temperatures = [0.5, 1.0, 5.0, 10.0, 100.0]
    fig, axes = plt.subplots(len(transformations) + 1, len(prior_funcs) + 1, figsize=((len(prior_funcs) + 1) * 1.4, (len(transformations) + 1) * 1.4), layout='constrained')
    temperature_colors = cm.plasma(torch.linspace(0.1, 0.9, len(temperatures))) # Use a subset of the colormap to avoid very light/dark ends

    # First row: Prior functions
    axes[0, 0].axis("off")  # Empty top-left cell

    for ax, (prior_name, prior_func, bounds) in zip(axes[0, 1:], prior_funcs):
        ax.set_title(prior_name)

        # Create x-values similar to the main plots
        x = torch.zeros(100, bounds.size()[1])
        x[:, 0] = torch.linspace(*bounds.T[0], x.size()[0])
        for i, (lower_bound, upper_bound) in enumerate(bounds.T[1:], start=1):
            x[:, i] = (lower_bound + upper_bound) / 2

        y_pred = prior_func(x)
        ax.plot(x[:, 0], y_pred, label="Prior Function", color="k")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])

    for row, (transformation_name, prob) in enumerate(transformations, start=1):
        ax = axes[row, 0]

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        ax.set_title(transformation_name)

        bounds = (-5., 5.)
        x = torch.linspace(*bounds, 100)
        y = prob(lambda x: x, 1.0, torch.tensor([bounds]).T)(x)
        ax.plot(x, y, label='__nolabel__', color="tab:blue")

        for temp_idx, temperature in enumerate(temperatures):
            for (prior_name, prior_func, bounds), ax in zip(prior_funcs, axes[row, 1:]):

                # Create an x-vector where only the first dimension is linearly spaced
                # and the rest are the midpoint of the bounds
                x = torch.zeros(100, bounds.size()[1])
                x[:, 0] = torch.linspace(*bounds.T[0], x.size()[0])
                for i, (lower_bound, upper_bound) in enumerate(bounds.T[1:], start=1):
                    x[:, i] = (lower_bound + upper_bound) / 2

                y_pred = prior_func(x)
                # Compress the y-values to be in the range [-5, 5] to take advantage of the transformation shape
                # independently of the location of the model values
                y = prob(lambda x: (prior_func(x) - y_pred.min()) / (y_pred.max() - y_pred.min()) * 10 - 5, temperature, bounds)(x)
                y_scaled = y / torch.sum(y, dim=0)

                x_1d = x if x.size()[1] == 1 else x[:, 0]
                ax.plot(x_1d, y_scaled, label=fr'$T = {temperature}$', color=temperature_colors[temp_idx])

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Remove y-axis ticks
                ax.set_yticks([])

    for ax, (prior_name, _prior_func, _bounds) in zip(axes[0, 1:], prior_funcs):  # Only the first row
        ax.set_title(prior_name)
    
    for ax in axes[-1]:
        ax.set_xlabel(r'$x_1$')
    # axes[-1,1].set_xlabel(r'$x_1 \mid x_i = \frac{x_i^\text{min} + x_i^\text{max}}{2} \forall i > 1$')
    # for ax in axes[-1][2:]:  # Only the last row
    #     ax.set_xlabel(r'$x_1 \mid \ldots$')
    axes[-1][0].set_xlabel(r'$\hat f(x)$')

    # Remove the x-axis labels for all but the last row and the first row
    for ax in axes[:-1, :].flatten():
        ax.set_xticks([])

    for ax in axes[:, 0]:  # Only the first column
        ax.set_ylabel(r'$\pi(x)$')

    # Only the first row, second column (prior functions)
    axes[0, 1].set_ylabel(r'$\hat f(x)$')

    # top_left_ax = axes[0, 0]
    # top_left_ax.legend()
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(temperatures), bbox_to_anchor=(0.5, -0.15))

    return fig, axes

def initial_samples_plot() -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates and plots initial samples for a Bayesian optimization process.

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the figure and axes of the plot.
    
    The function performs the following steps:
    1. Sets up the initial parameters and helper functions.
    2. Initializes the objective function and priors.
    3. Generates initial data and fits the model.
    4. Iteratively optimizes the acquisition function and updates the model.
    5. Plots the priors, Gaussian process paths, and acquisition functions at each iteration.

    The plot includes:
    - The prior distributions with different temperatures.
    - The Gaussian process mean and standard deviation.
    - The evaluated points and sample paths.
    - The acquisition functions.
    """

    num_iterations = 5
    num_paths = 4096
    num_paths_to_plot = 5
    num_resampling_paths = 512
    temperature = 0.1
    high_temperature = 1.0
    prior_offset = 4.0 # 0.05 or 4.0
    seed = 42
    
    # Setup
    torch.manual_seed(seed)
    objective = Sphere(dim=1, negate=True)
    fig, axes = plt.subplots(2, num_iterations + 1, figsize=((num_iterations + 1) * 4, 8), sharex=True)
    X = torch.linspace(*detach(objective.bounds.T[0]), 100)

    # Priors
    predict_func = lambda x: objective(x - prior_offset)
    user_prior = ModelBasedPrior(bounds=objective.bounds, predict_func=predict_func, temperature=temperature, seed=seed, minimize=False)
    user_prior_high_temperature = ModelBasedPrior(bounds=objective.bounds, predict_func=predict_func, temperature=high_temperature, seed=seed, minimize=False)

    # Initial model
    init_X, init_y = generate_data_from_prior(objective=objective, user_prior=user_prior, n=1)
    model = init_and_fit_model(init_X, init_y, objective.bounds)

    # Plot priors
    prior_ax = axes[1, 0]
    prior_ax.plot(detach(X), detach(torch.exp(user_prior.evaluate(normalize_X(X, objective.bounds)))), label=fr'$T={temperature}$', color='blue')
    prior_ax.plot(detach(X), detach(torch.exp(user_prior_high_temperature.evaluate(normalize_X(X, objective.bounds)))), label=fr'$T={high_temperature}$', color='saddlebrown', linestyle='dashed')
    prior_ax.axvline(user_prior.default, color='grey', linestyle='dashed', label='Default prior')
    prior_ax.set_xlabel(r'$x$')
    prior_ax.set_ylabel(r'$\pi(x)$')

    # Optimization iterations
    for i in range(1, num_iterations+1):
        paths = draw_matheron_paths(model=model, sample_shape=torch.Size([num_paths]))
        sampler = PathwiseSampler(sample_shape=torch.Size([num_paths]))

        # Define acquisition functions
        acq_funcs = {
            fr"ColaBO-EI ($T={temperature}$)": qPriorExpectedImprovement(
                model=model,
                paths=paths,
                sampler=sampler,
                X_baseline=init_X,
                user_prior=user_prior,
                resampling_fraction=num_resampling_paths/num_paths,
                custom_decay=1.0,
            ),
            fr"ColaBO-EI ($T={high_temperature}$)": qPriorExpectedImprovement(
                model=model,
                paths=paths,
                sampler=sampler,
                X_baseline=init_X,
                user_prior=user_prior_high_temperature,
                resampling_fraction=num_resampling_paths/num_paths,
                custom_decay=1.0,
            ),
            fr"ColaBO-EI (Uniform)": qPriorExpectedImprovement(
                model=model,
                paths=paths,
                sampler=sampler,
                X_baseline=init_X,
                user_prior=None,
                resampling_fraction=num_resampling_paths/num_paths,
                custom_decay=1.0,
            )
        }
        acq_func_to_use = acq_funcs[list(acq_funcs.keys())[0]]

        # Generate candidate points
        X_cand = torch.cat((init_X.unsqueeze(0).expand(X.size(0), -1, -1), X.view(X.size(0), 1, 1)), dim=1)
        acq_vals = {label: acq_func(X_cand) for label, acq_func in acq_funcs.items()}
        acq_vals[fr"MC-$\pi$BO"] = acq_vals[fr"ColaBO-EI (Uniform)"] * torch.pow(torch.exp(user_prior.evaluate(normalize_X(X, objective.bounds)).squeeze()), 1.0/i)

        # Generate paths
        paths = acq_func_to_use.sampling_model.paths.paths
        matheron_path = MatheronPath(paths.prior_paths, paths.update_paths)
        uniform_prior_samples = matheron_path(X.unsqueeze(-1))
        prior_samples = matheron_path(X.unsqueeze(-1), subset=acq_func_to_use.indices)
        prior_high_temperature_samples = matheron_path(X.unsqueeze(-1), subset=acq_funcs[fr"ColaBO-EI ($T={high_temperature}$)"].indices)
        uniform_prior_mean, uniform_prior_std = uniform_prior_samples.mean(dim=0), uniform_prior_samples.std(dim=0)
        prior_mean, prior_std = prior_samples.mean(dim=0), prior_samples.std(dim=0)
        prior_high_temperature_mean, prior_high_temperature_std = prior_high_temperature_samples.mean(dim=0), prior_high_temperature_samples.std(dim=0)

        # Plot paths
        gp_ax, acqf_ax = axes[0, i], axes[1, i]
        gp_ax.set_title(f'Iteration {i}')
        acqf_ax.set_xlabel(r'$x$')
        gp_ax.plot(detach(X), detach(uniform_prior_mean), label='Uniform prior', color='black', linestyle="dashed")
        gp_ax.fill_between(detach(X), detach(uniform_prior_mean) - detach(uniform_prior_std), detach(uniform_prior_mean) + detach(uniform_prior_std), color='black', linestyle="dashed", alpha=0.1)
        gp_ax.plot(detach(X), detach(prior_mean), label='Model-based prior', color='blue')
        gp_ax.fill_between(detach(X), detach(prior_mean) - detach(prior_std), detach(prior_mean) + detach(prior_std), color='blue', alpha=0.1)
        gp_ax.plot(detach(X), detach(prior_high_temperature_mean), label=fr'Model-based prior ($T={high_temperature}$)', color='saddlebrown', linestyle='dotted')
        gp_ax.fill_between(detach(X), detach(prior_high_temperature_mean) - detach(prior_high_temperature_std), detach(prior_high_temperature_mean) + detach(prior_high_temperature_std), color='saddlebrown', alpha=0.1)

        # Plot evaluated points
        gp_ax.plot(detach(unnormalize(acq_func_to_use.sampling_model.train_inputs[0][:-1], objective.bounds)), detach(acq_func_to_use.sampling_model.train_targets[:-1]), 'ko', label='Previous samples', markersize=4)
        gp_ax.plot(detach(unnormalize(acq_func_to_use.sampling_model.train_inputs[0][-1], objective.bounds)), detach(acq_func_to_use.sampling_model.train_targets[-1]), 'ro', label='Last sample', markersize=4)

        # Plot sample paths
        for path_idx in range(num_paths_to_plot):
            gp_ax.plot(detach(X), detach(uniform_prior_samples[path_idx]), color='black', alpha=0.9, linestyle=(0, (5, 5)), linewidth=0.3, label=r'$f_i \sim p(f \mid D)$' if path_idx == 0 else '__nolabel__')
            gp_ax.plot(detach(X), detach(prior_samples[path_idx]), color='blue', alpha=0.9, linewidth=0.3, label=r'$f_i \sim p(f \mid D, \rho)$' if path_idx == 0 else '__nolabel__')
            gp_ax.plot(detach(X), detach(prior_high_temperature_samples[path_idx]), color='saddlebrown', alpha=0.9, linestyle='dotted', linewidth=0.3, label=fr'$f_i \sim p(f \mid D, \rho, T={high_temperature})$' if path_idx == 0 else '__nolabel__')

        # Plot acquisition functions
        for acqf_idx, (label, ei) in enumerate(acq_vals.items()):
            acqf_ax.plot(detach(X), detach((ei - ei.min()) / (ei.max() - ei.min())), label=label, alpha=0.9, linestyle='solid' if acqf_idx == 0 else 'dashed', linewidth=1 if acqf_idx == 0 else 0.9)

        # Optimize acquisition function
        next_X, _ = optimize_acqf(
            acq_func_to_use,
            bounds=objective.bounds,
            q=1,
            num_restarts=8,
            raw_samples=256,
        )
        init_X, init_y = make_new_data(init_X, init_y, next_X, objective)
        model = init_and_fit_model(init_X, init_y, objective.bounds)

    # Formatting
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])

    axes[0, -1].legend()
    axes[1, -1].legend()
    axes[1, 0].legend()
    axes[0, 1].set_ylabel(r'$\mu(x)$')
    axes[1, 1].set_ylabel(r'$\alpha(x)$')
    axes[0, 0].axis('off')

    return fig, axes

def colabo_robustness_plot(seed: int | None = 256) -> Tuple[plt.Figure, plt.Axes]:
    def plot_objective_function(
            ax: plt.Axes,
            objective = Shekel1D(negate=True),
            n_points: int = 100,
            offset: Tuple[float, float] = (-0.3, 0.),
            annotate_objective_optimum: bool = False,
        ):
        if objective.dim != 1: raise ValueError("objective needs to have dim=1")
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        y = objective(x.unsqueeze(-1))

        ax.plot(x, y, color="k", label=r"$f(x)$")

        x_opt = objective._optimizers[0][0]
        optimum_objective = (x_opt, objective(torch.tensor(x_opt, dtype=torch.float64)).item())
        ax.axvline(x_opt, color="lightgray", linestyle=":", label=r"$\max_x f(x)$")
        if annotate_objective_optimum:
            text_position = unnormalize(normalize(x_opt, objective.bounds) + torch.tensor(offset), objective.bounds)
            ax.annotate(r'$\max_x f(x)$', xy=optimum_objective, xytext=text_position, xycoords='data',
                        arrowprops=dict(facecolor='k', edgecolor='none', shrink=0.05, width=2, headwidth=10),
                        verticalalignment='center', horizontalalignment='center')
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$f(x)$")

    def plot_objective_vs_prior_func(
            ax: plt.Axes,
            objective = Shekel1D(negate=True),
            prior_predictor = Shekel1DNoGlobal(negate=True),
            prior_predictor_optimizers = Shekel1DNoGlobal(negate=True)._optimizers,
            n_points: int = 100,
            offset: Tuple[float, float] = (0.3, 0.),
            annotate_objective_optimum: bool = False,
            annoate_prior_optimum: bool = False,
        ):
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        y_obj = objective(x.unsqueeze(-1))
        y_prior = prior_predictor(x.unsqueeze(-1))

        x_opt = objective._optimizers[0][0]
        optimum_objective = (x_opt, objective(torch.tensor(x_opt, dtype=torch.float64)).item())
        ax.axvline(x_opt, color="lightgray", linestyle=":")  # label=r"$x^*$"
        if annotate_objective_optimum:
            text_position = unnormalize(normalize(x_opt, objective.bounds) + torch.tensor(offset), objective.bounds)
            ax.annotate(r'$\max_x f(x)$', xy=optimum_objective, xytext=text_position, color="lightgray",
                        arrowprops=dict(facecolor='lightgray', edgecolor='none', shrink=0.05, width=2, headwidth=10),
                        verticalalignment='center', horizontalalignment='center')
        
        x_prior_opt = prior_predictor_optimizers[0][0]
        optimum_prior = (x_prior_opt, prior_predictor(torch.tensor(x_prior_opt, dtype=torch.float64)).item())
        ax.axvline(x_prior_opt, color="k", linestyle=":")  # label=r"$\hat{x}^*$"
        if annoate_prior_optimum:
            text_position = unnormalize(normalize(x_prior_opt, objective.bounds) + torch.tensor(offset), objective.bounds)
            ax.annotate(r'$\max_x \hat{f}(x)$', xy=optimum_prior, xytext=text_position,
                        arrowprops=dict(facecolor='k', edgecolor='none', shrink=0.05, width=2, headwidth=10),
                        verticalalignment='center', horizontalalignment='center')
        
        ax.plot(x, y_obj, color="lightgray", linestyle="--", label=r"$f(x)$")
        ax.plot(x, y_prior, color="k", label=r"$\hat{f}(x)$")

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min,y_min + 1.3 * (y_max - y_min))

        ax.legend(loc="upper center", ncols=2, facecolor="white", framealpha=1.0, bbox_to_anchor=(0.5, 1.05))

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\hat{f}(x)$")

    def plot_prior(
            ax: plt.Axes,
            objective = Shekel1D(negate=True),
            prior_predictor = Shekel1DNoGlobal(negate=True),
            prior_predictor_optimizers = Shekel1DNoGlobal(negate=True)._optimizers,
            n_points: int = 100,
            temperature: float = 1.0,
            norm_width: float = 1.0,
        ):
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        y_pred_raw = prior_predictor(x.unsqueeze(-1)).squeeze()
        y_pred_norm = normalize(y_pred_raw, torch.stack([y_pred_raw.min(), y_pred_raw.max()])) * norm_width - norm_width/2
        y_pred_temp = y_pred_norm / temperature
        y_pred = torch.exp(y_pred_temp)
        y_pred_scaled = y_pred / torch.trapezoid(y_pred, x)

        y_obj_raw = objective(x.unsqueeze(-1)).squeeze()
        y_obj_norm = normalize(y_obj_raw, torch.stack([y_obj_raw.min(), y_obj_raw.max()])) * norm_width - norm_width/2
        y_obj_temp = y_obj_norm / temperature
        y_obj = torch.exp(y_obj_temp)
        y_obj_scaled = y_obj / torch.trapezoid(y_obj, x)

        ax.axvline(objective._optimizers[0][0], color="lightgray", linestyle=":")  # label=r"$x^*$"
        ax.axvline(prior_predictor_optimizers[0][0], color="k", linestyle=":")  # label=r"$\hat{x}^*$"

        ax.plot(x, y_obj_scaled, color="lightgray", linestyle="--", label=r"$\pi_f(x)$")
        ax.plot(x, y_pred_scaled, color="k", label=r"$\pi_{\hat{f}}(x)$")

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min,y_min + 1.3 * (y_max - y_min))

        ax.legend(loc="upper center", ncols=2, facecolor="white", framealpha=1.0, bbox_to_anchor=(0.5, 1.04))

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\pi(x)$")

    def plot_surrogate(
            ax: plt.Axes,
            ax_acq_func: plt.Axes | None = None,
            objective = Shekel1D(negate=True),
            prior_predictor = Shekel1DNoGlobal(negate=True),
            prior_predictor_optimizers = Shekel1DNoGlobal(negate=True)._optimizers,
            n_points: int = 100,
            temperature: float = 1.0,
            seed: int = 42,
            num_paths: int = 2**16,
            num_resampling_paths: int = 4096,
            max_num_paths_to_plot: int = 5,
            x_samples_normalized = [0.1, 0.4, 0.8],
            plot_acqf_max: bool = False,
        ):
        # Plot optima
        ax.axvline(objective._optimizers[0][0], color="lightgray", linestyle=":")  # label=r"$x^*$"
        ax.axvline(prior_predictor_optimizers[0][0], color="k", linestyle=":")  # label=r"$\hat{x}^*$"

        # Initialization samples
        init_x = unnormalize(torch.tensor(x_samples_normalized), objective.bounds)
        init_y = objective(init_x.unsqueeze(-1))

        # Plot surrogate
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        user_prior = ModelBasedPrior(bounds=objective.bounds, predict_func=prior_predictor, temperature=temperature, seed=seed, minimize=False)
        user_prior_obj = ModelBasedPrior(bounds=objective.bounds, predict_func=objective, temperature=temperature, seed=seed, minimize=False)
        model = init_and_fit_model(init_x.unsqueeze(dim=-1), init_y.unsqueeze(dim=-1), objective.bounds)
        paths = draw_matheron_paths(model=model, sample_shape=torch.Size([num_paths]))
        sampler = PathwiseSampler(sample_shape=torch.Size([num_paths]))
        acq_func = qPriorExpectedImprovement(
            model=model,
            paths=paths,
            sampler=sampler,
            X_baseline=init_x.unsqueeze(dim=-1),
            user_prior=user_prior,
            resampling_fraction=num_resampling_paths/num_paths,
            custom_decay=1.0,
        )
        acq_func_obj = qPriorExpectedImprovement(
            model=model,
            paths=paths,
            sampler=sampler,
            X_baseline=init_x.unsqueeze(dim=-1),
            user_prior=user_prior_obj,
            resampling_fraction=num_resampling_paths/num_paths,
            custom_decay=1.0,
        )
        acq_func_uniform = qPriorExpectedImprovement(
            model=model,
            paths=paths,
            sampler=sampler,
            X_baseline=init_x.unsqueeze(dim=-1),
            user_prior=None,
            resampling_fraction=num_resampling_paths/num_paths,
            custom_decay=1.0,
        )
        paths = acq_func.sampling_model.paths.paths
        paths_obj = acq_func_obj.sampling_model.paths.paths
        matheron_path = MatheronPath(paths.prior_paths, paths.update_paths)
        matheron_path_obj = MatheronPath(paths_obj.prior_paths, paths_obj.update_paths)
        prior_samples = matheron_path(x.unsqueeze(-1), subset=acq_func.indices)
        prior_samples_obj = matheron_path_obj(x.unsqueeze(-1), subset=acq_func_obj.indices)
        prior_mean, prior_std = prior_samples.mean(dim=0), prior_samples.std(dim=0)
        prior_mean_obj, prior_std_obj = prior_samples_obj.mean(dim=0), prior_samples_obj.std(dim=0)
        uniform_prior_samples = matheron_path(x.unsqueeze(-1))
        uniform_prior_mean, uniform_prior_std = uniform_prior_samples.mean(dim=0), uniform_prior_samples.std(dim=0)
        ax.plot(detach(x), detach(prior_mean_obj), label=r'$p(f \mid \mathcal{D}, \rho_f)$', linestyle="--", color='lightgray')
        ax.fill_between(detach(x), detach(prior_mean_obj - prior_std_obj), detach(prior_mean_obj + prior_std_obj), color='lightgray', alpha=0.1)
        ax.plot(detach(x), detach(uniform_prior_mean), label=r'$p(f \mid \mathcal{D})$', linestyle="dashdot", color='gray')
        ax.fill_between(detach(x), detach(uniform_prior_mean - uniform_prior_std), detach(uniform_prior_mean + uniform_prior_std), color='gray', alpha=0.1)
        ax.plot(detach(x), detach(prior_mean), label=r'$p(f \mid \mathcal{D}, \rho_{\hat{f}})$', color='k')
        ax.fill_between(detach(x), detach(prior_mean - prior_std), detach(prior_mean + prior_std), color='k', alpha=0.1)

        # Plot sample paths
        for path_idx in range(min(max_num_paths_to_plot, prior_samples.size(0))):
            ax.plot(detach(x), detach(prior_samples[path_idx]), color='gray', alpha=0.9, linewidth=0.3, label=r'$f_i \sim p(f \mid \mathcal{D}, \rho)$' if path_idx == 0 else '__nolabel__')

        # Plot samples
        ax.scatter(detach(unnormalize(acq_func.sampling_model.train_inputs[0], objective.bounds)), detach(acq_func.sampling_model.train_targets), color="k", s=8, label=r"$\mathcal{D}$", zorder=99)

        # ax.legend(loc="upper left")
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min,y_min + 1.4 * (y_max - y_min))

        ax.legend(loc="upper center", ncols=2, facecolor="white", framealpha=1.0, bbox_to_anchor=(0.5, 1.05))
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$f(x)$")

        if ax_acq_func is None: return

        # Plot acquisition function
        # Generate candidate points
        X_cand = torch.cat((init_x.unsqueeze(-1).unsqueeze(0).expand(x.size(0), -1, -1), x.view(x.size(0), 1, 1)), dim=1)
        with torch.no_grad():
            ei = acq_func(X_cand)
            ei_obj = acq_func_obj(X_cand)
            ei_norm = normalize(ei, torch.stack([ei.min(), ei.max()]))
            ei_obj_norm = normalize(ei_obj, torch.stack([ei_obj.min(), ei_obj.max()]))
            ei_uniform = acq_func_uniform(X_cand)
            ei_uniform_norm = normalize(ei_uniform, torch.stack([ei_uniform.min(), ei_uniform.max()]))

        ax_acq_func.axvline(objective._optimizers[0][0], color="lightgray", linestyle=":")  # label=r"$x^*$"
        ax_acq_func.axvline(prior_predictor_optimizers[0][0], color="k", linestyle=":")  # label=r"$\hat{x}^*$"

        ax_acq_func.plot(x, detach(ei_obj_norm), label=r"$\alpha_{\pi_f} (x)$", linestyle="--", color="lightgray")
        ax_acq_func.plot(x, detach(ei_uniform_norm), label=r"$\alpha(x)$", linestyle="dashdot", color="gray")
        ax_acq_func.plot(x, detach(ei_norm), label=r"$\alpha_{\pi_{\hat{f}}} (x)$", color="k")

        if plot_acqf_max:
            ax_acq_func.scatter(x[ei_norm.argmax()], ei_norm.max(), color="r", label=r"$\max_x \alpha_{\pi_{\hat{f}}} (x)$")

        ax_acq_func.legend(loc="upper left")

        y_min, y_max = ax_acq_func.get_ylim()
        ax_acq_func.set_ylim(y_min,y_min + 1.3 * (y_max - y_min))

        ax_acq_func.legend(loc="upper center", ncols=3, facecolor="white", framealpha=1.0, bbox_to_anchor=(0.5, 1.04), columnspacing=0.9)

        ax_acq_func.set_xlabel(r"$x$")
        ax_acq_func.set_ylabel(r"$\alpha(x)$")
    
    if seed is not None: torch.manual_seed(seed)
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    # ax_obj: plt.Axes = axes[0]
    ax_prior_func_vs_obj: plt.Axes = axes[0,0]
    ax_prior: plt.Axes = axes[0,1]
    ax_surrogate: plt.Axes = axes[1,0]
    ax_acqf: plt.Axes = axes[1,1]
    # objective = Sphere(dim=1, negate=True)
    # offset = -4.0
    # prior_predictor = lambda x: objective(x - offset)
    # prior_predictor_optimizers = [(offset,)]
    # x_samples_normalized = [0.1, 0.4, 0.8]
    objective = Shekel1D(negate=True)
    prior_predictor = Shekel1DNoGlobal(negate=True)
    prior_predictor_optimizers = prior_predictor._optimizers
    x_samples_normalized = [0.81, 0.09, 0.52, 0.79, 0.0, 1.0] # + torch.linspace(0, 1, 20).numpy().tolist()
    # plot_objective_function(ax_obj, objective=objective)
    plot_objective_vs_prior_func(ax_prior_func_vs_obj, objective=objective, prior_predictor=prior_predictor, prior_predictor_optimizers=prior_predictor_optimizers)
    temp = .1
    plot_prior(ax_prior, temperature=temp, objective=objective, prior_predictor=prior_predictor, prior_predictor_optimizers=prior_predictor_optimizers)
    num_paths = 2**16
    num_resampling_paths = 2**16
    max_num_paths_to_plot = 0
    plot_surrogate(ax_surrogate, ax_acq_func=ax_acqf, temperature=temp, objective=objective, prior_predictor=prior_predictor, prior_predictor_optimizers=prior_predictor_optimizers, x_samples_normalized=x_samples_normalized, num_paths=num_paths, num_resampling_paths=num_resampling_paths, max_num_paths_to_plot=max_num_paths_to_plot, seed=seed)

    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    # for ax in axes[:len(axes)-1]:
    #     ax: plt.Axes
    #     ax.set_xlabel("")

    return fig, axes


def pibo_normalization_plot() -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(2, 2, figsize=(6, 5.5), sharex=True)

    def plot_prior_predict_func(
            ax: plt.Axes,
            f_min_perc: float = 0.1,
            f_max_perc: float = 0.6,
            objective = Sphere(dim=1, negate=True),
            n_points: int = 100,
            x_samples_normalized = [0.01, 0.4, 0.99],
        ):
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        y: torch.Tensor = objective(x.unsqueeze(-1))

        f_min_hat = y.quantile(f_min_perc)
        f_max_hat = y.quantile(f_max_perc)

        ax.axhline(f_min_hat, linestyle=":", color="lightgray", label=r"$[\hat{f}_{min}, \hat{f}_{max}]$")
        ax.axhline(f_max_hat, linestyle=":", color="lightgray")
        ax.fill_between(x, f_min_hat, f_max_hat, color='lightgray', alpha=0.1)

        ax.plot(x, y, color="k")

        # Plot samples
        init_x = unnormalize(torch.tensor(x_samples_normalized), objective.bounds)
        init_y = objective(init_x.unsqueeze(-1))
        ax.scatter(init_x, init_y, color="k", s=10, label=r"$\mathcal{D}$")

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_min + (y_max-y_min) * 1.35)
        ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.99))

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\hat{f}(x)$")

    def plot_normalized_prior_predict_func(
            ax: plt.Axes,
            f_min_perc: float = 0.1,
            f_max_perc: float = 0.6,
            objective = Sphere(dim=1, negate=True),
            n_points: int = 100,
        ):
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        y: torch.Tensor = objective(x.unsqueeze(-1))

        f_min_hat = y.quantile(f_min_perc)
        f_max_hat = y.quantile(f_max_perc)

        y_norm: torch.Tensor = normalize(objective(x.unsqueeze(-1)), torch.stack([y.min(), y.max()]))
        y_hat_norm: torch.Tensor = normalize(objective(x.unsqueeze(-1)), torch.stack([f_min_hat, f_max_hat]))

        ax.plot(x, y_norm, color="lightgray", linestyle=":", label=r"$\hat{f}_{\text{norm}}(x)$ (accurate)")
        ax.plot(x, y_hat_norm, color="k", label=r"$\hat{f}_{\text{norm}}(x)$ (inaccurate)")

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_min + (y_max-y_min) * 1.35)
        ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.04))

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\hat{f}_{\text{norm}}(x)$")

    def calculate_stable_prob(y_normalized, temp, x_coords, eps=1e-10):
            log_p_unnorm = y_normalized / temp
            log_p_max = torch.max(log_p_unnorm)

            # Compute log of trapezoidal integral stably
            # log(integral(exp(log_p_unnorm) dx)) = log_p_max + log(integral(exp(log_p_unnorm - log_p_max) dx))
            integrand_stable = torch.exp(log_p_unnorm - log_p_max)
            integral_stable = torch.trapezoid(integrand_stable, x_coords)

            # Add eps for stability before taking log
            log_norm_const = log_p_max + torch.log(integral_stable + eps)

            # Compute final log probability and probability
            log_p = log_p_unnorm - log_norm_const
            p = torch.exp(log_p)
            return p

    def plot_prior(
            ax: plt.Axes,
            f_min_perc: float = 0.1,
            f_max_perc: float = 0.6,
            objective = Sphere(dim=1, negate=True),
            n_points: int = 100,
            temperature: float = 1.0,
            norm_width: float = 10.0,
        ):
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        y: torch.Tensor = objective(x.unsqueeze(-1))

        f_min_hat = y.quantile(f_min_perc)
        f_max_hat = y.quantile(f_max_perc)

        # Ensure bounds for normalization are tensors
        actual_bounds = torch.stack([y.min(), y.max()])
        estimated_bounds = torch.stack([f_min_hat, f_max_hat])
        if y.dim() == 1:
             actual_bounds = actual_bounds.squeeze()
             estimated_bounds = estimated_bounds.squeeze()

        # Normalize y to [0, 1] range first, then scale and shift
        y_norm_01 = normalize(y, actual_bounds)
        y_norm = y_norm_01 * norm_width - norm_width / 2

        y_hat_norm_01 = normalize(y, estimated_bounds)
        y_hat_norm = y_hat_norm_01 * norm_width - norm_width / 2

        # Calculate probabilities using the stable method
        y_obj_scaled = calculate_stable_prob(y_norm, temperature, x)
        y_hat_scaled = calculate_stable_prob(y_hat_norm, temperature, x)

        ax.plot(x, y_obj_scaled, color="lightgray", linestyle=":", label=r"$\pi_{\hat{f}_{\text{norm}}}(x)$ (acc.)")
        ax.plot(x, y_hat_scaled, color="k", label=r"$\pi_{\hat{f}_{\text{norm}}}(x)$ (inacc.)")

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_min + (y_max-y_min) * 1.35)
        ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.04))

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\pi(x)$")

    def plot_acqf_func(
            ax: plt.Axes,
            f_min_perc: float = 0.1,
            f_max_perc: float = 0.6,
            objective = Sphere(dim=1, negate=True),
            n_points: int = 100,
            temperature: float = 1.0,
            norm_width: float = 10.0,
            x_samples_normalized = [0.01, 0.4, 0.99],
            plot_samples: bool = False,
        ):
        x = unnormalize(torch.linspace(0, 1, n_points), objective.bounds)
        y: torch.Tensor = objective(x.unsqueeze(-1))

        f_min_hat = y.quantile(f_min_perc)
        f_max_hat = y.quantile(f_max_perc)

        # Ensure bounds for normalization are tensors
        actual_bounds = torch.stack([y.min(), y.max()])
        estimated_bounds = torch.stack([f_min_hat, f_max_hat])
        if y.dim() == 1:
             actual_bounds = actual_bounds.squeeze()
             estimated_bounds = estimated_bounds.squeeze()

        # Normalize y to [0, 1] range first, then scale and shift
        y_norm_01 = normalize(y, actual_bounds)
        y_norm = y_norm_01 * norm_width - norm_width / 2

        y_hat_norm_01 = normalize(y, estimated_bounds)
        y_hat_norm = y_hat_norm_01 * norm_width - norm_width / 2

        # Calculate probabilities using the stable method
        y_obj_scaled = calculate_stable_prob(y_norm, temperature, x)
        y_hat_scaled = calculate_stable_prob(y_hat_norm, temperature, x)
        
        init_x = unnormalize(torch.tensor(x_samples_normalized), objective.bounds)
        init_y = objective(init_x.unsqueeze(-1))

        model = init_and_fit_model(init_x.unsqueeze(dim=-1), init_y.unsqueeze(dim=-1), objective.bounds)
        acq_func = ExpectedImprovement(model=model, best_f=init_y.max())
        with torch.no_grad():
            ei = acq_func(x.unsqueeze(-1).unsqueeze(-1))
            ei_obj = ei * y_obj_scaled
            ei_hat = ei_obj * y_hat_scaled
            ei_obj_norm = normalize(ei_obj, torch.stack([ei_obj.min(), ei_obj.max()]))
            ei_hat_norm = normalize(ei_hat, torch.stack([ei_hat.min(), ei_hat.max()]))

        if plot_samples:
            for x_idx, x_sample in enumerate(init_x):
                ax.axvline(x_sample, color="gray", linestyle="--", linewidth=0.7, label=r"$\mathcal{D}$" if x_idx == 0 else "__nolabel__")

        ax.plot(x, detach(ei_obj_norm), color="lightgray", linestyle=":", label=r"$\alpha_{\pi_{\hat{f}_{\text{norm}}}}(x)$ (acc.)")
        ax.plot(x, detach(ei_hat_norm), color="k", label=r"$\alpha_{\pi_{\hat{f}_{\text{norm}}}}(x)$ (inacc.)")

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_min + (y_max-y_min) * 1.35)
        ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.04))

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\alpha(x)$")

    temperature = 5.0
    # x_samples_normalized = [0.01, 0.99]
    objective = Shekel1DNoGlobal(negate=True)
    f_min_perc: float = 0.12
    f_max_perc: float = 0.8
    x_samples_normalized = [0.01, 0.34, 0.46, 0.6, 0.8, 0.99] # + torch.linspace(0, 1, 20).numpy().tolist()
    plot_prior_predict_func(axes[0,0], x_samples_normalized=x_samples_normalized, objective=objective, f_min_perc=f_min_perc, f_max_perc=f_max_perc)
    plot_normalized_prior_predict_func(axes[0,1], objective=objective, f_min_perc=f_min_perc, f_max_perc=f_max_perc)
    plot_prior(axes[1,0], temperature=temperature, objective=objective, f_min_perc=f_min_perc, f_max_perc=f_max_perc)
    plot_acqf_func(axes[1,1], temperature=temperature, x_samples_normalized=x_samples_normalized, objective=objective, f_min_perc=f_min_perc, f_max_perc=f_max_perc)

    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    # for ax in axes[:len(axes)-1]:
    #     ax: plt.Axes
    #     ax.set_xlabel("")

    fig.tight_layout(h_pad=2.0)

    return fig, axes


def pibo_acquisition_plot() -> Tuple[plt.Figure, plt.Axes]:
    num_iterations = 5
    temperature = 1.0
    baseline_temperatures = [0.1]
    prior_offset = 4.0 # 0.05 or 4.0
    seed = 42
    beta = 1.0
    surrogate_color = 'tab:blue'
    num_priors = 2 + len(baseline_temperatures)
    prior_colors = cm.plasma(torch.linspace(0.2, 0.8, num_priors)) # Use a subset of the colormap to avoid very light/dark ends
    
    # Setup
    torch.manual_seed(seed)
    objective = Sphere(dim=1, negate=True)
    fig, axes = plt.subplots(3+len(baseline_temperatures), num_iterations + 1, figsize=((num_iterations + 1) * 1.5, 6), sharex=True)
    X = torch.linspace(*detach(objective.bounds.T[0]), 101)

    # Priors
    predict_func = lambda x: objective(x - prior_offset)
    user_prior = ModelBasedPrior(bounds=objective.bounds, predict_func=predict_func, temperature=temperature, seed=seed, minimize=False)
    baseline_user_priors = {
        f"{temp}": ModelBasedPrior(bounds=objective.bounds, predict_func=predict_func, temperature=temp, seed=seed, minimize=False)
        for temp in baseline_temperatures
    }
    
    # Initial model
    init_X, init_y = generate_data_from_prior(objective=objective, user_prior=user_prior, n=1)
    model = init_and_fit_model(init_X, init_y, objective.bounds)

    # Plot priors
    uniform_prior_ax = axes[1,0]
    uniform_prior_ax.plot(detach(X), detach(torch.ones_like(X)), label=fr'Uniform', color='k')
    default_prior_ax = axes[2,0]
    y_default = torch.exp(user_prior.evaluate(normalize_X(X, objective.bounds))).squeeze()
    default_prior_ax.plot(detach(X), detach(y_default), label=fr'$T={temperature}$', color='k')
    y_baselines = {}
    for i, (temp, prior) in enumerate(baseline_user_priors.items()):
        y_baseline = torch.exp(prior.evaluate(normalize_X(X, objective.bounds))).squeeze()
        axes[3+i,0].plot(detach(X), detach(y_baseline), label=fr'$T={temp}$', color='k')
        y_baselines[f"{temp}"] = y_baseline

    # Optimization iterations
    for iteration in range(1, num_iterations+1):

        # Plot surrogate
        model.eval()
        posterior = model.posterior(X.unsqueeze(dim=-1))
        lower, upper = posterior.mvn.confidence_region()

        posterior_ax = axes[0,iteration]
        posterior_ax.plot(detach(X), detach(posterior.mean), color=surrogate_color)
        posterior_ax.fill_between(detach(X), detach(lower), detach(upper), color=surrogate_color, alpha=0.1)
        posterior_ax.plot(detach(init_X), detach(init_y), color='k', marker='*', linestyle='None', alpha=0.8)

        # Plot acquisition function
        acq_func = ExpectedImprovement(model=model, best_f=init_y.max())
        with torch.no_grad():
            ei = acq_func(X.unsqueeze(-1).unsqueeze(-1))
            axes[1,iteration].plot(detach(X), detach(ei), color=prior_colors[0])
            ei_default = ei * (y_default ** (beta / iteration))
            axes[2,iteration].plot(detach(X), detach(ei_default), color=prior_colors[1])
            for i, (temp, y_baseline) in enumerate(y_baselines.items()):
                axes[3+i,iteration].plot(detach(X), detach(ei * (y_baseline ** (beta / iteration))), color=prior_colors[2+i])

        # Find and plot max
        next_X = X[torch.argmax(ei_default)].unsqueeze(0).unsqueeze(0)
        # axes[2,iteration].plot(detach(next_X.squeeze()), detach(torch.max(ei_default)), 'ko')
        axes[2,iteration].axvline(detach(next_X.squeeze()), linestyle='--', linewidth=0.8, color='k', alpha=0.3)

        # Train model with new data
        model.train()
        init_X, init_y = make_new_data(init_X, init_y, next_X, objective)
        model = init_and_fit_model(init_X, init_y, objective.bounds)

    # Formatting
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])

    for ax in axes[1:,0]:
        ax.legend(loc="upper left")
        ax.set_ylabel(r'$\pi(x)$')

    for ax in axes[1:,1]:
        ax.set_ylabel(r'$\alpha(x)$')

    for ax in axes[-1,:]:
        ax.set_xlabel(r'$x$')

    axes[0,1].set_ylabel(r'$f(x)$')
    # axes[0,1].legend(["Mean", "Confidence", "Observed"])

    for i, ax in enumerate(axes[0,1:]):
        ax.set_title(f"Iteration {i+1}")

    y_lim_low, y_lim_high = zip(*[ax.get_ylim() for ax in axes[0,1:]])
    for ax in axes[0,:]:
        ax.set_ylim(min(y_lim_low), max(y_lim_high))

    for ax in axes[:,0]:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_min + (y_max - y_min) * 1.5)

    axes[0,0].axis('off')

    return fig, axes


def colabo_acquisition_plot() -> Tuple[plt.Figure, plt.Axes]:
    # --- Configuration ---
    num_iterations = 5
    temperature = 1.0
    baseline_temperatures = [0.1]
    prior_offset = 4.0
    seed = 42
    surrogate_color = 'tab:blue'
    num_paths = 2048
    num_resampling_paths = 512
    
    # --- Setup ---
    torch.manual_seed(seed)
    objective = Sphere(dim=1, negate=True)
    num_priors = 2 + len(baseline_temperatures)
    prior_colors = cm.plasma(torch.linspace(0.2, 0.8, num_priors))
    
    fig, axes = plt.subplots(
        num_priors * 2, 
        num_iterations + 1, 
        figsize=((num_iterations + 1) * 1.5, 8), 
        sharex=True
    )
    X_grid = torch.linspace(*detach(objective.bounds.T[0]), 101)

    # --- Priors Definition ---
    predict_func = lambda x: objective(x - prior_offset)
    default_prior = ModelBasedPrior(
        bounds=objective.bounds, predict_func=predict_func, temperature=temperature, seed=seed, minimize=False
    )
    baseline_priors = {
        temp: ModelBasedPrior(
            bounds=objective.bounds, predict_func=predict_func, temperature=temp, seed=seed, minimize=False
        ) for temp in baseline_temperatures
    }
    
    priors_config = [
        {'name': 'Uniform', 'prior': None, 'color': prior_colors[0], 'row_idx': 0},
        {'name': fr'$T={temperature}$', 'prior': default_prior, 'color': prior_colors[1], 'row_idx': 2},
    ]
    for i, temp in enumerate(baseline_temperatures):
        priors_config.append(
            {'name': fr'$T={temp}$', 'prior': baseline_priors[temp], 'color': prior_colors[2+i], 'row_idx': 4+2*i}
        )

    # --- Initial State (Column 0) ---
    axes[0, 0].plot(detach(X_grid), detach(torch.ones_like(X_grid)), color='k', label='Uniform')
    for config in priors_config[1:]: # Skip uniform
        prior_density = torch.exp(config['prior'].evaluate(normalize_X(X_grid, objective.bounds))).squeeze()
        axes[config['row_idx'], 0].plot(detach(X_grid), detach(prior_density), color='k', label=config['name'])

    # --- Initial Data ---
    train_X, train_y = generate_data_from_prior(objective=objective, user_prior=default_prior, n=1)

    # --- Optimization Loop ---
    for i in range(1, num_iterations + 1):
        model = init_and_fit_model(train_X, train_y, objective.bounds)
        paths = draw_matheron_paths(model=model, sample_shape=torch.Size([num_paths]))
        sampler = PathwiseSampler(sample_shape=torch.Size([num_paths]), seed=seed)

        acq_funcs = {}
        for config in priors_config:
            acq_funcs[config['name']] = qPriorExpectedImprovement(
                model=model,
                paths=paths,
                sampler=sampler,
                X_baseline=train_X,
                user_prior=config['prior'],
                resampling_fraction=num_resampling_paths / num_paths,
                custom_decay=1.0,
            )
        
        # --- Plot Surrogates and Acquisition Functions ---
        with torch.no_grad():
            for config in priors_config:
                row_idx = config['row_idx']
                acqf = acq_funcs[config['name']]
                
                # Plot Surrogate (Resampled Posterior)
                ax_surrogate = axes[row_idx, i]
                matheron_path = MatheronPath(acqf.sampling_model.paths.paths.prior_paths, acqf.sampling_model.paths.paths.update_paths)

                samples = matheron_path(X_grid.unsqueeze(-1), subset=acqf.indices if config['name'] != 'Uniform' else None)
                mean, std = samples.mean(dim=0), samples.std(dim=0)
                
                ax_surrogate.plot(detach(X_grid), detach(mean), color=surrogate_color)
                ax_surrogate.fill_between(detach(X_grid), detach(mean - std), detach(mean + std), color=surrogate_color, alpha=0.1)
                ax_surrogate.plot(detach(train_X), detach(acqf.sampling_model.train_targets), 'k*', linestyle='None')

                # Plot Acquisition Function
                ax_acqf = axes[row_idx + 1, i]
                acqf_vals = acqf(X_grid.view(-1, 1, 1))
                ax_acqf.plot(detach(X_grid), detach(acqf_vals), color=config['color'])
        
        # --- Select Next Point ---
        default_acqf = acq_funcs[priors_config[1]['name']] # Based on T=1.0
        default_acqf_vals = default_acqf(X_grid.view(-1, 1, 1))
        next_x_val = X_grid[default_acqf_vals.argmax()]
        next_X = next_x_val.unsqueeze(0).unsqueeze(0)
        
        # Mark chosen point on its acquisition function plot
        axes[priors_config[1]['row_idx'] + 1, i].axvline(
            detach(next_X.squeeze()), linestyle='--', linewidth=1.0, color='k', alpha=0.5
        )

        # --- Update Data for Next Iteration ---
        train_X, train_y = make_new_data(train_X, train_y, next_X, objective)

    # Formatting
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])

    for ax in axes[::2,0]:
        ax.legend(loc="upper left")
        ax.set_ylabel(r'$\pi(x)$')
        ax.set_xlabel(r'$x$')

    for ax in axes[::2, 1]:
        ax.set_ylabel(r'$f(x)$')

    for ax in axes[1::2, 1]:
        ax.set_ylabel(r'$\alpha(x)$')

    for ax in axes[-1,:]:
        ax.set_xlabel(r'$x$')

    for i, ax in enumerate(axes[0,1:]):
        ax.set_title(f"Iteration {i+1}")

    for row_idx in range(0,len(axes),2):
        surrogate_axes = axes[row_idx,1:]
        y_lim_low, y_lim_high = zip(*[ax.get_ylim() for ax in surrogate_axes])
        for ax in surrogate_axes:
            ax.set_ylim(min(y_lim_low), max(y_lim_high))

    for ax in axes[:,0]:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_min + (y_max - y_min) * 1.6)

    for ax in axes[1::2,0]:
        ax.axis('off')

    return fig, axes


def main():
    """Create the illustrative plots for the paper."""
    plots_dir = os.getenv('PLOTS_DIR')

    for plot_func, filename in [
        # (sphere_plot, 'sphere.png'),
        # (shekel_plot, 'shekel.png'),
        # (scatter_quality_plot, 'scatter_quality.png'),
        # (prior_temperature_plot, 'prior_temperature.png'),
        # (initial_samples_plot, 'initial_samples.png'),
        # (colabo_robustness_plot, 'colabo_robustness.png'),
        # (pibo_normalization_plot, 'pibo_normalization.png'),
        # (pibo_acquisition_plot, 'pibo_acquisition.png'),
        (colabo_acquisition_plot, 'colabo_acquisition.png'),
    ]:
        fig, ax = plot_func()
        fig.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        fig.tight_layout()

    # plt.show()

if __name__ == '__main__':
    main()