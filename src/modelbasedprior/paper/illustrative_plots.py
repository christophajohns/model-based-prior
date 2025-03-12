"""Create the illustrative plots for the paper."""

from typing import Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from botorch.utils.prior import normalize, unnormalize
from botorch.sampling.pathwise import MatheronPath, draw_matheron_paths
from botorch.sampling.pathwise_sampler import PathwiseSampler
from botorch.acquisition.prior_monte_carlo import qPriorExpectedImprovement
from botorch.optim import optimize_acqf
from dotenv import load_dotenv
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from modelbasedprior.optimization.bo import generate_data_from_prior, init_and_fit_model, make_new_data
from modelbasedprior.objectives.sphere import Sphere
from modelbasedprior.objectives.shekel import Shekel2D, Shekel2DNoGlobal, ShekelNoGlobal
from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss
from modelbasedprior.objectives.scatterplot_quality import ScatterPlotQualityLoss
from modelbasedprior.objectives.mr_layout_quality import MRLayoutQualityLoss
from modelbasedprior.prior import ModelBasedPrior

load_dotenv()

def sphere_plot() -> Tuple[plt.Figure, plt.Axes]:
    """Create an illustration of the Sphere objective
    and the prior belief as the shifted Sphere objective."""
    fig, ax = plt.subplots()

    # Create a grid
    x_limits = (-0.75, 1.2)
    x1 = x2 = torch.linspace(*x_limits, 100)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    X = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    # Set contour levels
    levels = torch.linspace(0, 0.4, 6)

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
    ax.text(*optimum_objective, r'$x^*$', verticalalignment='top', horizontalalignment='center')
    ax.text(*optimum_prior, r'$x^*_{\text{prior}}$', verticalalignment='bottom', horizontalalignment='center')

    # Denote the distance between the minima
    # and label it as delta
    ax.annotate('', xy=optimum_objective, xytext=optimum_prior,
                arrowprops=dict(arrowstyle='<->', lw=1), zorder=1)
    ax.text(0.25, 0.25, r'$\Delta$', verticalalignment='center', horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=1)

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
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

    shekel_function = Shekel2D(negate=True)

    # Create a grid
    x_limits, y_limits = shekel_function.bounds.T
    resolution = 100
    x1 = torch.linspace(*x_limits, resolution)
    x2 = torch.linspace(*y_limits, resolution)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    X = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    # Set contour levels
    levels = torch.linspace(0, 11, 20)

    # Create a contour plot of the Shekel objective
    Y = shekel_function(X).reshape(X1.shape)
    contours = axes[0].contour(X1, X2, Y, levels=levels, colors='black')
    # axes[0].clabel(contours, inline=True, fontsize=8)

    # Create a contour plot of the prior belief
    # as Shekel(x) without the global minimum
    shekel_function_no_global = Shekel2DNoGlobal(negate=True)
    Y_no_global = shekel_function_no_global(X).reshape(X1.shape)
    contours_prior = axes[1].contour(X1, X2, Y_no_global, levels=levels, colors='grey', linestyles='dashed')
    # axes[1].clabel(contours_prior, inline=True, fontsize=8)

    # Add the global minimum
    optimum = shekel_function.optimizers[0]
    for ax in axes:
        ax.plot(*optimum, 'ro', label=r'x^*', markersize=4)

    # Label the global minimum
    def annotate_optimum(ax, offset = (-1.7, -1.7)):
        text_position = (optimum[0] + offset[0], optimum[1] + offset[1])
        ax.annotate(r'$x^*$', xy=optimum, xytext=text_position,
                    arrowprops=dict(facecolor='red', edgecolor='none', shrink=0.05, width=2, headwidth=10),
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
    # axes[1].set_xlabel(r'$x_1$')
    # axes[1].set_ylabel(r'$x_2$')

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
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, label in zip(axes[:2], ['High objective value', 'Low objective value']):
        ax.set_title(label)

    for ax, label in zip(axes[2:], ['Low overplotting', 'High overplotting']):
        ax.set_title(label)

    # Load the Cars dataset
    df = sns.load_dataset('mpg')  # Load Cars dataset from seaborn
    x_data = torch.tensor(df['horsepower'].values, dtype=torch.float32)
    y_data = torch.tensor(df['mpg'].values, dtype=torch.float32)

    # Label the axes
    for ax in axes:
        ax.set_xlabel('horsepower')
        ax.set_ylabel('mpg')

    # Remove the box
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add the scatter plot
    good_marker_size = 15
    bad_marker_size = 50
    good_marker_opacity = 40
    bad_marker_opacity = 240
    axes[0].scatter(x_data, y_data, s=good_marker_size, c='black', alpha=good_marker_opacity / 255.0)
    axes[1].scatter(x_data, y_data, s=bad_marker_size, c='black', alpha=bad_marker_opacity / 255.0)
    axes[1].set_xlim(0, 750)
    axes[2].scatter(x_data, y_data, s=200, c='black', alpha=0.05)
    axes[3].scatter(x_data, y_data, s=200, c='black', alpha=0.95)

    # Evalute the ScatterPlotQualityLoss
    # scatter_quality_loss = ScatterPlotQualityLoss(x_data=x_data, y_data=y_data, negate=True)
    # ratings = [scatter_quality_loss(torch.tensor([marker_size, marker_opacity, 1.0]))
    #            for marker_size, marker_opacity in [(good_marker_size, good_marker_opacity),
    #                                                 (bad_marker_size, bad_marker_opacity)]]
    # print([rating.item() for rating in ratings])

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
        ('Image Similarity', image_similarity, image_similarity.bounds),
        ('MR Layout Quality', mr_layout_quality, mr_layout_quality.bounds),
        ('Scatter Plot Quality', scatterplot_quality, scatterplot_quality.bounds),
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
    fig, axes = plt.subplots(len(transformations) + 1, len(prior_funcs) + 1, figsize=((len(prior_funcs) + 1) * 2, (len(transformations) + 1) * 2), layout='constrained')

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
        ax.plot(x[:, 0], y_pred, label="Prior Function")
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
        ax.plot(x, y, label='__nolabel__')

        for temperature in temperatures:
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
                ax.plot(x_1d, y_scaled, label=fr'$T = {temperature}$')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Remove y-axis ticks
                ax.set_yticks([])

    for ax, (prior_name, _prior_func, _bounds) in zip(axes[0, 1:], prior_funcs):  # Only the first row
        ax.set_title(prior_name)
    
    for ax in axes[-1][1:]:  # Only the last row
        ax.set_xlabel(r'$x_1 \mid x_i = \frac{x_i^\text{min} + x_i^\text{max}}{2} \forall i > 1$')
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

    # Helper functions
    def detach(tensor):
        return tensor.detach().cpu().numpy()
    
    def normalize_X(X, bounds):
        return normalize(X.unsqueeze(-1), bounds)
    
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


def main():
    """Create the illustrative plots for the paper."""
    for plot_func, filename in [
        # (sphere_plot, 'sphere.png'),
        # (shekel_plot, 'shekel.png'),
        # (scatter_quality_plot, 'scatter_quality.png'),
        (prior_temperature_plot, 'prior_temperature.png'),
        # (initial_samples_plot, 'initial_samples.png'),
    ]:
        fig, ax = plot_func()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()