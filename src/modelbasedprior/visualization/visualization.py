import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from scipy.stats import linregress

import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import warnings
from gpytorch.utils.warnings import GPInputWarning

from typing import List, Tuple, Dict, Any


plt.rcParams["savefig.bbox"] = 'tight'

def get_log10_regret_over_iteration_fig(df: pd.DataFrame, *args, **kwargs) -> go.Figure:
    """
    Generate a Plotly figure showing the log10 regret over iterations for a single experiment.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing columns for iteration, log10_regret, and other experiment data.
    *args, **kwargs : 
        Additional arguments and keyword arguments passed to the Plotly `px.line` function.

    Returns:
    -------
    go.Figure
        A Plotly figure displaying the log10 regret over iterations.

    Examples:
    --------
    This example generates a synthetic DataFrame for testing and plots the log10 regret over iterations.

    >>> import pandas as pd
    >>> import numpy as np
    >>> from analysis import get_df
    >>> X = np.array([[0.1, 0.2], [0.4, 0.5], [0.6, 0.7]])
    >>> y = np.array([0.2, 0.6, 0.8])
    >>> optimal_value = 1.0
    >>> df = get_df(X, y, optimal_value)
    >>> fig = get_log10_regret_over_iteration_fig(df)
    >>> fig.show()  # Manually inspect the plot

    Expected behavior:
    - The plot should display the log10 regret decreasing over 3 iterations.
    - Each iteration should be clearly marked with a point.
    """
    title = kwargs.pop("title", "Bayesian Optimization with Prior over the Optimum")
    return px.line(
        data_frame=df,
        x="iteration",
        y="log10_regret",
        hover_data=df.columns,
        title=title,
        labels={
            "log10_regret": r"$\text{Log10 Regret }\log_{10}(f(\hat{x}_i) - f(x^*))$",
            "iteration": r"$\text{Iteration } i$",
        },
        markers=True,
        *args,
        **kwargs,
    )

def _generate_iqr_traces(
    summary: pd.DataFrame,
    name: str,
    color: str,
    light_color: str
) -> List[go.Scatter]:
    """
    Generate IQR traces for a single optimization approach.
    
    Parameters:
    -----------
    summary : pd.DataFrame
        DataFrame containing iteration, median, q25, and q75 columns
    name : str
        Name of the optimization approach
    color : str
        Color for the median line
    light_color : str
        Color for the IQR shaded area
    
    Returns:
    --------
    List[go.Scatter]
        List of three traces: median line, upper bound, and lower bound
    """
    return [
        go.Scatter(
            x=summary["iteration"],
            y=summary["median"],
            mode='lines+markers',
            name=f'{name} (Median)',
            line=dict(color=color),
            marker=dict(color=color)
        ),
        go.Scatter(
            x=summary["iteration"],
            y=summary["q75"],
            fill=None,
            mode='lines',
            line=dict(color=light_color),
            name=f'{name} (75th Percentile)'
        ),
        go.Scatter(
            x=summary["iteration"],
            y=summary["q25"],
            fill='tonexty',
            mode='lines',
            line=dict(color=light_color),
            name=f'{name} (25th Percentile)'
        )
    ]

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for log10 regret grouped by iteration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing iteration and log10_regret columns
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics including median, 25th, and 75th percentiles
    """
    return df.groupby("iteration")["log10_regret"].agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).reset_index()

def get_log10_regret_over_iteration_fig_iqr(
    df: pd.DataFrame,
    *args,
    **kwargs
) -> go.Figure:
    """
    Generate a Plotly figure showing the median and IQR of log10 regret over iterations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing columns for iteration, log10_regret, and experiment ID.
    *args, **kwargs :
        Additional arguments and keyword arguments passed to the Plotly `go.Figure` function.
    
    Returns:
    --------
    go.Figure
        A Plotly figure displaying the median log10 regret and IQR.
    """
    title = kwargs.pop("title", "Bayesian Optimization - Median and IQR of Log10 Regret")
    summary = get_summary_stats(df)
    
    fig = go.Figure()
    traces = _generate_iqr_traces(summary, "Optimization", "blue", "lightblue")
    for trace in traces:
        fig.add_trace(trace)
    
    fig.update_layout(
        title=title,
        xaxis_title=r"$\text{Iteration } i$",
        yaxis_title=r"$\text{Log10 Regret } \log_{10}(f(\hat{x}_i) - f(x^*))$",
        hovermode="x unified",
        *args,
        **kwargs
    )
    return fig

def get_comparative_log10_regret_fig_iqr(
    dfs: List[pd.DataFrame],
    names: List[str],
    colors: List[Tuple[str, str]],
    num_initial_iterations: int | None = None,
    *args,
    **kwargs
) -> go.Figure:
    """
    Generate a comparative Plotly figure showing multiple IQR traces.
    
    Parameters:
    -----------
    dfs : List[pd.DataFrame]
        List of DataFrames, each containing data for a different optimization approach
    names : List[str]
        List of names for each optimization approach
    colors : List[Tuple[str, str]]
        List of color tuples (main_color, light_color) for each approach
    *args, **kwargs :
        Additional arguments and keyword arguments passed to the Plotly `go.Figure` function.
    
    Returns:
    --------
    go.Figure
        A Plotly figure displaying multiple IQR traces for comparison.
    
    Example:
    --------
    >>> df1 = get_optimization_results_1()
    >>> df2 = get_optimization_results_2()
    >>> fig = get_comparative_log10_regret_fig_iqr(
    ...     dfs=[df1, df2],
    ...     names=["Method A", "Method B"],
    ...     colors=[("blue", "lightblue"), ("red", "lightpink")]
    ... )
    >>> fig.show()
    """
    title = kwargs.pop("title", "Comparison of Bayesian Optimization Approaches")
    
    fig = go.Figure()
    
    for df, name, (color, light_color) in zip(dfs, names, colors):
        summary = get_summary_stats(df)
        traces = _generate_iqr_traces(summary, name, color, light_color)
        for trace in traces:
            fig.add_trace(trace)

    if num_initial_iterations is not None:
        fig.add_vline(x=num_initial_iterations, line_dash="dash", line_color="gray", annotation_text="Initial Iterations")
    
    fig.update_layout(
        title=title,
        xaxis_title=r"$\text{Iteration } i$",
        yaxis_title=r"$\text{Log10 Regret } \log_{10}(f(\hat{x}_i) - f(x^*))$",
        hovermode="x unified",
        *args,
        **kwargs
    )
    return fig

def show_images(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def pad_images(images):
    max_width = max(img.size(2) for img in images)
    max_height = max(img.size(1) for img in images)
    return [F.pad(img, (0, 0, max_width - img.size(2), max_height - img.size(1))) for img in images]

def get_utility_vs_latent_utility_fig_pbo(utility_func: SyntheticTestFunction, pairwise_gaussian_process: PairwiseGP, grid_resolution: int = 50) -> go.Figure:
    # Plot the utility function and latent utility function (i.e., GP posterior mean + 95% CI)
    # in two subplots on top of each other; if 2D, plot the utility function, the mean of the latent utility function,
    # and the std of the latent utility function as contour plots on top of each other
    x = torch.linspace(*utility_func.bounds.T[0], grid_resolution)

    if utility_func.dim == 1:
        # Create the utility function plot
        x = x.view(-1, 1)
        y = utility_func(x).detach().numpy()

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Utility Function", "Latent Utility Function"))
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y.squeeze(), mode="lines", name="Utility Function"), row=1, col=1)

        # Create the latent utility function plot
        with torch.no_grad():
            multivariate_normal = pairwise_gaussian_process(x)
            y_mean = multivariate_normal.mean.detach().numpy()
            y_std = multivariate_normal.variance.sqrt().detach().numpy()
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y_mean.squeeze(), mode="lines", name="Latent Utility Function Mean"), row=1, col=2)
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y_mean.squeeze() + 1.96 * y_std.squeeze(), mode="lines", line=dict(dash="dash", color="gray"), name="Latent Utility Function 95% CI"), row=1, col=2)
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y_mean.squeeze() - 1.96 * y_std.squeeze(), mode="lines", line=dict(dash="dash", color="gray"), showlegend=False, fill="tonexty", fillcolor="rgba(0,0,0,0.1)"), row=1, col=2)

        # Add the evaluated points to each subplot
        x_samples = pairwise_gaussian_process.train_inputs[0].squeeze().numpy()
        y_samples = utility_func(pairwise_gaussian_process.train_inputs[0]).squeeze().numpy()

        with torch.no_grad():
            y_latent_samples = pairwise_gaussian_process(pairwise_gaussian_process.train_inputs[0]).mean.numpy()

        fig.add_trace(go.Scatter(x=x_samples, y=y_samples, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Utility)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_samples, y=y_latent_samples, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Latent Utility)"), row=1, col=2)

        fig.update_layout(xaxis_title="X", yaxis_title="Utility")

    elif utility_func.dim == 2:
        # Create the utility function plot
        y = torch.linspace(*utility_func.bounds.T[1], grid_resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
        Z = utility_func(XY).reshape(X.shape)
        fig = make_subplots(rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}]], subplot_titles=("Utility Function", "Latent Utility Function (Mean)", "Latent Utility Function (STD)"))
        fig.add_trace(go.Surface(z=Z.numpy(), x=X.numpy(), y=Y.numpy(), colorbar=dict(orientation="h", len=1/3, x=1/7, y=-0.2), opacity=0.9, name="Utility Function"), row=1, col=1)

        # Create the latent utility function plot
        with torch.no_grad():
            multivariate_normal = pairwise_gaussian_process(XY)
            Z_mean = multivariate_normal.mean.reshape(X.shape).detach().numpy()
            Z_std = multivariate_normal.variance.reshape(X.shape).sqrt().detach().numpy()
        fig.add_trace(go.Surface(z=Z_mean.squeeze(), x=X.squeeze().numpy(), y=Y.squeeze().numpy(), colorbar=dict(orientation="h", len=1/3, x=0.5, y=-0.2), opacity=0.9, name="Latent Utility Function Mean"), row=1, col=2)
        fig.add_trace(go.Surface(z=Z_std.squeeze(), x=X.squeeze().numpy(), y=Y.squeeze().numpy(), colorbar=dict(orientation="h", len=1/3, x=6/7, y=-0.2), opacity=0.9, name="Latent Utility Function STD"), row=1, col=3)
        fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))

        # Add the evaluated points to each subplot
        x_samples = pairwise_gaussian_process.train_inputs[0][:, 0].numpy()
        y_samples = pairwise_gaussian_process.train_inputs[0][:, 1].numpy()
        z_utility = utility_func(pairwise_gaussian_process.train_inputs[0]).numpy()

        with torch.no_grad():
            multivariate_normal_samples = pairwise_gaussian_process(pairwise_gaussian_process.train_inputs[0])
            z_latent_utility_mean = multivariate_normal_samples.mean.numpy()
            z_latent_utility_std = multivariate_normal_samples.variance.sqrt().numpy()

        fig.add_trace(go.Scatter3d(x=x_samples, y=y_samples, z=z_utility, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Utility)"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=x_samples, y=y_samples, z=z_latent_utility_mean, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Latent Utility Mean)"), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=x_samples, y=y_samples, z=z_latent_utility_std, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Latent Utility STD)"), row=1, col=3)

        fig.update_layout(scene=dict(
            xaxis_title="X1",
            yaxis_title="X2",
            zaxis_title="Utility",
        ))

    elif utility_func.dim > 2:
        # Plot the correlation between the utility function and the latent utility function (mean)
        # in a scatter plot (since the latent utility function is not to the same scale, the correlation will not be perfect 1 in any case)
        # Make a tight multi-dimensional grid of Sobol samples to evaluate against the utility function and the latent utility function
        X = draw_sobol_samples(bounds=utility_func.bounds, n=grid_resolution, q=1).squeeze()
        Y_true = utility_func(X).squeeze().numpy()

        with torch.no_grad():
            multivariate_normal = pairwise_gaussian_process(X)
            Y_mean = multivariate_normal.mean.squeeze().numpy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Y_true, y=Y_mean, mode="markers", marker=dict(size=5, color="blue"), name="Quasi-Random Sobol Samples"))

        # Add the evaluated points
        Y_samples = utility_func(pairwise_gaussian_process.train_inputs[0]).squeeze().numpy()
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", GPInputWarning)
                Y_samples_mean = pairwise_gaussian_process(pairwise_gaussian_process.train_inputs[0]).mean.squeeze().numpy()

        fig.add_trace(go.Scatter(x=Y_samples, y=Y_samples_mean, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points"))

        # Add a regression line and label it with the Pearson correlation coefficient (only use Sobol samples)
        slope, intercept, r_value, p_value, std_err = linregress(Y_true, Y_mean)
        fig.add_trace(go.Scatter(x=Y_true, y=slope * Y_true + intercept, mode="lines", name=f"Regression Line (R^2={(r_value ** 2):.2f}, p={p_value:.2f})"))

        fig.update_layout(xaxis_title="Utility", yaxis_title="Latent Utility (Mean)")

    return fig

def get_utility_vs_latent_utility_fig_bo(utility_func: SyntheticTestFunction, single_task_gp: SingleTaskGP, grid_resolution: int = 50) -> go.Figure:
    # Plot the utility function and latent utility function (i.e., GP posterior mean + 95% CI)
    x = torch.linspace(*utility_func.bounds.T[0], grid_resolution)

    if utility_func.dim == 1:
        # Create the utility function plot
        x = x.view(-1, 1)
        x_normalized = single_task_gp.input_transform(x)
        y_true = utility_func(x).detach().numpy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y_true.squeeze(), mode="lines", name="Utility Function"))

        # Create the latent utility function plot
        with torch.no_grad():
            posterior = single_task_gp(x_normalized)
            y_mean_normalized = posterior.mean
            y_std_normalized = posterior.variance.sqrt()

            y_mean, y_std = single_task_gp.outcome_transform.untransform(y_mean_normalized, y_std_normalized)
            y_mean = y_mean.detach().numpy()
            y_std = y_std.detach().numpy()
        
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y_mean.squeeze(), mode="lines", name="Latent Utility Function Mean"))
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y_mean.squeeze() + 1.96 * y_std.squeeze(), mode="lines", line=dict(dash="dash", color="gray"), name="Latent Utility Function 95% CI (Upper)"))
        fig.add_trace(go.Scatter(x=x.squeeze().numpy(), y=y_mean.squeeze() - 1.96 * y_std.squeeze(), mode="lines", line=dict(dash="dash", color="gray"), name="Latent Utility Function 95% CI (Lower)", fill="tonexty", fillcolor="rgba(0,0,0,0.1)"))

        # Add the evaluated points
        y_samples_normalized = single_task_gp.train_targets.squeeze().numpy()
        y_samples, _ = single_task_gp.outcome_transform.untransform(y_samples_normalized)
        y_samples = y_samples.squeeze().numpy()
        x_samples = unnormalize(single_task_gp.train_inputs[0], utility_func.bounds).squeeze().numpy()

        fig.add_trace(go.Scatter(x=x_samples, y=y_samples, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points"))

        fig.update_layout(xaxis_title="X", yaxis_title="Utility")

    elif utility_func.dim == 2:
        # Create the utility function plot
        y = torch.linspace(*utility_func.bounds.T[1], grid_resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
        XY_normalized = single_task_gp.input_transform(XY)
        Z_true = utility_func(XY).reshape(X.shape).detach().numpy()
        
        fig = make_subplots(rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}]], 
                           subplot_titles=("Utility Function", "Latent Utility Function (Mean)", "Latent Utility Function (STD)"))
        fig.add_trace(go.Surface(z=Z_true, x=X.numpy(), y=Y.numpy(), 
                                colorbar=dict(orientation="h", len=1/3, x=1/7, y=-0.2), name="Utility Function", opacity=0.9), row=1, col=1)

        # Create the latent utility function plot
        with torch.no_grad():
            posterior = single_task_gp(XY_normalized)
            Z_mean_normalized = posterior.mean.reshape(X.shape).detach().numpy()
            Z_std_normalized = posterior.variance.reshape(X.shape).sqrt().detach().numpy()
            Z_mean, Z_std = single_task_gp.outcome_transform.untransform(Z_mean_normalized, Z_std_normalized)
            Z_mean = Z_mean.detach().numpy()
            Z_std = Z_std.detach().numpy()
        
        fig.add_trace(go.Surface(z=Z_mean.squeeze(), x=X.squeeze().numpy(), y=Y.squeeze().numpy(), 
                                colorbar=dict(orientation="h", len=1/3, x=0.5, y=-0.2), name="Latent Utility Function Mean", opacity=0.9), row=1, col=2)
        fig.add_trace(go.Surface(z=Z_std.squeeze(), x=X.squeeze().numpy(), y=Y.squeeze().numpy(), 
                                colorbar=dict(orientation="h", len=1/3, x=6/7, y=-0.2), name="Latent Utility Function STD", opacity=0.9), row=1, col=3)
        fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))

        # Add the evaluated points to each subplot
        x_samples, y_samples = unnormalize(single_task_gp.train_inputs[0], utility_func.bounds).T
        z_samples_normalized = single_task_gp.train_targets.squeeze().numpy()
        z_samples, _ = single_task_gp.outcome_transform.untransform(z_samples_normalized)
        z_samples = z_samples.squeeze().numpy()

        # Map the samples to the STD
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", GPInputWarning)
                posterior_samples = single_task_gp(single_task_gp.train_inputs[0])
            z_samples_mean_normalized = posterior_samples.mean
            z_samples_std_normalized = posterior_samples.variance.sqrt()
            z_samples_mean, z_samples_std = single_task_gp.outcome_transform.untransform(z_samples_mean_normalized, z_samples_std_normalized)
            z_samples_std = z_samples_std.squeeze().numpy()

        fig.add_trace(go.Scatter3d(x=x_samples.numpy(), y=y_samples.numpy(), z=z_samples, mode="markers", 
                                  marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Utility)"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=x_samples, y=y_samples, z=z_samples, mode="markers", 
                                  marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Latent Utility Mean)"), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=x_samples, y=y_samples, z=z_samples_std, mode="markers", 
                                  marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points (Latent Utility STD)"), row=1, col=3)

        fig.update_layout(scene=dict(
            xaxis_title="X1",
            yaxis_title="X2",
            zaxis_title="Utility",
        ))

    elif utility_func.dim > 2:
        # Plot the correlation between the utility function and the latent utility function (mean)
        # in a scatter plot
        # Make a tight multi-dimensional grid of Sobol samples to evaluate against the utility function and the latent utility function
        X = draw_sobol_samples(bounds=utility_func.bounds, n=grid_resolution, q=1).squeeze()
        X_normalized = single_task_gp.input_transform(X)
        Y_true = utility_func(X).squeeze().numpy()

        with torch.no_grad():
            posterior = single_task_gp(X_normalized)
            Y_mean_normalized = posterior.mean
            Y_mean, _ = single_task_gp.outcome_transform.untransform(Y_mean_normalized)
            Y_mean = Y_mean.squeeze().numpy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Y_true, y=Y_mean, mode="markers", marker=dict(size=5, color="blue"), name="Quasi-Random Sobol Samples"))

        # Add the evaluated points
        Y_samples_normalized = single_task_gp.train_targets.squeeze().numpy()
        Y_samples, _ = single_task_gp.outcome_transform.untransform(Y_samples_normalized)
        Y_samples = Y_samples.squeeze().numpy()

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", GPInputWarning)
                Y_samples_mean_normalized = single_task_gp(single_task_gp.train_inputs[0]).mean
            Y_samples_mean, _ = single_task_gp.outcome_transform.untransform(Y_samples_mean_normalized)
            Y_samples_mean = Y_samples_mean.squeeze().numpy()

        fig.add_trace(go.Scatter(x=Y_samples, y=Y_samples_mean, mode="markers", marker=dict(size=5, color="red", symbol="cross"), name="Evaluated Points"))
        fig.update_layout(xaxis_title="Utility", yaxis_title="Latent Utility (Mean)")

        # Add a regression line and label it with the Pearson correlation coefficient (only use Sobol samples)
        slope, intercept, r_value, p_value, std_err = linregress(Y_true, Y_mean)
        fig.add_trace(go.Scatter(x=Y_true, y=slope * Y_true + intercept, mode="lines", name=f"Regression Line (R^2={(r_value ** 2):.2f}, p={p_value:.2f})"))

    return fig

def get_utility_vs_latent_utility_fig(utility_func: SyntheticTestFunction, gaussian_process: PairwiseGP | SingleTaskGP, grid_resolution: int = 50) -> go.Figure:
    if isinstance(gaussian_process, PairwiseGP):
        fig = get_utility_vs_latent_utility_fig_pbo(utility_func, gaussian_process, grid_resolution)
    elif isinstance(gaussian_process, SingleTaskGP):
        fig = get_utility_vs_latent_utility_fig_bo(utility_func, gaussian_process, grid_resolution)
    else:
        raise ValueError("Unknown GP type")
    
    fig.update_layout(title="Utility Function and Latent Utility Function")
    return fig

if __name__ == "__main__":
    # original_image = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)
    # grid = make_grid([original_image])
    # show_images(grid)

    import doctest
    doctest.testmod()