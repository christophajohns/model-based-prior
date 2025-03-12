import os
import logging
import torch
from dotenv import load_dotenv
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List
import functools
import operator
import itertools
import tempfile

from modelbasedprior.logger import setup_logger
from modelbasedprior.optimization.pbo import load_model as pbo_load_model
from modelbasedprior.optimization.bo import load_model as bo_load_model
from modelbasedprior.visualization.analysis import get_df, get_df_multi
from modelbasedprior.visualization.visualization import (
    get_log10_regret_over_iteration_fig,
    get_log10_regret_over_iteration_fig_iqr,
    get_comparative_log10_regret_fig_iqr,
    make_grid,
    show_images,
    get_utility_vs_latent_utility_fig,
    pad_images,
)
from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss
from modelbasedprior.objectives.scatterplot_quality import ScatterPlotQualityLoss
from modelbasedprior.benchmarking.runner import BenchmarkRunner, ExperimentConfig
from modelbasedprior.benchmarking.database import ExperimentConfig as DBExperimentConfig, Objective as DBObjective

def get_sample_quality_per_iteration_fig(X: torch.Tensor, y: torch.Tensor, objective: SyntheticTestFunction, title_suffix: str = ""):
    df = get_df(X=X, y=y, optimal_value=objective.optimal_value)
    fig = get_log10_regret_over_iteration_fig(df=df, title=f"Sample Quality per Iteration {title_suffix}")
    return fig

def get_outcome_quality_per_iteration_fig(best_X: torch.Tensor, best_y: torch.Tensor, objective: SyntheticTestFunction, title_suffix: str = ""):
    df = get_df(X=best_X, y=best_y, optimal_value=objective.optimal_value)
    fig = get_log10_regret_over_iteration_fig(df=df, title=f"Outcome Quality per Iteration {title_suffix}")
    return fig

def get_sample_quality_per_iteration_fig_iqr(X_lst: List[torch.Tensor], y_lst: List[torch.Tensor], optimal_value_lst: List[float], title_suffix: str = ""):
    df = get_df_multi(X_list=X_lst, y_list=y_lst, optimal_value_list=optimal_value_lst)
    fig = get_log10_regret_over_iteration_fig_iqr(df=df, title=f"Median and IQR Sample Quality per Iteration {title_suffix}")
    return fig

def get_outcome_quality_per_iteration_fig_iqr(best_X_lst: List[torch.Tensor], best_y_lst: List[torch.Tensor], optimal_value_lst: List[float], title_suffix: str = ""):
    df = get_df_multi(X_list=best_X_lst, y_list=best_y_lst, optimal_value_list=optimal_value_lst)
    fig = get_log10_regret_over_iteration_fig_iqr(df=df, title=f"Median and IQR Outcome Quality per Iteration {title_suffix}")
    return fig

def plot_image_similarity(X: torch.Tensor, objective: ImageSimilarityLoss):
    candidate_images = objective._generate_image(X)
    target_image = objective._original_target_image.squeeze(0)
    show_images(make_grid([objective._original_image, target_image, *candidate_images]))

def plot_scatterplots(X: torch.Tensor, objective: ScatterPlotQualityLoss):
    candidate_images = pad_images([objective._generate_plot_image(objective.x_data, objective.y_data, *[param.item() for param in x]) for x in X])
    show_images(make_grid(candidate_images))

def get_optimization_results(experiment_config: DBExperimentConfig, objective_from_db: DBObjective, runner: BenchmarkRunner, data_dir: str):
    optimization_method = runner.db.get_optimization_type(experiment_config.optimization_type_id).description
    objective = BenchmarkRunner.get_objective(objective_from_db.description, minimal_noise_std=MINIMAL_NOISE_STD)
    gp_path = os.path.join(data_dir, f"gp_{experiment_config.id}.pt")

    if optimization_method == "BO" or "PriorSampling":
        bo_results = runner.db.get_bo_experiment_results(experiment_config.id)
        result_y = torch.tensor([[res.rating] for res in bo_results]).double()
        result_X = torch.tensor([res.parameters for res in bo_results]).double()
        result_best_X = torch.tensor([res.best_parameters for res in bo_results]).double()
        result_best_y = torch.tensor([res.best_rating for res in bo_results]).double()
        gp = bo_load_model(gp_path, train_X=result_X, train_Y=result_y, bounds=objective.bounds)

    elif optimization_method == "PBO":
        pbo_results = runner.db.get_pbo_experiment_results(experiment_config.id)
        pbo_experiment_config = runner.db.get_pbo_experiment_configs(experiment_config.id)
        params = {pbo_param.id: pbo_param for pbo_param in pbo_experiment_config}
        result_y = torch.tensor([res.rating for res in pbo_experiment_config]).double()
        result_X = torch.tensor([res.parameters for res in pbo_experiment_config]).double()
        result_best_X = torch.tensor([res.parameters_best for res in pbo_results]).double()
        result_best_y = torch.tensor([res.best_rating for res in pbo_results]).double()
        result_comparisons = torch.tensor([[params[res.parameters_preferred_id].parameters_id, params[res.parameters_not_preferred_id].parameters_id] for res in pbo_results])
        gp = pbo_load_model(gp_path, datapoints=result_X, comparisons=result_comparisons, bounds=objective.bounds)

    return result_X, result_y, result_best_X, result_best_y, gp


load_dotenv()

SMOKE_TEST: bool = False  # If TRUE runs the evaluation with limited budget and temporary database

if SMOKE_TEST:
    temporary_data_dir = tempfile.TemporaryDirectory()
    temporary_db_path = os.path.join(temporary_data_dir.name, "experiments.db")
    
NUM_RUNS: int = 10 if not SMOKE_TEST else 3  # number of repeated runs with different seeds for each experiment condition
NUM_TRIALS: int = 40 if not SMOKE_TEST else 3
NUM_PATHS: int = 1024 if not SMOKE_TEST else 128  # 4096
NUM_INITIAL_SAMPLES: int = 4
MINIMAL_NOISE_STD = 0.001
SEED: int = 0
DATA_DIR: str = os.getenv("DATA_DIR") if not SMOKE_TEST else temporary_data_dir.name
DB_NAME: str = "experiments.db"
RUN_BENCHMARK: bool = True
VISUALIZE_RUNS: bool = False
EXPERIMENT_TO_PLOT_ID: int | None = None

logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

db_path = os.path.join(DATA_DIR, DB_NAME) if not SMOKE_TEST else temporary_db_path

bias_levels = ["Unbiased", "Biased"]
uncertainty_levels = ["Certain", "MoreCertain", "Uncertain", "MoreUncertain"]
prior_types = ["None", *bias_levels, *[f"{bias}{uncertainty}" for bias, uncertainty in itertools.product(bias_levels, uncertainty_levels)]]
# prior_types = ["None", "Biased"]  # Use above for full analysis; this is for a simple injection comparison
optimization_methods = ["BO", "PBO", "PriorSampling"]
injection_methods = ["None", "ColaBO", "piBO"]

def generate_configs(
        objectives = List[str],
        prior_types: List[str] = prior_types,
        optimization_methods: List[str] = optimization_methods,
        injection_methods: List[str] = injection_methods,
):
    configs = []
    for optimization_method, prior_type, prior_injection_method, objective in itertools.product(
        optimization_methods, prior_types, injection_methods, objectives
    ):
        if optimization_method != "PriorSampling" and "Unbiased" in prior_type: continue
        if optimization_method == "PriorSampling" and prior_injection_method != "None": continue
        if optimization_method == "PriorSampling" and prior_type == "None": continue
        if (prior_type == "None" and prior_injection_method != "None") or (prior_injection_method == "None" and prior_type != "None" and optimization_method != "PriorSampling"): continue
        configs.append((optimization_method, prior_type, prior_injection_method, objective))

    return configs

def get_config_name(optimization_method: str, prior_type: str, prior_injection_method: str):
    if optimization_method == "PriorSampling":
        return f"Prior Sampling ({'Uniform' if prior_type == 'None' else prior_type})"
    elif prior_injection_method == "None":
        return "MC-LogEI"
    else:
        return f"{prior_injection_method}-MC-LogEI ({prior_type})"
    
def get_configs(
        objectives = List[str],
        prior_types: List[str] = prior_types,
        optimization_methods: List[str] = optimization_methods,
        injection_methods: List[str] = injection_methods,
        num_trials: int = NUM_TRIALS,
        num_paths: int = NUM_PATHS,
        num_initial_samples: int = NUM_INITIAL_SAMPLES,
    ):
    configs = []

    for optimization_method, prior_type, prior_injection_method, objective in generate_configs(
        objectives, prior_types, optimization_methods, injection_methods, 
    ):
        config_name = get_config_name(optimization_method, prior_type, prior_injection_method)

        config = ExperimentConfig(
            optimization_method=optimization_method,
            objective=objective,
            num_trials=num_trials,
            num_paths=num_paths,
            num_initial_samples=num_initial_samples,  # alternative best practice: num_initial_samples=2 * (objective.ndim + 1)
            prior_type=prior_type,
            prior_injection_method=prior_injection_method,
        )
        config_dict = {"name": config_name, "config": config}

        configs.append(config_dict)
        
    return configs


def get_sphere_configs(*args, **kwargs):
    global prior_types
    all_prior_types = kwargs.pop("prior_types", prior_types)
    prior_types = [prior_type for prior_type in all_prior_types if "Unbiased" not in prior_type]
    sphere_configs = get_configs(objectives=["Sphere"], optimization_methods=["BO", "PriorSampling"], prior_types=prior_types, *args, **kwargs)
    sphere_pbo_configs = get_configs(objectives=["Sphere"], optimization_methods=["PBO", "PriorSampling"], prior_types=prior_types, *args, **kwargs)
    sphere_noisy_configs = get_configs(objectives=["SphereNoisy"], optimization_methods=["BO", "PriorSampling"], prior_types=prior_types, *args, **kwargs)
    sphere_many_initial_samples_configs = get_configs(objectives=["Sphere"], optimization_methods=["BO", "PriorSampling"], prior_types=prior_types, num_initial_samples=NUM_INITIAL_SAMPLES * 2, *args, **kwargs)
    sphere_many_iterations_configs = get_configs(objectives=["Sphere"], optimization_methods=["BO", "PriorSampling"], prior_types=prior_types, num_trials=NUM_TRIALS * 2, *args, **kwargs)
        
    return sphere_configs, sphere_pbo_configs, sphere_noisy_configs, sphere_many_initial_samples_configs, sphere_many_iterations_configs

def get_shekel_configs(*args, **kwargs):
    shekel_configs = get_configs(objectives=["Shekel"], optimization_methods=["BO", "PriorSampling"], *args, **kwargs)
    shekel_many_paths_configs = get_configs(objectives=["Shekel"], optimization_methods=["BO"], injection_methods=["ColaBO"], num_paths=65536, *args, **kwargs)
    return shekel_configs, shekel_many_paths_configs

def get_image_similarity_configs(*args, **kwargs):
    return get_configs(objectives=["ImageSimilarity"], *args, **kwargs)

def get_scatterplot_quality_configs(*args, **kwargs):
    global prior_types
    all_prior_types = kwargs.pop("prior_types", prior_types)
    prior_types = [prior_type for prior_type in all_prior_types if "Unbiased" not in prior_type]
    return get_configs(objectives=["ScatterPlotQuality"], optimization_methods=["BO", "PriorSampling"], prior_types=prior_types, *args, **kwargs)

def get_mrlayout_quality_configs(*args, **kwargs):
    return get_configs(objectives=["MRLayoutQuality"], optimization_methods=["BO", "PriorSampling"], *args, **kwargs)

sphere_configs, sphere_pbo_configs, sphere_noisy_configs, sphere_many_initial_samples_configs, sphere_many_iterations_configs = get_sphere_configs()
shekel_configs, shekel_many_paths_configs = get_shekel_configs()
image_similarity_configs = get_image_similarity_configs()
scatterplot_quality_configs = get_scatterplot_quality_configs()
mrlayout_quality_configs = get_mrlayout_quality_configs()

configs = [
    config_dict["config"]
    for experiment_configs in [
        # sphere_configs,
        # sphere_pbo_configs,
        # sphere_noisy_configs,
        # sphere_many_initial_samples_configs,
        # sphere_many_iterations_configs,
        shekel_configs,
        shekel_many_paths_configs,
        # image_similarity_configs,
        # scatterplot_quality_configs,
        # mrlayout_quality_configs,
    ]
    for config_dict in experiment_configs
]

runner = BenchmarkRunner(seed=SEED, db_path=db_path, gp_dir_path=DATA_DIR, logger=logger)
if RUN_BENCHMARK:
    runner.run(configs, NUM_RUNS, minimal_noise_std=MINIMAL_NOISE_STD)

if VISUALIZE_RUNS:
    # Plot the sample quality per iteration with IQR
    COLORS = [
        ("blue", "lightblue"),
        ("red", "lightpink"),
        ("yellow", "lightyellow"),
        ("green", "lightgreen"),
        ("purple", "violet"),
        ("orange", "peachpuff"),
        ("brown", "burlywood"),
        ("pink", "lightcoral"),
        ("gray", "lightgray"),
        ("cyan", "lightcyan"),
        ("magenta", "lightpink"),
        ("lime", "lightgreen"),
        ("teal", "paleturquoise"),
        ("lavender", "thistle"),
        ("maroon", "mistyrose"),
        ("navy", "lightsteelblue"),
        ("olive", "palegoldenrod"),
        ("silver", "gainsboro"),
        ("black", "whitesmoke"),
    ]
    num_initial_iterations = NUM_INITIAL_SAMPLES  # for BO: NUM_INITIAL_SAMPLES; for PBO: (NUM_INITIAL_SAMPLES * (NUM_INITIAL_SAMPLES - 1)) // 2

    # SUFFIX = "(Sphere)"
    # NAMES = [config_dict["name"] for config_dict in sphere_configs]
    # CONFIGS = [config_dict["config"] for config_dict in sphere_configs]

    # SUFFIX = "(Sphere Noisy)"
    # NAMES = [config_dict["name"] for config_dict in sphere_noisy_configs]
    # CONFIGS = [config_dict["config"] for config_dict in sphere_noisy_configs]

    # SUFFIX = "(Sphere PBO)"
    # NAMES = [config_dict["name"] for config_dict in sphere_pbo_configs]
    # CONFIGS = [config_dict["config"] for config_dict in sphere_pbo_configs]
    # num_initial_iterations = (NUM_INITIAL_SAMPLES * (NUM_INITIAL_SAMPLES - 1)) // 2

    # SUFFIX = "(Sphere, Many Initial Samples)"
    # NAMES = [config_dict["name"] for config_dict in sphere_many_initial_samples_configs]
    # CONFIGS = [config_dict["config"] for config_dict in sphere_many_initial_samples_configs]
    # num_initial_iterations = NUM_INITIAL_SAMPLES * 2

    # SUFFIX = "(Sphere, Many Iterations)"
    # NAMES = [config_dict["name"] for config_dict in sphere_many_iterations_configs]
    # CONFIGS = [config_dict["config"] for config_dict in sphere_many_iterations_configs]

    # SUFFIX = "(Shekel)"
    # NAMES = [config_dict["name"] for config_dict in shekel_configs]
    # CONFIGS = [config_dict["config"] for config_dict in shekel_configs]

    SUFFIX = "(Shekel, Many Paths)"
    NAMES = [config_dict["name"] for config_dict in shekel_many_paths_configs]
    CONFIGS = [config_dict["config"] for config_dict in shekel_many_paths_configs]

    # SUFFIX = "(Image Similarity)"
    # NAMES = [config_dict["name"] for config_dict in image_similarity_configs]
    # CONFIGS = [config_dict["config"] for config_dict in image_similarity_configs]

    # SUFFIX = "(Scatter Plot Quality)"
    # NAMES = [config_dict["name"] for config_dict in scatterplot_quality_configs]
    # CONFIGS = [config_dict["config"] for config_dict in scatterplot_quality_configs]

    # SUFFIX = "(MR Layout Quality)"
    # NAMES = [config_dict["name"] for config_dict in mrlayout_quality_configs]
    # CONFIGS = [config_dict["config"] for config_dict in mrlayout_quality_configs]

    colors = COLORS[:len(NAMES)]
    configs_from_db = [runner.db.get_experiment_configs(
            optimization_type=config.optimization_method,
            num_trials=config.num_trials,
            num_paths=config.num_paths,
            num_initial_samples=config.num_initial_samples,
            objective_type=config.objective,
            prior_type=config.prior_type if config.prior_type is not None else "None",
            prior_injection_method=config.prior_injection_method if config.prior_injection_method is not None else "None",
        ) for config in CONFIGS
    ]
    # Print config IDs per name
    for name, config_list in zip(NAMES, configs_from_db):
        print(f"{name} IDs: {[config.id for config in config_list]}")
    data = {name: {} for name in NAMES}
    flat_configs_from_db: List[DBExperimentConfig] = functools.reduce(operator.iconcat, configs_from_db, [])
    config_types = [NAMES[i] for i, config_list in enumerate(configs_from_db) for _ in config_list]
    for experiment_config, config_type in zip(flat_configs_from_db, config_types):
        experiment_config_from_db = runner.db.get_experiment_config(experiment_config.id)
        objective_from_db = runner.db.get_objective(experiment_config.objective_id)
        result_X, result_y, result_best_X, result_best_y, gp = get_optimization_results(experiment_config, objective_from_db, runner, DATA_DIR)

        data[config_type][experiment_config.id] = {
            "result_X": result_X,
            "result_y": result_y,
            "result_best_X": result_best_X,
            "result_best_y": result_best_y,
            "optimal_value": objective_from_db.optimal_value,
            "gp": gp,
        }

    dfs_sample = [
        get_df_multi(
            X_list=[data[prior_type][experiment_id]["result_X"] for experiment_id in data[prior_type]],
            y_list=[data[prior_type][experiment_id]["result_y"] for experiment_id in data[prior_type]],
            optimal_value_list=[data[prior_type][experiment_id]["optimal_value"] for experiment_id in data[prior_type]],
        )
        for prior_type in NAMES
    ]

    fig_compare_sample_quality = get_comparative_log10_regret_fig_iqr(
        dfs=dfs_sample,
        names=NAMES,
        colors=colors,
        num_initial_iterations=num_initial_iterations,
    )
    fig_compare_sample_quality.update_layout(title=f"Sample Quality per Iteration {SUFFIX}")
    fig_compare_sample_quality.show()

    # Plot the outcome quality per iteration with IQR
    dfs_outcome = [
        get_df_multi(
            X_list=[data[prior_type][experiment_id]["result_best_X"] for experiment_id in data[prior_type]],
            y_list=[data[prior_type][experiment_id]["result_best_y"] for experiment_id in data[prior_type]],
            optimal_value_list=[data[prior_type][experiment_id]["optimal_value"] for experiment_id in data[prior_type]],
        )
        for prior_type in NAMES
    ]
    fig_compare_outcome_quality = get_comparative_log10_regret_fig_iqr(
        dfs=dfs_outcome,
        names=NAMES,
        colors=colors,
        num_initial_iterations=num_initial_iterations,
    )
    fig_compare_outcome_quality.update_layout(title=f"Outcome Quality per Iteration {SUFFIX}")
    fig_compare_outcome_quality.show()


if EXPERIMENT_TO_PLOT_ID is not None:
    # Get the experiment results for a specific experiment
    experiment_config = runner.db.get_experiment_config(EXPERIMENT_TO_PLOT_ID)
    objective_from_db = runner.db.get_objective(experiment_config.objective_id)
    result_X, result_y, result_best_X, result_best_y, gp = get_optimization_results(experiment_config, objective_from_db, runner, DATA_DIR)
    objective = BenchmarkRunner.get_objective(objective_from_db.description, minimal_noise_std=MINIMAL_NOISE_STD)
    optimization_method = runner.db.get_optimization_type(experiment_config.optimization_type_id).description


    # Plot the sample quality per iteration
    fig_samples = get_sample_quality_per_iteration_fig(result_X, result_y, objective, title_suffix=f"({optimization_method})")
    fig_samples.show()

    # Plot the outcome quality per iteration
    fig_outcome = get_outcome_quality_per_iteration_fig(result_best_X, result_best_y, objective, title_suffix=f"({optimization_method})")
    fig_outcome.show()


    # Plot the utility vs latent utility
    fig_utility = get_utility_vs_latent_utility_fig(objective, gp)
    fig_utility.show()


    # Show the samples
    if objective.__class__.__name__ == "ImageSimilarityLoss":
        plot_image_similarity(result_X, objective)
        plot_image_similarity(result_best_X, objective)

    if objective.__class__.__name__ == "ScatterPlotQualityLoss":
        plot_scatterplots(result_X, objective)
        plot_scatterplots(result_best_X, objective)

if SMOKE_TEST:
    temporary_data_dir.cleanup()