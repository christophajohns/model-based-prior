import numpy as np
import pandas as pd

def _create_single_experiment_df(X: np.ndarray, y: np.ndarray, optimal_value: float, experiment_id: int = None) -> pd.DataFrame:
    """
    Create a DataFrame for a single experiment, including calculated regret, log regret, and log10 regret.

    Parameters:
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Objective values array of shape (n_samples,).
    optimal_value : float
        The known optimal value to calculate the regret.
    experiment_id : int, optional
        An identifier for the experiment, by default None.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns for each feature in X, the iteration number, y, regret, log_regret, and log10_regret.
        If experiment_id is provided, an "experiment" column is also included.

    Examples:
    --------
    >>> X = np.array([[0.1, 0.2], [0.4, 0.5]])
    >>> y = np.array([0.3, 0.6])
    >>> optimal_value = 1.0
    >>> _create_single_experiment_df(X, y, optimal_value)
       x_0  x_1  iteration    y  regret  log_regret  log10_regret
    0  0.1  0.2          1  0.3     0.7   -0.356675     -0.154902
    1  0.4  0.5          2  0.6     0.4   -0.916291     -0.397940

    >>> _create_single_experiment_df(X, y, optimal_value, experiment_id=1)
       x_0  x_1  iteration    y  regret  log_regret  log10_regret  experiment
    0  0.1  0.2          1  0.3     0.7   -0.356675     -0.154902           1
    1  0.4  0.5          2  0.6     0.4   -0.916291     -0.397940           1
    """
    df = pd.DataFrame(data=X, columns=[f"x_{i}" for i in range(X.shape[1])])
    df["iteration"] = range(1, len(y) + 1)
    df["y"] = y
    df["regret"] = df["y"].apply(lambda y: max(1e-10, optimal_value - y))  # Ensure no negative values or zeros
    df["log_regret"] = df["regret"].apply(np.log)
    df["log10_regret"] = df["regret"].apply(np.log10)
    
    if experiment_id is not None:
        df["experiment"] = experiment_id
    
    return df

def get_df(X: np.ndarray, y: np.ndarray, optimal_value: float) -> pd.DataFrame:
    """
    Create a DataFrame for a single experiment with regret and log10_regret calculations.

    Parameters:
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Objective values array of shape (n_samples,).
    optimal_value : float
        The known optimal value to calculate the regret.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns for each feature in X, the iteration number, y, regret, log_regret, and log10_regret.

    Examples:
    --------
    >>> X = np.array([[0.1, 0.2], [0.4, 0.5]])
    >>> y = np.array([0.3, 0.6])
    >>> optimal_value = 1.0
    >>> get_df(X, y, optimal_value)
       x_0  x_1  iteration    y  regret  log_regret  log10_regret
    0  0.1  0.2          1  0.3     0.7   -0.356675     -0.154902
    1  0.4  0.5          2  0.6     0.4   -0.916291     -0.397940
    """
    return _create_single_experiment_df(X, y, optimal_value)

def get_df_multi(X_list: list[np.ndarray], y_list: list[np.ndarray], optimal_value_list: list[float]) -> pd.DataFrame:
    """
    Create a combined DataFrame for multiple experiments, each with regret and log10_regret calculations.

    Parameters:
    ----------
    X_list : list of np.ndarray
        A list of feature matrices for each experiment. Each matrix has shape (n_samples, n_features).
    y_list : list of np.ndarray
        A list of objective value arrays for each experiment. Each array has shape (n_samples,).
    optimal_value_list : list of float
        A list of the known optimal values to calculate the regret.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns for each feature, iteration, y, regret, log_regret, log10_regret, and experiment ID.

    Examples:
    --------
    >>> X1 = np.array([[0.1, 0.2], [0.4, 0.5]])
    >>> y1 = np.array([0.3, 0.6])
    >>> X2 = np.array([[0.2, 0.3], [0.5, 0.6]])
    >>> y2 = np.array([0.4, 0.7])
    >>> optimal_values = [1.0, 1.0]
    >>> get_df_multi([X1, X2], [y1, y2], optimal_values)
       x_0  x_1  iteration    y  regret  log_regret  log10_regret  experiment
    0  0.1  0.2          1  0.3     0.7   -0.356675     -0.154902           1
    1  0.4  0.5          2  0.6     0.4   -0.916291     -0.397940           1
    2  0.2  0.3          1  0.4     0.6   -0.510826     -0.221849           2
    3  0.5  0.6          2  0.7     0.3   -1.203973     -0.522879           2
    """
    if not X_list or not y_list:
        raise ValueError("X_list and y_list cannot be empty.")
    
    if len(X_list) != len(y_list):
        raise ValueError("X_list and y_list must have the same length.")

    dataframes = [
        _create_single_experiment_df(X, y, optimal_value, experiment_id=i+1)
        for i, (X, y, optimal_value) in enumerate(zip(X_list, y_list, optimal_value_list))
    ]
    
    return pd.concat(dataframes, ignore_index=True)

if __name__ == "__main__":
    import doctest
    doctest.testmod()