import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from botorch.utils.transforms import unnormalize
import seaborn as sns
from dotenv import load_dotenv
import os
from modelbasedprior.objectives.scatterplot_quality import ScatterPlotQualityLoss, ScatterPlotQualityLossRegressor

load_dotenv()

PRETRAINED_MODELS_DIR = os.getenv("PRETRAINED_MODELS_DIR")
    
def get_loss_function() -> nn.Module:
    """
    Return Mean Squared Error Loss function for regression tasks.

    Returns:
        torch.nn.Module: MSELoss for regression tasks.
    """
    return nn.MSELoss()

def get_optimizer(model: nn.Module, learning_rate: float = 1e-3, weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """
    Get the Adam optimizer with optional weight decay (L2 regularization).

    Args:
        model (torch.nn.Module): The neural network model.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): L2 regularization strength (default 1e-5).

    Returns:
        torch.optim.Optimizer: Configured optimizer for model training.
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def get_lr_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a cosine annealing learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def normalize_features(X: torch.Tensor, bounds: list[tuple[float, float]]) -> torch.Tensor:
    """
    Normalize the input features to the range [0, 1] using min-max scaling.

    Args:
        X (torch.Tensor): Input tensor of shape [batch_size, 3] with dtype torch.float64.
        bounds (list[tuple[float, float]]): List of (min, max) tuples for each feature, 
                                            defining the bounds of the input tensor.

    Returns:
        torch.Tensor: Normalized input tensor with values between 0 and 1.
    """
    X_normalized = torch.zeros_like(X, dtype=torch.double)
    for i, (min_val, max_val) in enumerate(bounds):
        X_normalized[:, i] = (X[:, i] - min_val) / (max_val - min_val)
    return X_normalized

def train_val_test_split(X: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.7, 
                         val_ratio: float = 0.15, test_ratio: float = 0.15) -> tuple:
    """
    Split the data into training, validation, and test sets using sklearn's train_test_split.

    Args:
        X (torch.Tensor): Input tensor of shape [batch_size, 3] with dtype torch.float64.
        y (torch.Tensor): Target tensor of shape [batch_size, 1] with dtype torch.float64.
        train_ratio (float): Proportion of data to use for training (default 70%).
        val_ratio (float): Proportion of data to use for validation (default 15%).
        test_ratio (float): Proportion of data to use for testing (default 15%).

    Returns:
        tuple: Split datasets (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_ratio + test_ratio), random_state=42)
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_size), random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataloaders(X_train: torch.Tensor, y_train: torch.Tensor, 
                       X_val: torch.Tensor, y_val: torch.Tensor, 
                       X_test: torch.Tensor, y_test: torch.Tensor, 
                       batch_size: int = 32) -> tuple:
    """
    Create PyTorch DataLoaders for batch processing of training, validation, and test data.

    Args:
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        X_val (torch.Tensor): Validation input data.
        y_val (torch.Tensor): Validation target data.
        X_test (torch.Tensor): Test input data.
        y_test (torch.Tensor): Test target data.
        batch_size (int): Batch size for training (default: 32).

    Returns:
        tuple: DataLoaders for training, validation, and test sets.
    """
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                loss_fn: nn.Module, scheduler: torch.optim.lr_scheduler._LRScheduler, 
                epochs: int, patience: int) -> dict:
    """
    Train the model using early stopping and learning rate scheduling.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        loss_fn (torch.nn.Module): Loss function for regression.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait before early stopping if validation loss does not improve.

    Returns:
        dict: Dictionary containing training and validation loss history.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(os.getenv('PRETRAINED_MODELS_DIR'), 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return history

def build_model(input_dim: int, hidden_dims: list[int], output_dim: int = 1) -> nn.Module:
    """
    Build and return an instance of the ScatterPlotQualityLossRegressor model.

    Args:
        input_dim (int): Number of input features (e.g., 3 for your case).
        hidden_dims (list[int]): List of integers defining the number of neurons in each hidden layer.
        output_dim (int): The number of output units (default: 1 for regression).

    Returns:
        nn.Module: An instance of the ScatterPlotQualityLossRegressor model.
    """
    model = ScatterPlotQualityLossRegressor(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module) -> dict:
    """
    Evaluate the model on the test dataset using the provided loss function for the main evaluation metric,
    and also calculate MSE and MAE metrics.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        loss_fn (torch.nn.Module): Loss function for evaluation, e.g., MSELoss or custom loss.

    Returns:
        dict: Dictionary containing loss (using loss_fn), MSE, and MAE metrics.
    """
    model.eval()
    total_loss, total_mse, total_mae = 0.0, 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            batch_size = y_batch.size(0)
            
            # Calculate the provided loss function
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item() * batch_size  # Sum the loss over all batches
            
            # Calculate MSE and MAE as additional metrics
            total_mse += nn.functional.mse_loss(outputs, y_batch, reduction='sum').item()
            total_mae += nn.functional.l1_loss(outputs, y_batch, reduction='sum').item()
            
            total_samples += batch_size

    # Normalize by total number of samples
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    return {'loss': avg_loss, 'mse': avg_mse, 'mae': avg_mae}

def plot_loss_history(history: dict):
    """
    Plot the training and validation loss history.

    Args:
        history (dict): Dictionary containing the loss history from training.
    """
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def plot_predictions_vs_actuals(model: nn.Module, test_loader: DataLoader):
    """
    Plot predictions vs actual values for the test set.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.show()

def plot_residuals(model: nn.Module, test_loader: DataLoader):
    """
    Plot the residuals (errors) between predictions and actuals for the test set.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
    """
    model.eval()
    residuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            residuals.append((y_batch - outputs).cpu().numpy())

    residuals = np.concatenate(residuals)

    plt.hist(residuals, bins=50, alpha=0.75)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.show()

def generate_training_data(func: ScatterPlotQualityLoss, num_samples=5):
    """
    Generate synthetic training data for the regression task.

    Args:
        func (ScatterPlotQualityLoss): The function to generate the data.
        num_samples (int): Number of samples to generate.

    Returns:
        tuple: Tuple of input features (X_train) and target values (y_train).
    """
    X_train_unit = torch.rand(num_samples, 3).double()
    X_train = unnormalize(X_train_unit, func.bounds)
    y_train = func(X_train).view(-1, 1)
    return X_train, y_train

def main(X: torch.Tensor, y: torch.Tensor, bounds: list[tuple[float, float]]):
    """
    Main function for the complete training procedure. It preprocesses the data,
    builds the model, trains it, evaluates it, and saves the best model.

    Args:
        X (torch.Tensor): Input tensor of shape [batch_size, 3].
        y (torch.Tensor): Target tensor of shape [batch_size, 1].
    """

    # Step 1: Data Preprocessing
    print("Preprocessing data...")
    X_normalized = normalize_features(X, bounds)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_normalized, y)

    # Step 2: Create DataLoaders for batch processing
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 3: Build Model Architecture
    print("Building the model...")
    input_dim = X.shape[1]  # 3 features
    hidden_dims = [64, 64, 64]  # Three hidden layers with 64 units each
    model = build_model(input_dim=input_dim, hidden_dims=hidden_dims)

    # Step 4: Define Loss Function and Optimizer
    print("Setting up the optimizer and scheduler...")
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model, learning_rate=1e-3, weight_decay=1e-5)
    scheduler = get_lr_scheduler(optimizer)

    # Step 5: Train the Model
    print("Training the model...")
    epochs = 200  # Maximum number of epochs
    patience = 20  # Early stopping patience

    history = train_model(model=model, 
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          optimizer=optimizer, 
                          loss_fn=loss_fn, 
                          scheduler=scheduler, 
                          epochs=epochs, 
                          patience=patience)

    # Step 6: Visualize Loss History
    print("Plotting training and validation loss history...")
    plot_loss_history(history)

    # Step 7: Evaluate the Model on Test Set
    print("Evaluating the model on test data...")
    test_metrics = evaluate_model(model, test_loader, loss_fn)
    print(f"Test Results - MSE: {test_metrics['mse']}, MAE: {test_metrics['mae']}")

    # Step 8: Visualize Predictions and Residuals
    print("Plotting predictions vs actual values...")
    plot_predictions_vs_actuals(model, test_loader)
    
    print("Plotting residuals...")
    plot_residuals(model, test_loader)

    # Step 9: Save the Best Model
    print("Saving the best model...")
    torch.save(model.state_dict(), os.path.join(os.getenv('PRETRAINED_MODELS_DIR'), 'best_model.pth'))
    print("Model saved to 'best_model.pth'")

if __name__ == "__main__":
    # Seed for reproducibility
    torch.manual_seed(42)

    print("Loading data...")
    df = sns.load_dataset('mpg')  # Load Cars dataset from seaborn

    print("Creating ScatterPlotQualityLoss function...")
    x_data = torch.tensor(df['horsepower'].values, dtype=torch.float32)
    y_data = torch.tensor(df['mpg'].values, dtype=torch.float32)
    func = ScatterPlotQualityLoss(x_data=x_data, y_data=y_data, weight_overplotting = 0,negate=True)

    print("Generating synthetic data...")
    X, y = generate_training_data(func, num_samples=1000)

    # Train the model
    # main(X, y, func._bounds)

    print("Preprocessing data...")
    X_normalized = normalize_features(X, func._bounds)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_normalized, y)

    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    print("Loading the best model...")
    model = build_model(input_dim=X.shape[1], hidden_dims=[64, 64, 64])
    model.load_state_dict(torch.load(os.path.join(os.getenv('PRETRAINED_MODELS_DIR'), 'best_model.pth')))

    print("Plotting predictions vs actual values...")
    plot_predictions_vs_actuals(model, test_loader)
    
    print("Plotting residuals...")
    plot_residuals(model, test_loader)