import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import os

from layers import canonizetion, symetrizartion, sampled_symetrizartion, equiv_linear, augmentaion, invariant_linear


class Model(nn.Module):
    def __init__(self, feature_in: int, num_points: int, num_classes: int, num_layers: int = 4, hidden_dim: int = 64):
        super(Model, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(feature_in * num_points, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Last layer: hidden_dim -> num_classes
        self.layers.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        '''
        Forward pass through the extractor.

        Args:
        - x (torch.Tensor): Input tensor (batch_size, num_points, feature_in).

        Returns:
        - torch.Tensor: Extracted features (batch_size, num_classes).
        '''
        # Flatten the input if it has more than 2 dimensions
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_points * feature_in)

        # Pass through all layers except the last one with ReLU activation
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Final layer without activation
        x = self.layers[-1](x)

        return x


class SymmetricModel(nn.Module):
    def __init__(self, feature_in: int, num_classes: int, num_layers: int = 4, hidden_dim: int = 64):
        super(SymmetricModel, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(equiv_linear(feature_in, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(equiv_linear(hidden_dim, hidden_dim))

        self.invariant_layer = invariant_linear(num_classes)

    def forward(self, x):
        '''
        Forward pass through the symmetric model.

        Args:
        - x (torch.Tensor): Input tensor (batch_size, num_points, feature_in).

        Returns:
        - torch.Tensor: Extracted features (batch_size, num_classes).
        '''
        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.invariant_layer(x)
        x = torch.mean(x, dim=1)

        return x


class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, train: bool = True):
        '''
        Initialize the ModelNet40 dataset.

        Args:
        - data_path (str): Path to the ModelNet40 dataset directory.
        - train (bool): Whether to load the training set (True) or validation set (False).
        '''
        def load_file_to_list(file_path):
            '''
            Load file paths from a text file.

            Args:
            - file_path (str): Path to the text file containing file paths.

            Returns:
            - list: List of file paths.
            '''
            with open(file_path, 'r') as f:
                return [line.strip() for line in f.readlines()]

        file_paths_file = 'modelnet40_train.txt' if train else 'modelnet40_test.txt'
        self.data_path = data_path
        self.file_paths = load_file_to_list(os.path.join(data_path, file_paths_file))
        self.classes = load_file_to_list(os.path.join(data_path, 'modelnet40_shape_names.txt'))

    def __len__(self):
        '''
        Return the number of samples in the dataset.

        Returns:
        - int: Number of samples in the dataset.
        '''
        return len(self.file_paths)

    def __getitem__(self, idx):
        '''
        Get a sample from the dataset.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: A tuple containing the data and label for the sample.
        '''
        file_path = self.file_paths[idx]
        class_name = file_path.split('_')[:-1]
        class_name = '_'.join(class_name) if class_name else '....'

        if class_name not in self.classes:
            raise ValueError(f"Class '{class_name}' not found in classes list.")

        data = np.loadtxt(os.path.join(self.data_path, class_name, f'{file_path}.txt'), delimiter=',', dtype=np.float32)[:256]
        data = data[:, :3]
        label = self.classes.index(class_name)

        return data, label


def train_model(policy, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn, accuercy_fn, scheduler=None):
    '''
    Train a policy using behavior cloning.

    Args:
    - policy (torch.nn.Module): The policy model to train.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
    - val_dataloader (torch.utils.data.DataLoader): DataLoader providing validation data.
    - num_epochs (int): Number of epochs to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the policy.
    - loss_fn (callable): Loss function to compute the loss of a batch.
    - accuercy_fn (callable): Function to compute accuracy of the model.
    - scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.

    Returns:
    - policy (torch.nn.Module): The trained policy model.
    - losses_train (list): List of average training losses per epoch.
    - losses_validation (list): List of average validation losses per epoch.
    '''
    losses_train = []
    losses_validation = []

    progressbar = tqdm(total=num_epochs, desc="Training", unit="epoch")
    epoch_progressbar = tqdm(total=len(train_dataloader), desc="Epoch Progress", leave=False)

    for _ in range(num_epochs):
        avg_epoch_loss = train_epoch(train_dataloader, loss_fn, policy, optimizer, epoch_progressbar)
        losses_train.append(avg_epoch_loss)

        # Validate after each epoch
        avg_val_loss = validate_model(policy, val_dataloader, accuercy_fn)
        losses_validation.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_epoch_loss)

        epoch_progressbar.reset()
        progressbar.update(1)
        progressbar.set_postfix(train_loss=avg_epoch_loss, val_loss=avg_val_loss)

    return policy, losses_train, losses_validation


def train_epoch(dataloader, loss_fn, policy, optimizer, epoch_progressbar):
    '''
    Train the policy for one epoch.

    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
    - loss_fn (callable): Loss function to compute the loss of a batch.
    - policy (torch.nn.Module): The policy model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the policy.
    - epoch_progressbar (tqdm): Progress bar for the current epoch.

    Returns:
    - avg_epoch_loss (float): Average loss over the epoch.
    '''
    policy.train()
    epoch_losses = []

    for states, actions in dataloader:
        loss = loss_fn(policy, actions, states)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_progressbar.update(1)
        epoch_progressbar.set_postfix(loss=loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

    return avg_epoch_loss


def validate_model(policy, dataloader, loss_fn):
    '''
    Validate the policy model on a validation dataset.

    Args:
    - policy (torch.nn.Module): The policy model to validate.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing validation data.
    - loss_fn (callable): Loss function to compute the loss of a batch.

    Returns:
    - avg_loss (float): Average loss over the validation dataset.
    '''
    policy.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for states, actions in dataloader:
            loss = loss_fn(policy, actions, states)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    return avg_loss


if __name__ == "__main__":
    import os

    batch_size = 128
    data_path = os.path.join(os.getcwd(), 'modelnet40_normal_resampled')

    dataset_train = ModelNet40Dataset(data_path, train=True)
    dataset_val = ModelNet40Dataset(data_path, train=False)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = SymmetricModel(feature_in=3, num_classes=40, num_layers=4, hidden_dim=64)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    def loss_fn(policy, labels, images):
        """
        Custom loss function for training the policy.

        Args:
        - policy (torch.nn.Module): The policy model.
        - labels (torch.Tensor): The ground truth labels.
        - images (torch.Tensor): Input images or data.

        Returns:
        - torch.Tensor: Computed loss.
        """
        outputs = policy(images)
        return F.cross_entropy(outputs, labels)

    def accuracy_fn(policy, labels, images):
        """
        Custom accuracy function for evaluating the policy.

        Args:
        - policy (torch.nn.Module): The policy model.
        - labels (torch.Tensor): The ground truth labels.
        - images (torch.Tensor): Input images or data.

        Returns:
        - float: Computed accuracy.
        """
        outputs = policy(images)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum()

        return correct / labels.size(0)

    # Train the model
    model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, num_epochs=50, optimizer=optimizer, loss_fn=loss_fn, accuercy_fn=accuracy_fn, scheduler=scheduler)
