from itertools import permutations


import torch
from torch import nn
import torch.nn.functional as F


class canonizetion(nn.Module):
    '''
    a canonization layer with S_n symmetry
    '''

    def __init__(self):
        super(canonizetion, self).__init__()


    def canonize(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Canonizes the input tensor by sorting it's vectors by their sum

        Args:
        - x (torch.Tensor): Input tensor of shape (n, d)

        Returns:
        - torch.Tensor: Canonized tensor of the same shape as input.
        '''
        keys = x.sum(dim=1)
        idx = torch.argsort(keys)

        return x[idx]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Canonizes the input tensor by sorting along the last dimension.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, n, d)

        Returns:
        - torch.Tensor: Canonized tensor of the same shape as input.
        '''
        # map over the batch dimension
        x = torch.stack([self.canonize(x_i) for x_i in x], dim=0)

        return x


class symetrizartion(nn.Module):
    '''
    applies a symmetrization of a model by averaging over all permutations
    '''

    def __init__(self, model: nn.Module):
        super(symetrizartion, self).__init__()
        self.model = model


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies the model to the input tensor and averages the results over all permutations.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, n, d)

        Returns:
        - torch.Tensor: Symmetrized tensor of the same shape as input.
        '''
        _, n, _ = x.shape

        permutations_indices = list(permutations(range(n)))

        batchs = [x[:, perm, :] for perm in permutations_indices]

        outputs = [self.model(batch) for batch in batchs]

        output = torch.mean(torch.stack(outputs), dim=0)

        return output


class sampled_symetrizartion(nn.Module):
    '''
    applies a symmetrization of a model by averaging over a random sample of permutations
    '''

    def __init__(self, model: nn.Module, num_samples: int = 100):
        super(sampled_symetrizartion, self).__init__()
        self.model = model
        self.num_samples = num_samples


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies the model to the input tensor and averages the results over a random sample of permutations.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, n, d)

        Returns:
        - torch.Tensor: Symmetrized tensor of the same shape as input.
        '''
        _, n, _ = x.shape

        permutations_indices = [torch.randperm(n).tolist() for _ in range(self.num_samples)]

        batchs = [x[:, perm, :] for perm in permutations_indices]

        outputs = [self.model(batch) for batch in batchs]

        output = torch.mean(torch.stack(outputs), dim=0)

        return output


class equiv_linear(nn.Module):
    '''
    a linear layer with S_n symmetry
    '''

    def __init__(self, in_features, out_features):
        '''
        Initilizes a symmetric linear layer.

        Args:
        - in_features (int): Number of input features d.
        - out_features (int): Number of output features d'.
        '''
        super(equiv_linear, self).__init__()

        self.linear1 = nn.Parameter(torch.Tensor(in_features, out_features))
        self.linear2 = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.xavier_uniform_(self.linear1)
        nn.init.xavier_uniform_(self.linear2)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        '''
        Applies the linear equivariant transformation to the input tensor.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, n, d).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, n, d').
        '''

        h_self = torch.matmul(x, self.linear1)
        x_sum = torch.sum(x, dim=1, keepdim=True)
        h_sum = torch.matmul(x_sum, self.linear2)
        res = h_self + h_sum + self.bias

        return res


class invariant_linear(nn.Module):
    '''
    a linear layer that is invariant to permutations
    '''

    def __init__(self, out_features):
        '''
        Initializes an invariant linear layer.

        Args:
        - out_features (int): Number of output features d'.
        '''
        super(invariant_linear, self).__init__()

        self.linear = nn.Parameter(torch.Tensor(1, out_features))
        self.bias = nn.Parameter(torch.Tensor(1, out_features))

        nn.init.xavier_uniform_(self.linear)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        '''
        Applies the invariant linear transformation to the input tensor.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, n, d).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, d').
        '''
        raise NotImplementedError('this layer is not implemented yet')
        batch_size, n, d = x.shape
        x_sum = torch.sum(x, dim=-1)
        res = torch.matmul(x_sum, self.linear).expand(batch_size, n, -1) + self.bias.expand(batch_size, n, -1)

        return res


class augmentaion(nn.Module):
    '''
    a layer that augments the input with the S_n symmetry
    '''

    def __init__(self):
        '''
        Initializes an augmentation layer.

        Args:
        - in_features (int): Number of input features d.
        - out_features (int): Number of output features d'.
        '''
        super(augmentaion, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Augments the input tensor by random permutation of the rows.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, n, d).

        Returns:
        - torch.Tensor: Augmented tensor of shape (batch_size, n, d).
        '''
        if not self.training:
            return x

        _, n, _ = x.shape
        permuted_indices = torch.randperm(n)
        x_permuted = x[:, permuted_indices, :]

        return x_permuted


def test_input_shape(net, n, d, d_tag):
    '''
    Tests if the model can handle inputs of shape (batch_size, n, d).

    Args:
    - net (nn.Module): The model to test.
    - n (int): Number of elements in the input tensor.
    - d (int): Number of features in the input tensor.
    - d_tag (int): Expected number of output features.

    Returns:
    - bool: True if the model can handle the input shape, False otherwise.
    '''
    x = torch.randn(1, n, d)
    output = net(x)

    assert output.shape == (1, n, d_tag), f"Expected output shape (1, {n}, {d_tag}), but got {output.shape}"

    return True


def test_equivariance(net, n, d, atol = 1e-6):
    '''
    Tests the equivariance of the model by checking if the output is invariant to permutations.

    Args:
    - net (nn.Module): The model to test.
    - n (int): Number of elements in the input tensor.
    - d (int): Number of features in the input tensor.
    - atol (float): Absolute tolerance for the test.

    Returns:
    - bool: True if the model is equivariant, False otherwise.
    '''
    torch.random.manual_seed(0)
    x = torch.randn(1, n, d)
    permutation = torch.randperm(n)
    permuted_x = x[0, permutation, :].unsqueeze(0)

    permuted_output = net(permuted_x)
    output = net(x)

    output = output[0, permutation, :].unsqueeze(0)

    is_equiv = torch.allclose(output, permuted_output, atol=atol)

    return is_equiv


def test_invariance(net, n, d, atol = 1e-6):
    '''
    Tests the invariance of the model by checking if the output is invariant to permutations.

    Args:
    - net (nn.Module): The model to test.
    - n (int): Number of elements in the input tensor.
    - d (int): Number of features in the input tensor.
    - atol (float): Absolute tolerance for the test.

    Returns:
    - bool: True if the model is invariant, False otherwise.
    '''
    torch.random.manual_seed(0)
    x = torch.randn(1, n, d)
    permutation = torch.randperm(n)
    permuted_x = x[:, permutation, :]

    permuted_output = net(permuted_x)
    output = net(x)

    is_invariant = torch.allclose(output, permuted_output, atol=atol)

    return is_invariant


if __name__ == "__main__":
    class BasicModel(nn.Module):
        '''
        A basic model that applies a linear transformation followed by a ReLU activation.
        '''

        def __init__(self, in_features, out_features):
            '''
            Initializes the basic model.

            Args:
            - in_features (int): Number of input features d.
            - out_features (int): Number of output features d'.
            '''
            super(BasicModel, self).__init__()
            self.linear1 = nn.Linear(in_features, 128)
            self.linear2 = nn.Linear(128, 128)
            self.linear3 = nn.Linear(128, out_features)


        def forward(self, x):
            '''
            Applies the model to the input tensor.

            Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, n, d).

            Returns:
            - torch.Tensor: Output tensor of shape (batch_size, n, d').
            '''
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)

            return x

    class BasicModelWithSymmetry(nn.Module):
        '''
        A basic model that applies a linear transformation followed by a ReLU activation,
        with S_n symmetry.
        '''

        def __init__(self, in_features, out_features):
            '''
            Initializes the basic model with symmetry.

            Args:
            - in_features (int): Number of input features d.
            - out_features (int): Number of output features d'.
            '''
            super(BasicModelWithSymmetry, self).__init__()
            self.linear1 = equiv_linear(in_features, 128)
            self.linear2 = equiv_linear(128, 128)
            self.linear3 = equiv_linear(128, out_features)


        def forward(self, x):
            '''
            Applies the model to the input tensor.

            Args:
            - x (torch.Tensor): Input tensor of shape (batch_size, n, d).

            Returns:
            - torch.Tensor: Output tensor of shape (batch_size, n, d').
            '''
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)

            return x

    n = 4
    d = 5
    d_tag = 3

    canonizated_model = nn.Sequential(canonizetion(), symetrizartion(BasicModel(d, d_tag)))
    symetrized_model = symetrizartion(BasicModel(d, d_tag))
    sampled_symetrized_model = sampled_symetrizartion(BasicModel(d, d_tag), num_samples=12)
    augmentated_model = nn.Sequential(augmentaion(), BasicModel(d, d_tag))
    symetric_model = BasicModelWithSymmetry(d, d_tag)
    invariant_model = nn.Sequential(symetric_model, invariant_linear(d_tag))

    # Test input shape
    test_input_shape(canonizated_model, n, d, d_tag)
    test_input_shape(symetrized_model, n, d, d_tag)
    test_input_shape(sampled_symetrized_model, n, d, d_tag)
    test_input_shape(augmentated_model, n, d, d_tag)
    test_input_shape(symetric_model, n, d, d_tag)
    test_input_shape(invariant_model, n, d, d_tag)

    # for all models print if the model is equivariant, invariant or none
    # Test canonizated model
    canonizated_equiv = test_equivariance(canonizated_model, n, d)
    canonizated_invar = test_invariance(canonizated_model, n, d)

    # Test symetrized model
    symetrized_equiv = test_equivariance(symetrized_model, n, d)
    symetrized_invar = test_invariance(symetrized_model, n, d)

    # Test sampled symetrized model
    sampled_symetrized_equiv = test_equivariance(sampled_symetrized_model, n, d)
    sampled_symetrized_invar = test_invariance(sampled_symetrized_model, n, d)

    # Test augmentated model
    augmentated_equiv = test_equivariance(augmentated_model, n, d)
    augmentated_invar = test_invariance(augmentated_model, n, d)

    # Test symetric model
    symetric_equiv = test_equivariance(symetric_model, n, d)
    symetric_invar = test_invariance(symetric_model, n, d)

    # Test invariant model
    invariant_equiv = test_equivariance(invariant_model, n, d)
    invariant_invar = test_invariance(invariant_model, n, d)

    print('\nshould be invariant')
    print("Canonizated Model Equivariance:", canonizated_equiv)
    print("Canonizated Model Invariance:", canonizated_invar)
    if not (canonizated_equiv and canonizated_invar):
        raise ValueError("Canonizated model should be invariant and equivariant")

    print('\nshould be invariant')
    print("Symetrized Model Equivariance:", symetrized_equiv)
    print("Symetrized Model Invariance:", symetrized_invar)
    if not (symetrized_equiv and symetrized_invar):
        raise ValueError("Symetrized model should be invariant and equivariant")

    print('\nshould not be invariant or equivariant')
    print("Sampled Symetrized Model Equivariance:", sampled_symetrized_equiv)
    print("Sampled Symetrized Model Invariance:", sampled_symetrized_invar)
    if sampled_symetrized_equiv or sampled_symetrized_invar:
        raise ValueError("Sampled symetrized model should not be invariant or equivariant")

    print('\nshould not be invariant or equivariant')
    print("Augmentated Model Equivariance:", augmentated_equiv)
    print("Augmentated Model Invariance:", augmentated_invar)
    if augmentated_equiv or augmentated_invar:
        raise ValueError("Augmentated model should not be invariant or equivariant")

    print('\nshould be equivariant')
    print("Symetric Model Equivariance:", symetric_equiv)
    print("Symetric Model Invariance:", symetric_invar)
    if not (symetric_equiv and not symetric_invar):
        raise ValueError("Symetric model should be equivariant and not invariant")

    print('\nshould be invariant')
    print("Invariant Model Equivariance:", invariant_equiv)
    print("Invariant Model Invariance:", invariant_invar)
    if not (invariant_equiv and invariant_invar):
        raise ValueError("Invariant model should be invariant and equivariant")
