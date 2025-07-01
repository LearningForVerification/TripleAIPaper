import torch
import torch.nn as nn
from torch import Tensor
import abc
import copy
from abc import abstractmethod
import torch


################################# LINEAR FUNCTIONS #############################################################

class LinearFunctions:
    """
    A class representing a set of n linear functions f(i) of m input variables x

    matrix is an (n x m) Tensor
    offset is an (n) Tensor

    f(i) = matrix[i]*x + offset[i]

    """

    def __init__(self, matrix: torch.Tensor, offset: torch.Tensor):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def __repr__(self):
        return "LinearFunctions({})".format(self.size)

    def clone(self):
        return LinearFunctions(copy.deepcopy(self.matrix), copy.deepcopy(self.offset))

    def mask_zero_outputs(self, zero_outputs):
        mask = torch.diag(
            torch.Tensor([0 if neuron_n in zero_outputs else 1 for neuron_n in range(self.size)])
        )
        return LinearFunctions(torch.matmul(mask, self.matrix), torch.matmul(mask, self.offset))

    def get_size(self) -> int:
        return self.size

    def get_matrix(self) -> torch.Tensor:
        return self.matrix

    def get_offset(self) -> torch.Tensor:
        return self.offset

    def compute_max_values(self, input_bounds) -> torch.Tensor:
        return torch.matmul(torch.clamp(self.matrix, min=0), input_bounds.get_upper()) + \
            torch.matmul(torch.clamp(self.matrix, max=0), input_bounds.get_lower()) + \
            self.offset

    def compute_min_values(self, input_bounds) -> torch.Tensor:
        return torch.matmul(torch.clamp(self.matrix, min=0), input_bounds.get_lower()) + \
            torch.matmul(torch.clamp(self.matrix, max=0), input_bounds.get_upper()) + \
            self.offset


######################################## ABSTRACT BOUNDS #########################################

class AbstractBounds(abc.ABC):
    """
    Abstract class that defines the abstraction of lower and upper bounds for a neural network layer

    Attributes
    ----------
    lower: Any
        The lower bounds
    upper: Any
        The upper bounds
    size: int
        The number of dimensions of the lower and upper bounds
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.size = self.get_size()

    def __repr__(self):
        return ', '.join(["({}, {})".format(self.lower[i], self.upper[i]) for i in range(self.size)])

    @abstractmethod
    def get_lower(self):
        raise NotImplementedError

    @abstractmethod
    def get_upper(self):
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError


class HyperRectangleBounds(AbstractBounds):
    """
    Class that defines the hyper-rectangle bounds for a neural network layer, i.e.,
    bounding the variables with individual lower and upper bounds.

    Methods
    -------
    get_dimension_bounds(int)
        Procedure to get the bounds for a specific dimension
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        super(HyperRectangleBounds, self).__init__(lower, upper)

    def __repr__(self):
        return ', '.join(["(lb[{}]: {:.5f}, ub[{}]: {:.5f})".format(i, self.lower[i], i, self.upper[i])
                          for i in range(self.size)])

    def get_lower(self) -> torch.Tensor:
        return self.lower

    def get_upper(self) -> torch.Tensor:
        return self.upper

    def get_size(self) -> int:
        return len(self.lower)

    def clone(self):
        return HyperRectangleBounds(copy.deepcopy(self.lower), copy.deepcopy(self.upper))

    def get_dimension_bounds(self, dim: int) -> tuple[float, float]:
        """Procedure to get the bounds for a specific dimension"""
        if 0 <= dim < self.size:
            return self.lower[dim].item(), self.upper[dim].item()
        else:
            raise Exception("Dimension {} is out of range for size {}".format(dim, self.size))


class SymbolicLinearBounds(AbstractBounds):
    """
    Class that defines the symbolic linear bounds for a neural network layer, i.e.,
    the linear equations for the lower and upper bounds.

    Methods
    -------
    get_upper_bounds(HyperRectangleBounds) -> torch.Tensor
        Procedure to compute the numeric upper bounds
    get_lower_bounds(HyperRectangleBounds) -> torch.Tensor
        Procedure to compute the numeric lower bounds
    get_all_bounds(HyperRectangleBounds) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Procedure to compute all bounds
    to_hyper_rectangle_bounds(HyperRectangleBounds) -> HyperRectangleBounds
        Procedure to compute the hyper-rectangle bounds
    """

    def __init__(self, lower: LinearFunctions, upper: LinearFunctions):
        super(SymbolicLinearBounds, self).__init__(lower, upper)

    def get_lower(self) -> LinearFunctions:
        return self.lower

    def get_upper(self) -> LinearFunctions:
        return self.upper

    def get_size(self) -> int:
        return self.lower.get_size()

    def get_upper_bounds(self, input_bounds: HyperRectangleBounds) -> torch.Tensor:
        """Procedure to compute the numeric upper bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return self.upper.compute_max_values(input_bounds)

    def get_lower_bounds(self, input_bounds: HyperRectangleBounds) -> torch.Tensor:
        """Procedure to compute the numeric lower bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return self.lower.compute_min_values(input_bounds)

    def get_all_bounds(self, input_bounds: HyperRectangleBounds) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Procedure to compute all bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return self.lower.compute_min_values(input_bounds), \
            self.lower.compute_max_values(input_bounds), \
            self.upper.compute_min_values(input_bounds), \
            self.upper.compute_max_values(input_bounds)

    def to_hyper_rectangle_bounds(self, input_bounds: HyperRectangleBounds) -> HyperRectangleBounds:
        """Procedure to compute the hyper-rectangle bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return HyperRectangleBounds(self.lower.compute_min_values(input_bounds),
                                    self.upper.compute_max_values(input_bounds))



#############################################################################################


class BoundsManager:
    """
    This class manages the symbolic bounds propagation framework for NeVer2.
    It is designed to handle feed-forward neural networks as computational graphs and can be instantiated
    either with fixed lower and upper bounds or with a structured verification property.

    Attributes
    ----------
    ref_nn: NeuralNetwork
        The reference NN that defines the structure of the graph
    abs_nn: AbsNeuralNetwork
        The abstract NN that contains the abstraction of the layers
    topological_stack: list[str]
        The topological sort of the layers in the NN used for the propagation
    direction: BoundsDirection
        The direction in which the bounds are computed, either forwards or backwards
    bounds_dict: VerboseBounds
        The data structure storing all bounds information
    input_bounds: HyperRectangleBounds
        The input bounds to propagate
    statistics: BoundsStats
        Statistics about neurons stability

    Methods
    ----------
    init_symbolic_bounds()
        Procedure to set up the initial symbolic bounds
    propagate_bounds(HyperRectangleBounds | None, SymbolicLinearBounds | None, LayerNode | None)
        Recursive procedure to propagate the bounds. When invoked as a root level, all parameters are None
    update_stats(AbsLayerNode, HyperRectangleBounds)
        Procedure to update statistics
    """

    def __init__(self, model: torch.nn.Sequential, input_bounds: tuple[torch.Tensor, torch.Tensor]):

        self.weights = []
        self.biases = []
        self.input_bounds = HyperRectangleBounds(input_bounds[0], input_bounds[1])

        for layer in model.hidden_layers:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight)
                self.biases.append(layer.bias)

    def init_symbolic_bounds(self) -> SymbolicLinearBounds:
        """Initialize the input symbolic linear bounds"""
        input_size = self.input_bounds.get_size()
        lower_equation = LinearFunctions(torch.eye(input_size), torch.zeros(input_size))
        upper_equation = LinearFunctions(torch.eye(input_size), torch.zeros(input_size))

        return SymbolicLinearBounds(lower_equation, upper_equation)

    def compute_bounds_hidden_layers(self, in_sym_bounds = None, in_num_bounds = None) -> list:
        """
        Entry point

        N.B. inside the propagation we use abstract layers but with their concrete counterpart identifier
        """

        bounds_list = list()

        # Number of hidden layers
        num_hidden_layers = len(self.weights)

        assert num_hidden_layers == len(self.biases)


        if in_sym_bounds is None:
            in_sym_bounds = self.init_symbolic_bounds()
        if in_num_bounds is None:
            in_num_bounds = self.input_bounds



        # Current layer data
        cur_sym_bounds = in_sym_bounds
        cur_num_bounds = in_num_bounds

        for index in range(num_hidden_layers):
            symbolic_in = compute_dense_output_bounds(self.weights[index], self.biases[index], cur_sym_bounds)
            numeric_in = symbolic_in.to_hyper_rectangle_bounds(self.input_bounds)

            relu_lin = LinearizeReLU(fixed_neurons={}, input_hyper_rect=self.input_bounds)

            symbolic_out = relu_lin.compute_output_linear_bounds(symbolic_in)
            numeric_out = compute_output_numeric_bounds(numeric_in, symbolic_in)

            cur_sym_bounds = symbolic_out
            cur_num_bounds = numeric_out

            bounds_list.append(cur_num_bounds)

        return bounds_list







def compute_lower(weights_minus, weights_plus, input_lower, input_upper):
    """Procedure that computes the matrix of coefficients for a lower bounds linear function.

    Parameters
    ----------
    weights_minus: Tensor
        The negative part of the weights
    weights_plus: Tensor
        The positive part of the weights
    input_lower: Tensor
        The lower input bounds
    input_upper: Tensor
        The upper input bounds

    Returns
    -----------
    Tensor
        The lower bounds matrix
    """
    return torch.matmul(weights_plus, input_lower) + torch.matmul(weights_minus, input_upper)




def compute_upper(weights_minus, weights_plus, input_lower, input_upper):
    """Procedure that computes the matrix of coefficients for an upper bounds linear function.

    Parameters
    ----------
    weights_minus: Tensor
        The negative part of the weights
    weights_plus: Tensor
        The positive part of the weights
    input_lower: Tensor
        The lower input bounds
    input_upper: Tensor
        The upper input bounds

    Returns
    -----------
    Tensor
        The upper bounds matrix
    """
    return torch.matmul(weights_plus, input_upper) + torch.matmul(weights_minus, input_lower)


def compute_dense_output_bounds(weight, bias, inputs: SymbolicLinearBounds) -> SymbolicLinearBounds:
    """Compute the forwards symbolic output bounds for the layer

    Parameters
    ----------
    layer: FullyConnectedNode
        The linear layer
    inputs: SymbolicLinearBounds
        The input symbolic bounds

    Returns
    ----------
    SymbolicLinearBounds
        The symbolic output bounds for the layer
    """
    weights_plus = torch.clamp(weight, min=0)
    weights_minus = torch.clamp(weight, max=0)

    lm = inputs.get_lower().get_matrix()
    um = inputs.get_upper().get_matrix()
    lo = inputs.get_lower().get_offset()
    uo = inputs.get_upper().get_offset()

    lower_matrix = compute_lower(weights_minus, weights_plus, lm, um)
    lower_offset = compute_lower(weights_minus, weights_plus, lo, uo) + bias
    upper_matrix = compute_upper(weights_minus, weights_plus, lm, um)
    upper_offset = compute_upper(weights_minus, weights_plus, lo, uo) + bias

    return SymbolicLinearBounds(LinearFunctions(lower_matrix, lower_offset),
                                LinearFunctions(upper_matrix, upper_offset))



###################################### OLD BP ###########################################

def compute_output_numeric_bounds(cur_numeric_bounds: HyperRectangleBounds,
                                  cur_symbolic_bounds: SymbolicLinearBounds) -> HyperRectangleBounds:
    """
    Compute the numeric post-activation bounds of the linearized ReLU function
    using the information about currently inactive neurons

    """

    cur_layer_output_num_bounds = HyperRectangleBounds(
        torch.max(cur_numeric_bounds.get_lower(), torch.zeros(cur_numeric_bounds.get_size())),
        torch.max(cur_numeric_bounds.get_upper(), torch.zeros(cur_numeric_bounds.get_size())))


    return cur_layer_output_num_bounds


class LinearizeReLU:
    """
    This class provides the linearization for the ReLU function enhanced by information
    about currently active and inactive neurons

    """

    USE_FIXED_NEURONS = False

    def __init__(self, fixed_neurons: dict, input_hyper_rect: HyperRectangleBounds):
        self.fixed_neurons = fixed_neurons
        self.input_hyper_rect = input_hyper_rect

    def compute_output_linear_bounds(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:

        lower_l, lower_u, upper_l, upper_u = input_eq.get_all_bounds(self.input_hyper_rect)
        lower, upper = LinearizeReLU.compute_symb_lin_bounds_equations(input_eq, lower_l, lower_u, upper_l, upper_u)

        return SymbolicLinearBounds(lower, upper)

    @staticmethod
    def compute_relu_equation(preact_num_lower, preact_num_upper):
        lower_relu_eq, postact_lower = LinearizeReLU.get_relu_relax_lower_bound_equation(preact_num_lower,
                                                                                         preact_num_upper)
        upper_relu_eq, postact_upper = LinearizeReLU.get_relu_relax_upper_bound_equation(preact_num_lower,
                                                                                         preact_num_upper)

        return SymbolicLinearBounds(lower_relu_eq, upper_relu_eq), HyperRectangleBounds(postact_lower, postact_upper)

    @staticmethod
    def get_relu_relax_lower_bound_equation(preact_lower_bounds, preact_upper_bounds):
        """
        The lower bound of unstable nodes is either 0, or
        the linear relaxation of the preactivation (hence, the slope).

        The latter is the case when the upper bound is greater than or equal to the absolute value of the lower bound,
        thus resulting in a triangle of smaller area than the one formed by 0.

        The former is the case when the absolute value of the lower bound is greater than the upper bound,
        thus resulting is a triangle of smaller area than the one formed by the slope.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = torch.eye(size)
        offset = torch.zeros(size)

        postact_lower_bounds = Tensor(preact_lower_bounds)

        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the lower bound is exactly the preactivation
                # it remains 1
                pass

            elif preact_upper_bounds[i] >= -preact_lower_bounds[i]:
                # Unstable node, lower bound is linear relaxation of the equation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i][i] = k
                postact_lower_bounds[i] *= k

            else:  # upper[i] <= 0 (inactive node)
                # or
                # -lower[i] > upper[i]
                # lower bound is 0
                matrix[i][i] = 0
                postact_lower_bounds[i] = 0

        return LinearFunctions(matrix, offset), postact_lower_bounds

    @staticmethod
    def get_relu_relax_upper_bound_equation(preact_lower_bounds, preact_upper_bounds):
        """
        Compute the resulting upper bound equation after relaxing ReLU,
        qiven a preactivation upper bound equation.

        input_bounds are required for computing the concrete bounds.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = torch.eye(size)
        offset = torch.zeros(size)

        postact_upper_bounds = Tensor(preact_upper_bounds)
        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the upper bound is exactly the preactivation
                # it remains 1
                pass

            elif preact_upper_bounds[i] >= 0:
                # Unstable node - linear relaxation of preactivation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i][i] = k
                offset[i] = - preact_lower_bounds[i] * k

            else:  # preact_upper_bounds[i] <= 0 (inactive node)
                # The upper bound is 0
                matrix[i][i] = 0
                postact_upper_bounds[i] = 0

        return LinearFunctions(matrix, offset), postact_upper_bounds


    @staticmethod
    def compute_symb_lin_bounds_equations(inputs, lower_l, lower_u, upper_l, upper_u):
        k_lower, b_lower = LinearizeReLU.get_array_lin_lower_bound_coefficients(lower_l, lower_u)
        k_upper, b_upper = LinearizeReLU.get_array_lin_upper_bound_coefficients(upper_l, upper_u)

        lower_matrix = LinearizeReLU.get_transformed_matrix(inputs.get_lower().get_matrix(), k_lower)
        upper_matrix = LinearizeReLU.get_transformed_matrix(inputs.get_upper().get_matrix(), k_upper)

        lower_offset = LinearizeReLU.get_transformed_offset(inputs.get_lower().get_offset(), k_lower, b_lower)
        upper_offset = LinearizeReLU.get_transformed_offset(inputs.get_upper().get_offset(), k_upper, b_upper)

        lower = LinearFunctions(lower_matrix, lower_offset)
        upper = LinearFunctions(upper_matrix, upper_offset)

        return lower, upper

    @staticmethod
    def get_transformed_matrix(matrix, k):
        return matrix * k[:, None]

    @staticmethod
    def get_transformed_offset(offset, k, b):
        return offset * k + b

    @staticmethod
    def get_array_lin_lower_bound_coefficients(lower, upper):
        ks = torch.zeros(len(lower))
        bs = torch.zeros(len(lower))

        for i in range(len(lower)):
            k, b = LinearizeReLU.get_lin_lower_bound_coefficients(lower[i], upper[i])
            ks[i] = k
            bs[i] = b

        return ks, bs

    @staticmethod
    def get_array_lin_upper_bound_coefficients(lower, upper):
        ks = torch.zeros(len(lower))
        bs = torch.zeros(len(lower))

        for i in range(len(lower)):
            k, b = LinearizeReLU.get_lin_upper_bound_coefficients(lower[i], upper[i])
            ks[i] = k
            bs[i] = b

        return ks, bs

    @staticmethod
    def get_lin_lower_bound_coefficients(lower, upper):
        if lower >= 0:
            return 1, 0

        if upper >= - lower:
            mult = upper / (upper - lower)
            return mult, 0

        # upper <= 0:
        # or
        # -lower > upper, i.e., 0 is a tighter lower bound that the slope mult above
        return 0, 0

    @staticmethod
    def get_lin_upper_bound_coefficients(lower, upper):
        if lower >= 0:
            return 1, 0

        if upper <= 0:
            return 0, 0

        mult = upper / (upper - lower)
        add = -mult * lower

        return mult, add

def create_test_network():
    """Create a simple test neural network"""
    model = torch.nn.Sequential()
    model.hidden_layers = torch.nn.ModuleList([
        torch.nn.Linear(784, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 10)
    ])
    return model


def main():
    # Create test network
    test_net = create_test_network()

    # Create 10 random input bounds
    input_bounds_list = []
    for i in range(10):
        input_lower = torch.rand(784)  # Random values between 0 and 1 
        input_upper = input_lower + 0.015  # Add epsilon=0.015
        input_upper = torch.clamp(input_upper, max=1.0)  # Clamp to ensure values <= 1
        input_bounds_list.append((input_lower, input_upper))

    # Process each input bound and store results
    for i, bounds in enumerate(input_bounds_list):
        print(f"\nProcessing input bounds {i + 1}:")

        # Create bounds manager
        bounds_mgr = BoundsManager(test_net, bounds)

        # Compute bounds through hidden layers
        bounds_list = bounds_mgr.compute_bounds_hidden_layers()

        # Print results
        print("Bounds for each layer:")
        for j, layer_bounds in enumerate(bounds_list):
            print(f"Layer {j}:")
            print(f"Lower bounds: {layer_bounds.get_lower()}")
            print(f"Upper bounds: {layer_bounds.get_upper()}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
