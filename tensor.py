import numpy as np
import matplotlib.pyplot as plt

def als(c, l, epsilon=1e-2, max_iter = 100):
    """Solve argmin(0.5 x^T * C * x - l * x) where x is a second order tensor,
    C is a second order tensor operator and l is a second order tensor."""

    iter_count = 0

    # We check that c and l have appropriate dimensions
    assert c.dim_count == 2 and l.dim_count == 2, "ALS only works with second order tensors!"

    # We extract the size of the solution
    Nx = l.get_size(0)
    Nv = l.get_size(0)

    # We generate random vectors for starters
    xm = np.random.rand(Nx)
    vm = np.random.rand(Nv)
    xm_old = np.zeros(Nx)
    vm_old = np.zeros(Nv)
    norm_evolution = 1.

    while iter_count < max_iter and norm_evolution > epsilon:
        iter_count += 1
        xm_old = xm
        vm_old = vm

        left_measure = lambda x: np.dot(np.dot(x, xm), xm)
        left_operator = c.integrate(0, left_measure).untensorized()
        left_measure = lambda x: np.dot(x, xm)
        left_operator_2 = l.integrate(0, left_measure).untensorized()

        try:
            vm = np.linalg.solve(left_operator, left_operator_2)
        except:
            vm = np.zeros(Nx)

        right_measure = lambda v: np.dot(np.dot(v, vm), vm)
        right_operator = c.integrate(1, right_measure).untensorized()
        right_measure = lambda v: np.dot(v, vm)
        right_operator_2 = l.integrate(1, right_measure).untensorized()

        try:
            xm = np.linalg.solve(right_operator, right_operator_2)
        except:
            xm = np.zeros(Nv)

        norm_evolution = abs(xm.dot(xm) * vm.dot(vm) - xm_old.dot(xm_old) * vm_old.dot(vm_old))
    if iter_count >= max_iter:
        # print("ALS diverge!")
        pass
    return Tensor.from_arrays([xm, vm])


class Tensor:
    """Interface to do some basic tensor computing."""

    def __init__(self, sizes):
        """DIM: INT tensor order, sizes: LIST of INT the dimension of each order.
        Example: I want to represent a function of R^2 with discretised space 256 and 128.
        I can create a dim = 2, sizes = [256, 128] Tensor.
        The sizes are fixed at Tensor creation."""
        super().__init__()

        self.dim = [[] for _ in range(len(sizes))]
        self.sizes = sizes

    @property
    def rank(self):
        return len(self.dim[0])

    @property
    def dim_count(self):
        return len(self.sizes)

    def eval_rank(self, at, rank):
        result = 1.
        for d in range(self.dim_count):
            result *= self.dim[d][rank][at[d]]

        return result

    def __iadd__(self, other):
        assert self.sizes == other.sizes, "Mismatch sizes."

        self = (self + other)
        return self

    def __add__(self, other):
        # assert self.sizes == other.sizes, "Mismatch sizes."

        result_tensor = Tensor(self.sizes)
        for d in range(self.dim_count):
            for k in range(self.rank):
                result_tensor.dim[d].append(np.copy(self.dim[d][k]))
            for k in range(other.rank):
                result_tensor.dim[d].append(np.copy(other.dim[d][k]))
        return result_tensor

    def __sub__(self, other):
        return self + (-other)

    def __call__(self, at):
        assert len(at) == self.dim_count, "Mismatch sizes."

        result = 0.
        for k in range(self.rank):
            result += self.eval_rank(at, k)

        return result

    def __neg__(self):
        return -1. * self

    def __rmul__(self, other):
        tensor = Tensor(self.sizes)

        for k in range(self.rank):
            tensor.dim[0].append(other * np.copy(self.dim[0][k]))
            for d in range(1, self.dim_count):
                tensor.dim[d].append(np.copy(self.dim[d][k]))

        return tensor

    def get_size(self, dim):
        """To counter numpy weird behavior of returning one dim tuple instead of int."""
        if isinstance(self.sizes[dim], tuple) and len(self.sizes[dim]) == 1:
            return self.sizes[dim][0]
        else:
            return self.sizes[dim]

    def apply(self, other):
        result = Tensor(other.sizes)

        for k_self in range(self.rank):
            for k_other in range(other.rank):
                for d in range(self.dim_count):
                    result.dim[d].append(np.dot(self.dim[d][k_self], other.dim[d][k_other]))

        return result

    def __mul__(self, other):
        assert self.dim_count == other.dim_count, "Dimensions missmatch."
        assert self.rank > 0 and other.rank > 0, "Need at least rank 1"

        if self.sizes != other.sizes:
        # if isinstance(self.sizes[0], tuple) and len(self.sizes[0]) > 1:
           return self.apply(other)

        result = 0.
        for k_self in range(self.rank):
            for k_other in range(other.rank):
                temp_result = 1.
                for d in range(self.dim_count):
                    temp_result *= np.dot(self.dim[d][k_self], other.dim[d][k_other])
                result += temp_result

        return result

    def norm(self):
        # O(Rank ^ 2 * dimension)
        # Careful! Seems to be precision problem with this method (around 1e-7)
        return abs(self * self) ** 0.5

    def normalize(self):

        self = (1. / self.norm()) * self
    
    def untensorized(self):
        """Only work with dim 2 and vector value. Return equivalent full size
        matrix."""
        assert self.dim_count <= 2, "Can't untensorized higher order tensor."

        if self.dim_count == 2:
            equivalent_mat = np.outer(self.dim[0][0], self.dim[1][0])
            for k in range(1, self.rank):
                equivalent_mat += np.outer(self.dim[0][k], self.dim[1][k])

            return equivalent_mat
        else:
            equivalent_vector = self.dim[0][0]

            for k in range(1, self.rank):
                equivalent_vector += self.dim[0][k]

            return equivalent_vector

    def integrate(self, dim, measure):
        """Integrate over DIM with MEASURE.
        It returns a dim_count - 1 tensor.
        measure should be that takes values in size[dim] and returns a scalar."""

        sizes = self.sizes[:dim] + self.sizes[dim+1:]
        result_tensor = Tensor(sizes)

        result_coeffs = []

        for k in range(self.rank):
            result_coeffs.append(measure(self.dim[dim][k]))

        # Special case
        if dim == 0:
            for k in range(self.rank):
                result_tensor.dim[0].append(result_coeffs[k] * self.dim[1][k])
            for d in range(2, self.dim_count):
                result_tensor.dim[d-1].append(self.dim[d][k])
        else:
            for k in range(self.rank):
                result_tensor.dim[0].append(result_coeffs[k] * self.dim[0][k])
            for d in range(1, dim):
                result_tensor.dim[d].append(self.dim[d][k])
            for d in range(dim+1, self.dim_count):
                result_tensor.dim[d-1].append(self.dim[d][k])
        return result_tensor

    def is_vector(self):
        """Return True if Tensor is a vector, False otherwise."""

        return True

    def is_operator(self):
        """Return True if Tensor is an operator, False otherwise."""

        return not self.is_vector()

    def draw(self):
        assert self.is_vector() and self.dim_count == 2, "Can't draw higher dimension Tensor."

        self_mat = self.untensorized()
        plot, ax = plt.subplots()

        ax.imshow(self_mat)
        plot.show()
    def compress(self, epsilon=1e-3, max_iter = 50):
        """Use POD algorithm in combination with ALS to return a compressed version of the tensor."""

        assert self.dim_count == 2 and self.is_vector()

        identity = Tensor.eye(self.sizes)
        self_copy = 1. * self
        iter_count = 0
        norm_evolution = 1.
        tensor_compressed = Tensor(self.sizes)

        while iter_count < max_iter and norm_evolution > epsilon:
            residue = als(identity, self_copy)
            self_copy = self_copy - residue
            tensor_compressed = tensor_compressed + residue

            norm_evolution = residue.norm()
            iter_count += 1

        return tensor_compressed
    @staticmethod
    def zeros(sizes, rank):
        null_tensor = Tensor(sizes)
        for k in range(rank):
            for d in range(null_tensor.dim_count):
                null_tensor.dim[d].append(np.zeros(null_tensor.sizes[d]))

        return null_tensor

    @staticmethod
    def ones(sizes, rank):
        ones_tensor = Tensor(sizes)
        for k in range(rank):
            for d in range(ones_tensor.dim_count):
                ones_tensor.dim[d].append(np.ones(ones_tensor.sizes[d]))

        return ones_tensor

    @staticmethod
    def rand(sizes, rank):
        rand_tensor = Tensor(sizes)
        for k in range(rank):
            for d in range(rand_tensor.dim_count):
                rand_tensor.dim[d].append(np.random.rand(rand_tensor.sizes[d]))

        return rand_tensor

    @staticmethod
    def from_functions(functions, grids):
        """Create a rank 1 tensor populated with functions evaluated in grids.
        Functions should be vectorized."""
        assert len(functions) == len(grids), "Should have one grid per function."
        sizes = [len(grid) for grid in grids]

        functions_tensor = Tensor(sizes)
        for d in range(functions_tensor.dim_count):
            function_vectorized = np.vectorize(functions[d])
            functions_tensor.dim[d].append(function_vectorized(grids[d]))

        return functions_tensor

    @staticmethod
    def from_arrays(arrays):
        """Create a rank 1 tensor from a list of arrays."""
        sizes = [np.shape(array) for array in arrays]

        arrays_tensor = Tensor(sizes)
        for d in range(arrays_tensor.dim_count):
            arrays_tensor.dim[d].append(arrays[d])

        return arrays_tensor

    @staticmethod
    def eye(sizes):
        eye_tensor = Tensor.from_arrays([np.eye(size) for size in sizes])
        return eye_tensor
