import unittest
from tensor import Tensor
from tensor import als
import numpy as np


class TestTensor(unittest.TestCase):
    def test_zeros(self):
        null_tensor = Tensor.zeros([128, 256], 12)

        self.assertEqual(null_tensor.dim[1][10][232], 0., "Should be 0.")
        self.assertEqual(null_tensor((21, 234)), 0., "Should also be 0.")

    def test_eval(self):
        rank = 128
        ones_tensor = Tensor.ones([1, 1], rank)

        self.assertEqual(ones_tensor((0, 0)), rank, f"Should be {rank}.")
        self.assertEqual(ones_tensor([0, 0]), rank, f"Should be {rank}.")

    def test_sum(self):
        rank1 = 12
        rank2 = 24
        tensor1 = Tensor.rand([12, 12], rank1)
        tensor2 = Tensor.rand([12, 12], rank2)

        at = [3, 2]

        self.assertEqual(tensor1.rank + tensor2.rank, (tensor1 + tensor2).rank, "Rank should add whend summing two tensors.")
        self.assertAlmostEqual(tensor1(at) + tensor2(at), (tensor1 + tensor2)(at),
                         msg="Evaluation of sum at should be the same as sum of evaluation at.")
        before_inplace_sum = tensor1(at) + tensor2(at)
        tensor1 += tensor2

        self.assertAlmostEqual(before_inplace_sum, tensor1(at),
                               msg="Evaluation of sum at should be the same as sum of evaluation at.")

    def test_from_functions(self):
        def f0(x):
            return x ** 2

        def f1(v):
            return 3. * v

        gridx = np.linspace(0., 10., 128)
        gridv = np.linspace(-10., 10., 256)

        f0 = np.vectorize(f0)
        f1 = np.vectorize(f1)

        functions_tensor = Tensor.from_functions([f0, f1], [gridx, gridv])
        self.assertEqual(functions_tensor.rank, 1, "Functions tensor rank should be one.")
        self.assertTrue((functions_tensor.untensorized() == np.outer(f0(gridx), f1(gridv))).all(),
                               msg="Functions under matrix form and tensor form should be equivalent.")

    def test_from_arrays(self):
        # Tests for vectors
        x = np.linspace(0., 10., 128)
        v = np.linspace(-10., 10., 128)

        arrays_tensor = Tensor.from_arrays([x, v])
        self.assertEqual(arrays_tensor.rank, 1, "Arrays tensor rank should be one.")
        self.assertTrue((arrays_tensor.untensorized() == np.outer(x, v)).all(),
                               msg="Arrays tensor under matrix form and tensor form should be equivalent.")

        # Tests for matrices
        A = np.diag(np.random.rand(24))
        B = np.diag(np.random.rand(28))

        arrays_tensor = Tensor.from_arrays([A, B])
        self.assertEqual(arrays_tensor.rank, 1, "Rank should be one.")

    def test_product(self):
        # Test vector against vector i.e. scalar product
        # 0 test
        tensor_a = Tensor.zeros((128, 256), 2)
        tensor_b = Tensor.zeros((128, 256), 5)

        self.assertAlmostEqual(tensor_a * tensor_b, 0., msg="Should be 0.")
        self.assertAlmostEqual(tensor_a * tensor_a, 0., msg="Should be 0.")

        # Norm test
        tensor_a = Tensor.rand((128, 256), 12)
        mat_a = tensor_a.untensorized()
        norm_a = tensor_a.norm()

        self.assertAlmostEqual(norm_a, np.linalg.norm(mat_a),
                               msg="Scalar product should be equivalent in tensor form and in matrice form.")

        # Test matrices against vector i.e. operator
        # Identity test
        tensor_a = Tensor.rand((128, 256), 5)
        operator_a = Tensor.from_arrays([np.eye(128), np.eye(256)])
        result_tensor = tensor_a - operator_a * tensor_a

        self.assertAlmostEqual(result_tensor.norm(), 0, 5,
                               msg="This operator should do nothing.")

        # Null test
        operator_b = Tensor.from_arrays([np.zeros((128, 128)), np.zeros((256, 256))])
        result_tensor = operator_b * tensor_a

        self.assertAlmostEqual(result_tensor.norm(), 0.,
                               msg="This operator should nullify the tensor.")

        # Distributivity
        tensor_b = Tensor.rand((128, 256), 7)
        result_tensor_1 = (operator_a + operator_b) * (tensor_a + tensor_b)
        result_tensor_2 = (operator_a * tensor_a) + (operator_a * tensor_b) + (operator_b * tensor_a) + (operator_b * tensor_b)

        result_tensor = result_tensor_2 - result_tensor_1

        self.assertAlmostEqual(result_tensor.norm(), 0., 5,
                               msg="Add and mult should be distributive.")

    def test_norm(self):
        tensor_a = Tensor.from_arrays([np.ones(10), np.zeros(25)])

        self.assertAlmostEqual(tensor_a.norm(), 0., msg="Should be 0.")

        tensor_b = Tensor.from_arrays([np.zeros(10), np.ones(25)])
        res_tensor = tensor_a + tensor_b

        self.assertAlmostEqual(res_tensor.norm(), 0., msg="Should be 0.")

        tensor_a = Tensor.from_arrays([np.ones(10), np.ones(25)])
        tensor_b = Tensor.from_arrays([-np.ones(10), np.ones(25)])
        res_tensor = tensor_a + tensor_b
        res_mat = res_tensor.untensorized()

        self.assertAlmostEqual(np.linalg.norm(res_mat), 0., msg="Should be 0.")
        self.assertAlmostEqual(res_tensor.norm(), 0., msg="Should be 0.")

        tensor_a = Tensor.rand((10, 25), 130)
        tensor_b = (-tensor_a)

        res_tensor = tensor_a + tensor_b
        res_mat = res_tensor.untensorized()
        self.assertAlmostEqual(np.linalg.norm(res_mat), 0., msg="Should be 0.")
        self.assertAlmostEqual(res_tensor.norm(), 0., 5, msg="Should be 0.")


    def test_scalar_product(self):
        tensor_a = Tensor.rand((256, 128), 25)
        scalar = 12.

        tensor_res = scalar * tensor_a

        self.assertAlmostEqual(tensor_a.norm() * scalar, tensor_res.norm(),
                               msg="Should have the same norm.")

    def test_als(self):
        operator = Tensor.from_arrays([np.eye(128), np.eye(256)])
        vector = Tensor.from_arrays([np.ones(128), np.ones(256)])

        result = als(operator, vector)

        self.assertIsNotNone(result, "Should be a tensor.")
    def test_integrate(self):
        tensor_a = Tensor.rand((256, 128), 12)
        tensor_res = tensor_a.integrate(0, lambda vec: sum(vec))

        self.assertEqual(tensor_res.dim_count, 1, "Should have dim 1.")
        vector_res = tensor_res.untensorized()
        self.assertEqual(np.shape(vector_res), (128,), "Should be a vector.")

        Nx = 128
        Nv = 256

        def fx(x):
            return 1.

        def fv(v):
            return np.cos(v)

        gridx = np.linspace(0., 1., Nx)
        gridv = np.linspace(0., 2 * np.pi, Nv)

        # TODO: Write integration test. (and not integration test ;))


if __name__ == '__main__':
    unittest.main()
