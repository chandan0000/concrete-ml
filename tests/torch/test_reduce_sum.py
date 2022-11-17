"""Tests for ReduceSum operator on a Torch model."""

import numpy
import pytest
import torch
from torch import nn

from concrete.ml.torch.compile import compile_torch_model
from concrete.ml.torch.numpy_module import NumpyModule


class _TorchReduceSum(nn.Module):
    """Torch model to test ReduceSum."""

    def __init__(self, dim=1, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        """Forward pass."""
        return (
            x.sum(dim=self.dim, keepdim=self.keepdim)
            if self.dim is not None
            else x.sum()
        )


def get_numpy_input_and_inputset(n_values, max_value, n_samples=1, signed=False):
    """Create the input values needed for testing ReduceSum."""
    if signed:
        max_value = max_value // 2
        min_value = -max_value
    else:
        min_value = 0.0

    # Set an array of n_values integers with values in [min_value, max_value[
    numpy_input = numpy.random.randint(
        low=min_value, high=max_value, size=(n_samples, n_values)
    ).astype(numpy.float64)

    # Initialize the inputset for compilation
    if n_samples > 1:
        inputset = numpy.random.randint(
            low=min_value, high=max_value, size=(1, n_samples, n_values)
        ).astype(numpy.float64)

    # Setting the inputset with extreme values in order to make sure the scale and zero point values
    # will be computed correctly during the quantization and dequantization process (for inputs as
    # well as outputs)
    else:
        inputset = numpy.tile(
            numpy.array([[min_value], [max_value - 1]], dtype=numpy.float64), (1, n_values)
        )

    return numpy_input, inputset


def execute_reduce_sum(
    torch_model,
    numpy_input,
    inputset,
    n_bits,
    in_fhe,
    use_virtual_lib,
    configuration,
):
    """Execute a Torch model using ReduceSum."""

    # Execute the torch model in FHE
    if in_fhe:
        if isinstance(n_bits, int):
            n_bits_dict = {
                "model_inputs": n_bits,
                "op_weights": n_bits,
                "op_inputs": n_bits,
                "model_outputs": n_bits,
            }

        # Compile the torch model
        quantized_numpy_module = compile_torch_model(
            torch_model,
            inputset,
            configuration=configuration,
            use_virtual_lib=use_virtual_lib,
            n_bits=n_bits_dict,
        )

        # Check that no error comes from the quantization process
        quantizer = quantized_numpy_module.input_quantizers[0]

        assert quantizer.scale == 1.0 and quantizer.zero_point == 2 ** (
            n_bits - 1
        ), "Wrong quantization of inputs: should be 'one to one'."

        # Quantize the input
        q_input = quantized_numpy_module.quantize_input(numpy_input)

        if not isinstance(q_input, tuple):
            q_input = (q_input,)

        assert quantized_numpy_module.is_compiled, "Torch model is not compiled"

        # Execute the sum in FHE
        q_result = quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*q_input)

        # Dequantize the output
        return quantized_numpy_module.dequantize_output(q_result)

    # Convert the torch model to a NumpyModule
    numpy_module = NumpyModule(
        torch_model,
        dummy_input=torch.from_numpy(numpy_input),
    )

    # Compute the sum
    return numpy_module(numpy_input)


def generate_test_sum_cases():
    """Generate all tests cases use in the test_sum."""

    def generate_sum_id(n_values, n_bits, in_fhe, use_virtual_lib):
        return (
            f"reduce_sum_{n_values}_values_{n_bits}_bits"
            + "_in_fhe" * in_fhe
            + "_(VL)" * use_virtual_lib
        )

    tests_cases = []

    # One test in FHE
    tests_cases.append(
        pytest.param(
            2 ** (numpy.random.randint(8)),
            4,
            True,
            False,
            id=generate_sum_id(2 ** (numpy.random.randint(8)), 4, True, False),
        )
    )

    # Several tests in FHE (VL) and non-FHE
    for in_fhe in [True, False]:
        for n_bits in range(1, 6):
            for power_n_values in range(8):
                tests_cases.append(
                    pytest.param(
                        2**power_n_values,
                        n_bits,
                        in_fhe,
                        True,
                        id=generate_sum_id(2**power_n_values, n_bits, in_fhe, True),
                    )
                )
    return tests_cases


@pytest.mark.parametrize(
    "n_values, n_bits, in_fhe, use_virtual_lib",
    generate_test_sum_cases(),
)
def test_sum(n_values, n_bits, in_fhe, use_virtual_lib, default_configuration, is_vl_only_option):
    """Tests ReduceSum ONNX operator on a torch model."""

    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    max_value = 2**n_bits

    # Create a Torch module that only sums the elements of an array
    torch_model = _TorchReduceSum()

    # Get the test values
    numpy_input, inputset = get_numpy_input_and_inputset(
        n_values=n_values, max_value=max_value, signed=True
    )

    computed_sum = execute_reduce_sum(
        torch_model=torch_model,
        numpy_input=numpy_input,
        inputset=inputset,
        n_bits=n_bits,
        in_fhe=in_fhe,
        use_virtual_lib=use_virtual_lib,
        configuration=default_configuration,
    )

    # Set the maximum error potentially created by the workaround. This max error is relevant only:
    # - if there is not quantization error
    # - if n_values is a power of 2
    # The idea is that for each depth d (from 1 to total_depth) of the "tree sum", we lose or earn
    # up to n_values//(2**depth) * 2**(depth-1), so a total of (n_value//2)*total_depth. More
    # information in the QuantizedReduceSum operator.
    total_depth = int(numpy.log2(n_values))
    max_error = (n_values // 2) * total_depth

    # Compute the expected sum
    expected_sum = numpy.sum(numpy_input)

    # Check if the error does not exceed the theoretical limit. An error term is added for
    # handling minor quantization artifacts.
    if n_values > 1:
        error = abs(expected_sum - computed_sum[0]) / max_error
        assert (
            error <= 1 + 10e-6
        ), f"Error reached {error*100:0.2f}% of the max possible error ({max_error})"

    # If only a single input value was considered, we expect no error from the sum.
    else:
        error = abs(expected_sum - computed_sum[0])
        assert error < 10e-6, f"Got an unexpected error of {error:0.2f} with a single input value."


def generate_wrong_parameters_and_id():
    """Generator for parameters used in test_reduce_sum_wrong_parameters"""
    wrong_parameters = [
        {"n_values": 13},
        {"n_samples": 3},
        {"keepdims": False},
        {"axes": 0},
        {"axes": [0, 1]},
        {"axes": None},
        {"one_dim": True},
    ]

    for wrong_parameter in wrong_parameters:
        wrong_parameter_name = list(wrong_parameter.keys())[0]
        wrong_parameter_value = list(wrong_parameter.values())[0]

        n_values = wrong_parameter.get("n_values", 2)
        n_samples = wrong_parameter.get("n_samples", 1)
        keepdims = wrong_parameter.get("keepdims", True)
        one_dim = wrong_parameter.get("one_dim", False)
        axes = 0 if one_dim else wrong_parameter.get("axes", 1)

        pytest_id = "reduce_sum_wrong_parameters"

        pytest_id += f"_{wrong_parameter_name}_{wrong_parameter_value}_in_FHE_(VL)"

        parameters = (n_values, n_samples, keepdims, axes, one_dim)
        yield parameters, pytest_id


@pytest.mark.parametrize(
    "n_values, n_samples, keepdims, axes, one_dim",
    [
        pytest.param(
            *parameters,
            id=pytest_id,
        )
        for parameters, pytest_id in generate_wrong_parameters_and_id()
    ],
)
def test_reduce_sum_wrong_parameters(
    n_values, n_samples, keepdims, axes, one_dim, default_configuration
):
    """Test all parameters forbidden for QuantizedReduceSum on a torch model."""
    n_bits = 5

    # Create a Torch module that only sums the elements of an array
    torch_model = _TorchReduceSum(dim=axes, keepdim=keepdims)

    # Get the test values
    numpy_input, inputset = get_numpy_input_and_inputset(
        n_values=n_values, max_value=2**n_bits, n_samples=n_samples
    )

    if one_dim:
        numpy_input = numpy_input.flatten()

    with pytest.raises(AssertionError):
        execute_reduce_sum(
            torch_model=torch_model,
            numpy_input=numpy_input,
            inputset=inputset,
            n_bits=n_bits,
            in_fhe=True,
            use_virtual_lib=True,
            configuration=default_configuration,
        )
