"""QuantizedModule API."""
import copy
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy
from concrete.numpy.compilation.artifacts import DebugArtifacts
from concrete.numpy.compilation.circuit import Circuit
from concrete.numpy.compilation.compiler import Compiler
from concrete.numpy.compilation.configuration import Configuration

from ..common.debugging import assert_true
from ..common.utils import generate_proxy_function
from .base_quantized_op import QuantizedOp
from .quantized_array import QuantizedArray


class QuantizedModule:
    """Inference for a quantized model."""

    ordered_module_input_names: Tuple[str, ...]
    ordered_module_output_names: Tuple[str, ...]
    quant_layers_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]
    q_inputs: List[QuantizedArray]
    forward_fhe: Union[None, Circuit]

    def __init__(
        self,
        ordered_module_input_names: Iterable[str],
        ordered_module_output_names: Iterable[str],
        quant_layers_dict: dict,
    ):
        self.ordered_module_input_names = tuple(ordered_module_input_names)
        self.ordered_module_output_names = tuple(ordered_module_output_names)

        assert_true(
            (num_outputs := len(self.ordered_module_output_names)) == 1,
            f"{QuantizedModule.__class__.__name__} only supports a single output for now, "
            f"got {num_outputs}",
        )

        self.quant_layers_dict = copy.deepcopy(quant_layers_dict)
        self._is_compiled = False
        self.forward_fhe = None
        self.q_inputs = []

    @property
    def is_compiled(self) -> bool:
        """Return the compiled status of the module.

        Returns:
            bool: the compiled status of the module.
        """
        return self._is_compiled

    def __call__(self, *x: numpy.ndarray):
        return self.forward(*x)

    def forward(self, *qvalues: numpy.ndarray) -> numpy.ndarray:
        """Forward pass with numpy function only.

        Args:
            *qvalues (numpy.ndarray): numpy.array containing the quantized values.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        # Make sure that the input is quantized
        assert_true(
            len(
                invalid_inputs := tuple(
                    (idx, qvalue)
                    for idx, qvalue in enumerate(qvalues)
                    if not issubclass(qvalue.dtype.type, numpy.integer)
                )
            )
            == 0,
            f"Inputs: {', '.join(f'#{val[0]} ({val[1].dtype})' for val in invalid_inputs)} are not "
            "integer types. Make sure you quantize your input before calling forward.",
            ValueError,
        )

        return self._forward(*qvalues)

    def _forward(self, *qvalues: numpy.ndarray) -> numpy.ndarray:
        """Forward function for the FHE circuit.

        Args:
            *qvalues (numpy.ndarray): numpy.array containing the quantized values.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """

        assert_true(
            (n_qvalues := len(qvalues)) == (n_qinputs := len(self.q_inputs)),
            f"Got {n_qvalues} inputs, expected {n_qinputs}",
            TypeError,
        )
        for idx, qvalue in enumerate(qvalues):
            self.q_inputs[idx].update_quantized_values(qvalue)

        # Init layer_results with the inputs
        layer_results = dict(zip(self.ordered_module_input_names, self.q_inputs))

        for output_name, (input_names, layer) in self.quant_layers_dict.items():
            inputs = (layer_results[input_name] for input_name in input_names)
            output = layer(*inputs)
            layer_results[output_name] = output

        outputs = tuple(
            layer_results[output_name] for output_name in self.ordered_module_output_names
        )

        assert_true(len(outputs) == 1)

        return outputs[0].qvalues

    def forward_and_dequant(self, *q_x: numpy.ndarray) -> numpy.ndarray:
        """Forward pass with numpy function only plus dequantization.

        Args:
            *q_x (numpy.ndarray): numpy.ndarray containing the quantized input values. Requires the
                input dtype to be uint8.

        Returns:
            (numpy.ndarray): Predictions of the quantized model
        """
        q_out = self.forward(*q_x)
        return self.dequantize_output(q_out)

    def quantize_input(
        self, *values: numpy.ndarray
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Take the inputs in fp32 and quantize it using the learned quantization parameters.

        Args:
            *values (numpy.ndarray): Floating point values.

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: Quantized (numpy.uint32) values.
        """

        assert_true(
            (n_values := len(values)) == (n_qinputs := len(self.q_inputs)),
            f"Got {n_values} inputs, expected {n_qinputs}",
            TypeError,
        )
        for idx, qvalue in enumerate(values):
            self.q_inputs[idx].update_values(qvalue)

        qvalues = tuple(q_input.qvalues for q_input in self.q_inputs)
        qvalues_as_uint32 = tuple(qvalue.astype(numpy.uint32) for qvalue in qvalues)
        assert_true(
            all(
                numpy.array_equal(qval_uint32, qval)
                for qval_uint32, qval in zip(qvalues_as_uint32, qvalues)
            ),
            on_error_msg="Input quantizer does not give values within uint32.",
        )
        return qvalues_as_uint32[0] if len(qvalues_as_uint32) == 1 else qvalues_as_uint32

    def dequantize_output(self, qvalues: numpy.ndarray) -> numpy.ndarray:
        """Take the last layer q_out and use its dequant function.

        Args:
            qvalues (numpy.ndarray): Quantized values of the last layer.

        Returns:
            numpy.ndarray: Dequantized values of the last layer.
        """
        output_layers = (
            self.quant_layers_dict[output_name][1]
            for output_name in self.ordered_module_output_names
        )
        real_values = tuple(
            QuantizedArray(
                output_layer.n_bits,
                qvalues,
                value_is_float=False,
                scale=output_layer.output_scale,
                zero_point=output_layer.output_zero_point,
            ).dequant()
            for output_layer in output_layers
        )

        assert_true(len(real_values) == 1)

        return real_values[0]

    def set_inputs_quantization_parameters(self, *input_q_params: QuantizedArray):
        """Set the quantization parameters for the module's inputs.

        Args:
            *input_q_params (QuantizedArray): The quantization parameters for the module in the form
                of QuantizedArrays.
        """
        assert_true(
            (n_values := len(input_q_params)) == (n_inputs := len(self.ordered_module_input_names)),
            f"Got {n_values} inputs, expected {n_inputs}",
            TypeError,
        )

        self.q_inputs.clear()
        self.q_inputs.extend(copy.deepcopy(q_params) for q_params in input_q_params)

    def compile(
        self,
        q_inputs: Union[Tuple[QuantizedArray, ...], QuantizedArray],
        configuration: Optional[Configuration] = None,
        compilation_artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        use_virtual_lib: bool = False,
    ) -> Circuit:
        """Compile the forward function of the module.

        Args:
            q_inputs (Union[Tuple[QuantizedArray, ...], QuantizedArray]): Needed for tracing and
                building the boundaries.
            configuration (Optional[Configuration]): Configuration object
                                                                            to use during
                                                                            compilation
            compilation_artifacts (Optional[DebugArtifacts]): Artifacts object to fill during
            show_mlir (bool): if set, the MLIR produced by the converter and which is
                going to be sent to the compiler backend is shown on the screen, e.g., for debugging
                or demo. Defaults to False.
            use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
                Defaults to False.

        Returns:
            Circuit: the compiled Circuit.
        """

        if not isinstance(q_inputs, tuple):
            q_inputs = (q_inputs,)
        else:
            ref_len = q_inputs[0].values.shape[0]
            assert_true(
                all(q_input.values.shape[0] == ref_len for q_input in q_inputs),
                "Mismatched dataset lengths",
            )

        # concrete-numpy does not support variable *args-syle functions, so compile a proxy function
        # dynamically with a suitable number of arguments
        forward_proxy, orig_args_to_proxy_func_args = generate_proxy_function(
            self._forward, self.ordered_module_input_names
        )

        compiler = Compiler(
            forward_proxy,
            {arg_name: "encrypted" for arg_name in orig_args_to_proxy_func_args.values()},
        )

        def get_inputset_iterable():
            if len(self.q_inputs) > 1:
                return (
                    tuple(numpy.expand_dims(q_input.qvalues[idx], 0) for q_input in q_inputs)
                    for idx in range(ref_len)
                )
            return (numpy.expand_dims(arr, 0) for arr in q_inputs[0].qvalues)

        inputset = get_inputset_iterable()

        self.forward_fhe = compiler.compile(
            inputset,
            configuration,
            compilation_artifacts,
            show_mlir=show_mlir,
            virtual=use_virtual_lib,
        )

        self._is_compiled = True

        return self.forward_fhe
