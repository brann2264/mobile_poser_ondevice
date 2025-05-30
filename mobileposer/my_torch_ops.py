# my_torch_ops.py
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

@register_torch_op
def isnan(context, node):
    torchTensor, = _get_inputs(context, node, expected=1)
    x = mb.const(val=(torchTensor != torchTensor), name=node.name)
    context.add(x, node.name)