"""  Various neural network models in haiku 
  Inspired by: 
  https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/jax_models.py 
"""

import haiku as hk
import jax
import jax.numpy as jnp
from haiku.initializers import Constant
import functools
from resnet import ResNet

he_normal = hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')
_DEFAULT_BN_CONFIG = {
    'decay_rate': 0.9,
    'eps': 1e-5,
    'create_scale': True,
    'create_offset': True
}

class FilterResponseNorm(hk.Module):
    def __init__(self, eps=1e-6, name='frn'):
        super().__init__(name=name)
        self.eps = eps

    def __call__(self, x, **unused_kwargs):
        del unused_kwargs
        par_shape = (1, 1, 1, x.shape[-1])  # [1,1,1,C]
        tau = hk.get_parameter('tau', par_shape, x.dtype, init=jnp.zeros)
        beta = hk.get_parameter('beta', par_shape, x.dtype, init=jnp.zeros)
        gamma = hk.get_parameter('gamma', par_shape, x.dtype, init=jnp.ones)
        nu2 = jnp.mean(jnp.square(x), axis=[1, 2], keepdims=True)
        x = x * jax.lax.rsqrt(nu2 + self.eps)
        y = gamma * x + beta
        z = jnp.maximum(y, tau)

        return z

def _resnet_layer(
        inputs, num_filters, normalization_layer, kernel_size=3, strides=1,
        activation=lambda x: x, use_bias=True, is_training=True
):
    x = inputs
    x = hk.Conv2D(
        num_filters, kernel_size, stride=strides, padding='same',
        w_init=he_normal, with_bias=use_bias)(x)
    x = normalization_layer()(x, is_training=is_training)
    x = activation(x)
    return x




def make_resnet_classification(num_classes): 
    def forward(x, is_training):
        net = ResNet(num_classes = num_classes)
        return net(x, is_training = is_training)

    return forward



def get_model(model_name, num_classes, **kwargs):
    _MODEL_FNS = {
        "resnet": make_resnet_classification,
    }

    if model_name not in _MODEL_FNS.keys():
        raise NameError('Available keys:', _MODEL_FNS.keys())

    net_fn = _MODEL_FNS[model_name](num_classes, **kwargs)
    net = hk.transform_with_state(net_fn)

    return net.apply, net.init
