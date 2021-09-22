from typing import Callable

import haiku as hk
from jax import jit


# TODO: make this easier to understand
def map_variable_name(params: hk.Params, fn: Callable) -> hk.Params:
  params = hk.data_structures.to_mutable_dict(params)
  for module in params:
    params[module] = {
      fn(var_name): array for var_name, array in params[module].items()
    }
  return hk.data_structures.to_immutable_dict(params)


def predicate_mean(module_name, name, value):
  return name == "w_mu" or name == "b_mu"


def predicate_var(module_name, name, value):
  return name == "w_logvar" or name == "b_logvar"


def predicate_batchnorm(module_name, name, value):
  return name not in {
    "w_mu",
    "b_mu",
    "w_logvar",
    "b_logvar",
  }


@jit
def partition_params(params):
  params_log_var, params_rest = hk.data_structures.partition(predicate_var, params)

  def predicate_is_mu_with_log_var(module_name, name, value):
    logvar_name = f"{name.split('_')[0]}_logvar"
    return (
      predicate_mean(module_name, name, value)
      and module_name in params_log_var
      and logvar_name in params_log_var[module_name]
    )

  params_mean, params_deterministic = hk.data_structures.partition(
    predicate_is_mu_with_log_var, params_rest
  )
  return params_mean, params_log_var, params_deterministic
