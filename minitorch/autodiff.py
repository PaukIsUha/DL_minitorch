from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_up = list(vals)
    vals_down = list(vals)
    
    vals_up[arg] += epsilon
    vals_down[arg] -= epsilon
    
    f_up = f(*vals_up)
    f_down = f(*vals_down)
    
    derivative = (f_up - f_down) / (2 * epsilon)
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    sorted_list = []
    visited = set()

    def visit(var):
        if id(var) not in visited:
            visited.add(id(var))
            if var.history and var.history.inputs:
                for input_var in var.history.inputs:
                    visit(input_var)
            sorted_list.append(var)

    visit(variable)
    return reversed(sorted_list)


def backpropagate(variable: Variable, deriv: Any) -> None:
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)
    else:
        variable.derivative = deriv

    for var in topological_sort(variable):
        if var.history is not None and var.history.last_fn is not None:
            d_output = var.derivative
            gradients = var.chain_rule(d_output)
            for input_var, grad in gradients:
                if input_var.is_leaf():
                    input_var.accumulate_derivative(grad)
                else:
                    input_var.derivative = grad



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
