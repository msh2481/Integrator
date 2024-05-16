import re
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd
from beartype import beartype as typed
from beartype.typing import Any, Callable
from jax import Array, jit, make_jaxpr
from jaxtyping import Float, Int
from numpy import ndarray as ND


@typed
def parse_program(
    program: str,
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    program = program.replace("\\\n", " ")
    init_values = {}
    diff_eqs = {}

    init_section = False
    diff_section = False
    args: dict[str, Any] = {}

    for line in program.split("\n"):
        line = line.strip()
        if line == "INIT:":
            init_section = True
            diff_section = False
            continue
        elif line == "DIFF:":
            init_section = False
            diff_section = True
            continue
        elif any(line.startswith(key + ":") for key in ["T0", "T1", "DT", "METHOD"]):
            args[line.split(":")[0].strip()] = line.split(":")[1].strip()
        elif not line:
            continue

        if init_section:
            var, value = line.split(":")
            init_values[var.strip()] = float(value.strip())
        elif diff_section:
            var, eq = line.split(":")
            diff_eqs[var.strip()] = eq.strip()

    return args, init_values, diff_eqs


@typed
def generate_get_x0(init_values: dict[str, float]) -> Callable[[], Float[ND, "n"]]:
    @typed
    def get_x0() -> Float[ND, "n"]:
        return np.array(list(init_values.values()))

    return get_x0


@typed
def generate_get_dxdt(
    diff_eqs: dict[str, str], init_values: dict[str, float]
) -> Callable[[Float[Array, ""], Float[Array, "n"]], Float[Array, "n"]]:
    variables = list(init_values.keys())
    var_to_idx = {var: idx for idx, var in enumerate(variables)}

    def substitute_vars(eq):
        for var, idx in var_to_idx.items():
            eq = re.sub(r"\b" + re.escape(var) + r"\b", f"x[{idx}]", eq)
        return eq

    substituted_eqs = {var: substitute_vars(eq) for var, eq in diff_eqs.items()}
    print(substituted_eqs)

    math_functions = {
        "sin": jnp.sin,
        "cos": jnp.cos,
        "tan": jnp.tan,
        "arcsin": jnp.arcsin,
        "arccos": jnp.arccos,
        "arctan": jnp.arctan,
        "sinh": jnp.sinh,
        "cosh": jnp.cosh,
        "tanh": jnp.tanh,
        "exp": jnp.exp,
        "log": jnp.log,
        "log10": jnp.log10,
        "sqrt": jnp.sqrt,
        "abs": jnp.abs,
        "ceil": jnp.ceil,
        "floor": jnp.floor,
        "sign": jnp.sign,
        "round": jnp.round,
        "pi": jnp.pi,
    }

    def jax_dxdt(t: Float[Array, ""], x: Float[Array, "n"]) -> Float[Array, "n"]:
        dxdt = []
        for var in variables:
            derivative = 0.0
            if var in substituted_eqs:
                derivative = eval(
                    substituted_eqs[var], {"t": t, "x": x, **math_functions}
                )
            dxdt.append(derivative)
        return jnp.array(dxdt)

    return jit(jax_dxdt)


@typed
def compile_program(
    program: str,
) -> tuple[
    dict[str, Any],
    list[str],
    Callable[[], Float[ND, "n"]],
    Callable[[Float[Array, ""], Float[Array, "n"]], Float[Array, "n"]],
]:
    args, init_values, diff_eqs = parse_program(program)
    variables = list(init_values.keys())
    get_x0 = generate_get_x0(init_values)
    get_dxdt = generate_get_dxdt(diff_eqs, init_values)
    return args, variables, get_x0, get_dxdt


@typed
def simulate(
    args: dict[str, Any],
    get_x0: Callable[[], Float[ND, "n"]],
    get_dxdt_core: Callable[[Float[Array, ""], Float[Array, "n"]], Float[Array, "n"]],
) -> tuple[Float[ND, "periods"], Float[ND, "periods n"]]:
    t0 = float(args["T0"])
    t1 = float(args["T1"])
    assert t0 < t1
    dt = float(args["DT"])
    assert dt > 0
    method = args["METHOD"]
    assert method in ["Euler", "Heun"]

    periods = jnp.arange(t0, t1, dt)
    h = np.zeros((len(periods), len(get_x0())))
    h[0] = get_x0()

    # TODO: check that lags are compatible with dt

    def get_dxdt(i: int, t: Float[Array, ""]) -> Float[Array, "n"]:
        return get_dxdt_core(t, jnp.asarray(h[i]))

    x = h[0]
    if method == "Euler":
        for i, t in enumerate(periods[:-1]):
            dxdt = get_dxdt(i, t)
            x += dxdt * dt
            h[i + 1] = x
    elif method == "Heun":
        for i, t in enumerate(periods[:-1]):
            dxdt0 = get_dxdt(i, t)
            h[i + 1] = x + dxdt0 * dt
            dxdt1 = get_dxdt(i + 1, t + dt)
            x += 0.5 * (dxdt0 + dxdt1) * dt
            h[i + 1] = x
    return np.array(periods), h


with open("code.txt", "r", encoding="utf-8") as f:
    args, variables, get_x0, get_dxdt = compile_program(f.read())

with open("jaxpr.txt", "w", encoding="utf-8") as f:
    print(
        make_jaxpr(get_dxdt)(jnp.array(0.0), get_x0()).pretty_print(use_color=False),
        file=f,
    )


start_time = time.time()
t, h = simulate(args, get_x0, get_dxdt)
finish_time = time.time()
print(f"Elapsed time: {finish_time - start_time} s")

df = pd.DataFrame({"t": t})
for i, var in enumerate(variables):
    df[var] = h[:, i]
df.to_csv("result.csv", index=False)

"""
Now I need to introduce time-lagged variables. They should look like this in the code:

T_cur: (T_goal - T_cur) * 0.2 + (T_out[0.1] - T_cur) * 0.05

It can be simplified by creating a new variable:

INIT:
T_out[0.1]: <same as T_out>
...

Then during parsing I should check that the requested time delta is representable with current time step. And during 
the simulation I should put the values from T_out to T_out[0.1] with the desired lag, before passing x to get_dxdt.
For the first few periods there will be no reference value to take, so just take the initial one.

What to do if I want to apply backward Euler method or something like that? Well, I would first do a honest step with 
forward method, which will use all needed substitutions. Then I will take dxdt in that point, average with the initial one,
and take a new step. Should be easy.
"""
