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
    "max": jnp.maximum,
    "min": jnp.minimum,
    "clip": jnp.clip,
    "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-x)),
}


@typed
def parse_program(
    program: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    lines = []
    for line in program.split("\n"):
        if "#" in line:
            line = line[: line.index("#")]
        lines.append(line.strip())
    program = "\n".join(lines).replace("\\\n", " ")
    init_values = {}
    diff_eqs = {}

    section = None
    args: dict[str, Any] = {}
    defines: dict[str, Any] = {}

    for line in program.split("\n"):
        line = line.strip()
        if any(line.startswith(key + ":") for key in ["DEFINE", "INIT", "DIFF"]):
            section = line.split(":")[0].strip()
            continue
        if any(line.startswith(key + ":") for key in ["T0", "T1", "DT", "METHOD"]):
            args[line.split(":")[0].strip()] = line.split(":")[1].strip()
            continue
        if not line:
            continue

        if section == "INIT":
            var, value = line.split(":")
            var = var.strip()
            for src, target in defines.items():
                var = re.sub("\\b" + re.escape(src) + "\\b", str(target), var)
            init_values[var] = value.strip()
            if var not in diff_eqs:
                diff_eqs[var] = "0.0"
        elif section == "DEFINE":
            var, value = line.split(":")
            var = var.strip()
            value = value.strip()
            assert var not in defines
            try:
                # if already a number, put as is
                _ = float(value)
                defines[var] = value
            except ValueError:
                defines[var] = eval(value, {**math_functions, **defines, **args})
        elif section == "DIFF":
            var, eq = line.split(":")
            diff_eqs[var.strip()] = eq.strip()

    return args, defines, init_values, diff_eqs


@typed
def generate_get_x0(
    init_values: dict[str, str],
    defines: dict[str, Any],
    args: dict[str, Any],
) -> Callable[[], Float[ND, "n"]]:
    @typed
    def get_x0() -> Float[ND, "n"]:
        return np.array(
            [
                eval(e, {**math_functions, **defines, **args})
                for e in init_values.values()
            ]
        )

    return get_x0


@typed
def generate_get_dxdt(
    diff_eqs: dict[str, str],
    variables: list[str],
    defines: dict[str, Any],
    args: dict[str, Any],
) -> Callable[[Float[Array, ""], Float[Array, "n"]], Float[Array, "n"]]:
    var_to_idx = {var: idx for idx, var in enumerate(variables)}

    def substitute_vars(eq):
        for name, value in defines.items():
            eq = re.sub("\\b" + re.escape(name) + "\\b", str(value), eq)
        for var, idx in var_to_idx.items():
            if "[" in var:
                eq = re.sub(re.escape(var), f"x[{idx}]", eq)
        for var, idx in var_to_idx.items():
            if "[" not in var:
                eq = re.sub("\\b" + re.escape(var) + "\\b", f"x[{idx}]", eq)
        return eq

    substituted_eqs = {var: substitute_vars(eq) for var, eq in diff_eqs.items()}

    print(substituted_eqs)
    assert "t" not in defines and "x" not in defines

    def jax_dxdt(t: Float[Array, ""], x: Float[Array, "n"]) -> Float[Array, "n"]:
        dxdt = []
        for var in variables:
            derivative = 0.0
            if var in substituted_eqs:
                derivative = eval(
                    substituted_eqs[var],
                    {"t": t, "x": x, **args, **math_functions},
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
    args, defines, init_values, diff_eqs = parse_program(program)
    print(args)
    print(defines)
    print(init_values)
    assert set(init_values.keys()) == set(diff_eqs.keys())
    variables = list(init_values.keys())
    get_x0 = generate_get_x0(init_values, defines, args)
    get_dxdt = generate_get_dxdt(diff_eqs, variables, defines, args)
    return args, variables, get_x0, get_dxdt


@typed
def simulate(
    args: dict[str, Any],
    variables: list[str],
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

    lag_vars: list[tuple[int, int, int]] = []
    for name in variables:
        if "[" not in name:
            continue
        base = name.split("[")[0]
        lag = float(name.split("[")[1].split("]")[0])
        lag_periods = int(np.ceil(lag / dt))
        assert (
            abs(lag - lag_periods * dt) < 1e-6
        ), "lag must be an integer multiple of dt"
        lag_vars.append((variables.index(base), variables.index(name), lag_periods))

    def get_dxdt(i: int, t: Float[Array, ""]) -> Float[Array, "n"]:
        x = h[i]
        # this assignment has side effects, it updates values in h
        for src, dst, d in lag_vars:
            x[dst] = h[max(i - d, 0), src]
        return get_dxdt_core(t, jnp.asarray(x))

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
    _ = get_dxdt(len(periods) - 1, periods[-1])  # to correctly set lagged variables
    return np.array(periods), h


with open("code.txt", "r", encoding="utf-8") as f:
    args, variables, get_x0, get_dxdt = compile_program(f.read())

with open("jaxpr.txt", "w", encoding="utf-8") as f:
    print(
        make_jaxpr(get_dxdt)(jnp.array(0.0), get_x0()).pretty_print(use_color=False),
        file=f,
    )


start_time = time.time()
t, h = simulate(args, variables, get_x0, get_dxdt)
finish_time = time.time()
print(f"Elapsed time: {finish_time - start_time} s")

df = pd.DataFrame({"t": t})
for i, var in enumerate(variables):
    df[var] = h[:, i]
df.to_csv("result.csv", index=False)
