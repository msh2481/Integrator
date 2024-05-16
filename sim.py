import re
import time

import jax.numpy as jnp
from beartype import beartype as typed
from beartype.typing import Any, Callable
from jax import Array, jit, make_jaxpr
from jaxtyping import Array, Float, Int


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
        elif any(line.startswith(key + ":") for key in ["T0", "DT", "METHOD"]):
            init_section = False
            diff_section = False
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
def generate_get_x0(init_values: dict[str, float]) -> Callable[[], Float[Array, "n"]]:
    @typed
    def get_x0() -> Float[Array, "n"]:
        return jnp.array(list(init_values.values()))

    return jit(get_x0)


@typed
def generate_get_dxdt(
    diff_eqs: dict[str, str], init_values: dict[str, float]
) -> Callable[[float, Float[Array, "n"]], Float[Array, "n"]]:
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

    @typed
    def get_dxdt(t: Float[Array, ""], x: Float[Array, "n"]) -> Float[Array, "n"]:
        dxdt = []
        for var in variables:
            derivative = 0.0
            if var in substituted_eqs:
                derivative = eval(
                    substituted_eqs[var], {"t": t, "x": x, **math_functions}
                )
            dxdt.append(derivative)
        return jnp.array(dxdt)

    return jit(get_dxdt)


@typed
def compile_program(
    program: str,
) -> tuple[
    dict[str, Any],
    Callable[[], Float[Array, "n"]],
    Callable[[float, Float[Array, "n"]], Float[Array, "n"]],
]:
    args, init_values, diff_eqs = parse_program(program)
    get_x0 = generate_get_x0(init_values)
    get_dxdt = generate_get_dxdt(diff_eqs, init_values)
    return args, get_x0, get_dxdt


@typed
def simulate(
    args: dict[str, Any],
    get_x0: Callable[[], Float[Array, "n"]],
    get_dxdt_core: Callable[[float, Float[Array, "n"]], Float[Array, "n"]],
) -> tuple[Float[Array, "periods"], Float[Array, "periods n"]]:
    t0 = args["T0"]
    t1 = args["T1"]
    assert isinstance(t0, float) and isinstance(t1, float)
    assert t0 < t1
    dt = args["DT"]
    assert isinstance(dt, float) and dt > 0
    method = args["METHOD"]
    assert method in ["Euler", "Heun"]

    h_list: list[Float[Array, "n"]] = [get_x0()]
    t_list: list[float] = [t0]

    # TODO: check that lags are compatible with dt

    @typed
    def get_dxdt() -> Float[Array, "n"]:
        t = t_list[-1]
        x = h_list[-1]
        return get_dxdt_core(t, x)

    if method == "Euler":
        for t in jnp.arange(t0, t1, dt):
            x = h_list[-1]
            dxdt = get_dxdt()
            t_list.append(t + dt)
            h_list.append(x + dxdt * dt)
    elif method == "Heun":
        for t in jnp.arange(t0, t1, dt):
            x = h_list[-1]
            dxdt0 = get_dxdt()
            t_list.append(t + dt)
            h_list.append(x + dxdt0 * dt)
            dxdt1 = get_dxdt()
            h_list[-1] = x + 0.5 * (dxdt0 + dxdt1) * dt
    return jnp.stack(t_list), jnp.stack(h_list)


with open("code.txt", "r", encoding="utf-8") as f:
    args, get_x0, get_dxdt = compile_program(f.read())

with open("jaxpr.txt", "w", encoding="utf-8") as f:
    print(make_jaxpr(get_dxdt)(0.0, get_x0()).pretty_print(use_color=False), file=f)

x0 = get_x0()
print("Initial values:", x0)

t = 0.0
x = x0
dxdt = get_dxdt(t, x)
print("dxdt at t=0:", dxdt)


@typed
def benchmark(N: int) -> float:
    t = 0.0
    x = x0
    start_time = time.time()
    for _ in range(N):
        d = get_dxdt(t, x)
        x = x + d
        t = t + 1
    finish_time = time.time()
    return (finish_time - start_time) / N


for N in [1, 10, 100, 1000, 10000]:
    print(f"{N} iterations: {benchmark(N)} sec/it")

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
