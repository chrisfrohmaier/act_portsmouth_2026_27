"""
Gradient descent, written out by hand.

Everything in this file is plain NumPy so you can read it, copy it and adapt it
for your own problems (for example Exercise 1 in Scikit-learn_basics.ipynb).
The Streamlit pages only *call* these functions and draw the results.

The one idea behind every function here:

    new parameter = old parameter - learning_rate * gradient of the loss

The gradient points uphill, so stepping against it moves you downhill.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Fitting a straight line  y_pred = m * x + c  with the mean squared error
# ---------------------------------------------------------------------------

def predict_line(m, c, x):
    """The model: a straight line with gradient m and intercept c."""
    return m * x + c


def mse(y_pred, y):
    """Mean squared error: the average of (prediction - truth)^2."""
    return np.mean((y_pred - y) ** 2)


def mse_gradients(m, c, x, y):
    """
    Partial derivatives of the MSE with respect to m and c.

        L(m, c) = (1/n) * sum( (m*x + c - y)^2 )

        dL/dm   = (2/n) * sum( (m*x + c - y) * x )
        dL/dc   = (2/n) * sum( (m*x + c - y) )

    Note: the error is NOT squared in the gradient - the chain rule turns
    (error)^2 into 2 * error * d(error)/d(parameter).
    """
    error = predict_line(m, c, x) - y
    dm = 2 * np.mean(error * x)
    dc = 2 * np.mean(error)
    return dm, dc


def fit_line_gd(x, y, m=0.0, c=0.0, learning_rate=0.01, n_steps=1000,
                batch_size=None, seed=0):
    """
    Fit y = m*x + c by gradient descent.

    Parameters
    ----------
    x, y : 1D NumPy arrays of the same length (feature and target)
    m, c : the starting guess
    learning_rate : the step size
    n_steps : how many updates to make
    batch_size : None  -> use all the data for every step ("batch" gradient descent)
                 1     -> one random point per step ("stochastic" gradient descent)
                 k     -> k random points per step ("mini-batch" gradient descent)
    seed : random seed, so mini-batch runs are reproducible

    Returns
    -------
    dict of NumPy arrays with one entry per step (index 0 is the starting guess):
        "m", "c" : the parameters after each step
        "loss"   : the MSE on the FULL data set at those parameters
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    use_batches = batch_size is not None and batch_size < n

    ms, cs = [m], [c]
    losses = [mse(predict_line(m, c, x), y)]
    order, position = rng.permutation(n), 0

    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            if use_batches:
                # Walk through a shuffled copy of the data; reshuffle each epoch.
                if position + batch_size > n:
                    order, position = rng.permutation(n), 0
                idx = order[position:position + batch_size]
                position += batch_size
                xb, yb = x[idx], y[idx]
            else:
                xb, yb = x, y

            dm, dc = mse_gradients(m, c, xb, yb)
            m = m - learning_rate * dm
            c = c - learning_rate * dc

            loss = mse(predict_line(m, c, x), y)
            ms.append(m)
            cs.append(c)
            losses.append(loss)
            if not np.isfinite(loss) or loss > 1e30:
                break  # the steps are making things worse: we have diverged

    return {"m": np.array(ms), "c": np.array(cs), "loss": np.array(losses)}


def mse_surface(x, y, m_grid, c_grid):
    """
    The MSE for every (m, c) pair on a grid - the 'loss landscape'.

    Expanding the square gives a formula that only needs a few averages,
    so the whole grid is computed at once instead of in a loop.
    """
    sx, sy = np.mean(x), np.mean(y)
    sxx, sxy, syy = np.mean(x * x), np.mean(x * y), np.mean(y * y)
    M, C = np.meshgrid(m_grid, c_grid)
    return (M**2 * sxx + 2 * M * C * sx + C**2
            - 2 * M * sxy - 2 * C * sy + syy)


# ---------------------------------------------------------------------------
# 2. Minimising a function of one variable
# ---------------------------------------------------------------------------

def gradient_descent_1d(grad, x0, learning_rate, n_steps, momentum=0.0):
    """
    Minimise f(x) given its derivative grad(x) = df/dx.

    With momentum = 0 this is plain gradient descent:
        x <- x - learning_rate * grad(x)

    With momentum > 0 we keep a running 'velocity' v, like a heavy ball:
        v <- momentum * v - learning_rate * grad(x)
        x <- x + v

    Returns the array of x positions, starting with x0.
    """
    x, v = float(x0), 0.0
    xs = [x]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            v = momentum * v - learning_rate * grad(x)
            x = x + v
            xs.append(x)
            if not np.isfinite(x) or abs(x) > 1e6:
                break
    return np.array(xs)


# ---------------------------------------------------------------------------
# 3. Three optimisers for a function of two parameters p = (p1, p2)
# ---------------------------------------------------------------------------

def gd_step(p, g, state, learning_rate, t):
    """Plain gradient descent: step straight downhill."""
    return p - learning_rate * g


def momentum_step(p, g, state, learning_rate, t, beta=0.9):
    """
    Momentum: accumulate a velocity, so consistent directions speed up and
    zig-zags (gradients that flip sign) cancel out.
    """
    state["v"] = beta * state.get("v", 0.0) - learning_rate * g
    return p + state["v"]


def adam_step(p, g, state, learning_rate, t, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam: momentum on the gradient (m) plus a running average of its size (s).
    Dividing by sqrt(s) gives every parameter its own step size, so steep and
    shallow directions both make progress.
    """
    state["m"] = beta1 * state.get("m", 0.0) + (1 - beta1) * g
    state["s"] = beta2 * state.get("s", 0.0) + (1 - beta2) * g**2
    m_hat = state["m"] / (1 - beta1**t)  # bias correction for the early steps
    s_hat = state["s"] / (1 - beta2**t)
    return p - learning_rate * m_hat / (np.sqrt(s_hat) + eps)


OPTIMISERS = {
    "Gradient descent": gd_step,
    "Momentum": momentum_step,
    "Adam": adam_step,
}


def run_optimiser(step_fn, grad, start, learning_rate, n_steps):
    """Apply an optimiser's step function n_steps times; return the path."""
    p = np.array(start, dtype=float)
    state = {}
    path = [p.copy()]
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, n_steps + 1):
            g = grad(p)
            p = step_fn(p, g, state, learning_rate, t)
            path.append(p.copy())
            if not np.all(np.isfinite(p)) or np.max(np.abs(p)) > 1e6:
                break
    return np.array(path)
