"""
The functions and data sets the demos optimise.

Each 1D function has f(x) and its derivative df/dx; each 2D surface has
f(p) and its gradient, where p = (x, y). Writing the derivative by hand is
exactly what PyTorch's autograd does for you automatically.
"""

import numpy as np

# ---------------------------------------------------------------------------
# One-variable functions (page: Rolling downhill)
# ---------------------------------------------------------------------------

FUNCTIONS_1D = {
    "Bowl: f(x) = x²": {
        "f": lambda x: x**2,
        "grad": lambda x: 2 * x,
        "latex": r"f(x) = x^2,\qquad \frac{df}{dx} = 2x",
        "x_range": (-3.0, 3.0),
        "start": 2.5,
        "note": "One minimum at x = 0. The update is x ← (1 − 2η)·x, so η = 0.5 "
                "lands on the minimum in one step, η = 1 bounces between ±x for ever, "
                "and anything above 1 runs away.",
    },
    "Two valleys: f(x) = x⁴ − 3x² + x": {
        "f": lambda x: x**4 - 3 * x**2 + x,
        "grad": lambda x: 4 * x**3 - 6 * x + 1,
        "latex": r"f(x) = x^4 - 3x^2 + x,\qquad \frac{df}{dx} = 4x^3 - 6x + 1",
        "x_range": (-2.2, 2.2),
        "start": 2.0,
        "note": "A deep valley on the left (x ≈ −1.30) and a shallower one on the right "
                "(x ≈ 1.13). Where you start decides where you finish.",
    },
    "Bumpy bowl: f(x) = 0.3x² + sin(3x)": {
        "f": lambda x: 0.3 * x**2 + np.sin(3 * x),
        "grad": lambda x: 0.6 * x + 3 * np.cos(3 * x),
        "latex": r"f(x) = 0.3x^2 + \sin 3x,\qquad \frac{df}{dx} = 0.6x + 3\cos 3x",
        "x_range": (-4.0, 4.0),
        "start": 3.9,
        "note": "Lots of little dips. Plain gradient descent stops in the first one it "
                "meets; momentum can carry the ball over the bumps.",
    },
}


# ---------------------------------------------------------------------------
# Two-variable surfaces (page: Optimiser race)
# ---------------------------------------------------------------------------

def _valley(p):
    x, y = p[..., 0], p[..., 1]
    return 0.05 * x**2 + 2 * y**2


def _valley_grad(p):
    x, y = p
    return np.array([0.1 * x, 4 * y])


def _rosenbrock(p):
    x, y = p[..., 0], p[..., 1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def _rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


def _himmelblau(p):
    x, y = p[..., 0], p[..., 1]
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def _himmelblau_grad(p):
    x, y = p
    a = x**2 + y - 11
    b = x + y**2 - 7
    return np.array([4 * x * a + 2 * b, 2 * a + 4 * y * b])


SURFACES_2D = {
    "Long narrow valley": {
        "f": _valley,
        "grad": _valley_grad,
        "latex": r"f(x, y) = 0.05x^2 + 2y^2",
        "x_range": (-10.0, 10.0),
        "y_range": (-3.0, 3.0),
        "start": (-9.0, 2.0),
        "minima": [(0.0, 0.0)],
        "log_colour": False,
        "learning_rates": {"Gradient descent": 0.45, "Momentum": 0.1, "Adam": 0.5},
        "steps": 60,
        "note": "Steep across, shallow along. Plain gradient descent zig-zags across the "
                "valley; momentum cancels the zig-zag; Adam rescales each direction.",
    },
    "Banana (Rosenbrock)": {
        "f": _rosenbrock,
        "grad": _rosenbrock_grad,
        "latex": r"f(x, y) = (1 - x)^2 + 100\,(y - x^2)^2",
        "x_range": (-2.0, 2.0),
        "y_range": (-1.0, 3.0),
        "start": (-1.5, 2.0),
        "minima": [(1.0, 1.0)],
        "log_colour": True,
        "learning_rates": {"Gradient descent": 0.0015, "Momentum": 0.001, "Adam": 0.1},
        "steps": 500,
        "note": "A classic test: finding the curved valley is easy, following it to the "
                "minimum at (1, 1) is hard.",
    },
    "Four valleys (Himmelblau)": {
        "f": _himmelblau,
        "grad": _himmelblau_grad,
        "latex": r"f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        "x_range": (-5.0, 5.0),
        "y_range": (-5.0, 5.0),
        "start": (-0.3, -0.9),
        "minima": [(3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)],
        "log_colour": True,
        "learning_rates": {"Gradient descent": 0.01, "Momentum": 0.005, "Adam": 0.1},
        "steps": 150,
        "note": "Four equally good minima. Starting on the central hill, different "
                "optimisers can roll into different valleys.",
    },
}


# ---------------------------------------------------------------------------
# Curves for the over/underfitting demo
# ---------------------------------------------------------------------------

def supernova_light_curve(t):
    """
    A smooth rise-and-fall shape (the Bazin et al. 2009 form used for supernova
    light curves): flux in arbitrary units against time in days.
    """
    t0, tau_rise, tau_fall = 25.0, 4.0, 20.0
    return np.exp(-(t - t0) / tau_fall) / (1 + np.exp(-(t - t0) / tau_rise))


CURVES = {
    "Supernova-like light curve": {
        "f": supernova_light_curve,
        "t_range": (0.0, 100.0),
        "x_label": "Days",
        "y_label": "Flux (arbitrary units)",
        "noise_default": 0.06,
    },
    "Sine wave": {
        "f": lambda t: np.sin(2 * np.pi * t),
        "t_range": (0.0, 1.0),
        "x_label": "x",
        "y_label": "y",
        "noise_default": 0.25,
    },
}
