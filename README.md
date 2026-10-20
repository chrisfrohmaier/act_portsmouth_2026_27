# ML Playground

Interactive demos for the ACT *Intro to Machine Learning* lecture
(University of Portsmouth, Institute of Cosmology and Gravitation).

| Page | What it shows |
|---|---|
| 1 · Rolling downhill | Gradient descent on a curve: learning rate too small / right / too big, local minima, momentum |
| 2 · Fit a line | The diabetes-vs-BMI regression from the lecture: loss landscape, the path gradient descent takes, feature scaling, mini-batch and stochastic gradient descent |
| 3 · Optimiser race | Gradient descent vs momentum vs Adam on a narrow valley, the Rosenbrock "banana" and Himmelblau's four minima |
| 4 · Under- and overfitting | Polynomials of rising degree on a supernova-like light curve; training vs test error; ridge regularisation |
| 5 · Decision boundaries | Logistic regression (with and without extra features), a neural network trained epoch by epoch, k-nearest neighbours, and k-means clustering on the two-moons data; accuracy, confusion matrix, precision and recall |

Every page has preset buttons for the lecture, a ▶ Play button that replays the
optimisation step by step, and a *Show the code* panel with the real source.

## Run it on your own computer

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens in your browser at http://localhost:8501. It needs Python 3.10 or newer.
Nothing is downloaded at run time: all data sets ship with scikit-learn or are generated.

## The code you can reuse

`mlcore/gd.py` is plain NumPy with no Streamlit in it - gradient descent written out
by hand, meant to be read and adapted:

- `mse`, `mse_gradients`, `fit_line_gd` - fitting `y = m*x + c` by gradient descent
  (batch, mini-batch or stochastic). This is the starting point for **Exercise 1** in
  `Scikit-learn_basics.ipynb`: swap in your own `x` and `y` arrays.
- `gradient_descent_1d` - minimising any function of one variable, with optional momentum.
- `gd_step`, `momentum_step`, `adam_step`, `run_optimiser` - three optimisers for
  functions of two parameters.

```python
import numpy as np
from mlcore.gd import fit_line_gd

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.1, 3.9, 6.2, 7.8])
history = fit_line_gd(x, y, learning_rate=0.05, n_steps=500)
print(history["m"][-1], history["c"][-1], history["loss"][-1])
```

## Layout

```
streamlit_app.py      entry point and page navigation
app_pages/            one file per page (the Streamlit layout and charts)
mlcore/gd.py          the maths: losses, gradients, optimisers
mlcore/problems.py    the functions and curves the demos use
mlcore/style.py       colours and Plotly helpers
mlcore/ui.py          small page-layout helpers
tests/test_app.py     smoke tests: every page and preset runs (pip install pytest; pytest -q)
.streamlit/config.toml  colours and server settings
```

## Put it online (Streamlit Community Cloud, free)

1. Push this folder to a GitHub repository (public is simplest).
2. Sign in at https://share.streamlit.io with GitHub and click **Create app**.
3. Choose **Yup, I have an app**, then pick the repository, the branch and the main
   file `streamlit_app.py`. Optionally choose a memorable subdomain under **App URL**.
4. Under **Advanced settings**, Python 3.12 (the default) is fine. Deploy.

Community Cloud puts an app to sleep after 12 hours without visitors. Open the link
10 minutes before a lecture; if it shows the sleeping page, click the button to wake
it and give it a minute or two to start.

## Credits

Built for the ACT module, extending the original gradient-descent notebook. The
supernova-like curve uses the rise-and-decline form of Bazin et al. (2009). The
diabetes data (Efron et al. 2004) and the two-moons, circles and blobs generators
come with scikit-learn.
