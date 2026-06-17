#| code-fold: true
#| code-summary: "Show the code"

# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import scipy as sp                        # SciPy for scientific computing
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from scipy.integrate import solve_ivp     # ODE solver
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


t, A, x0 = sym.symbols('t A x_0', real=True)

x = sym.S(1) / (1 + A * sym.exp(-t))

from IPython.display import Math, display
display(Math(r'x(t) = ' + sym.latex(x)))


lhs = sym.diff(x, t)
rhs = x * (1 - x)

residual = sym.simplify(lhs - rhs)

print("LHS  (dx/dt)  =", sym.simplify(lhs))
print("RHS  (x(1-x)) =", sym.simplify(rhs))
print("LHS - RHS     =", residual)


ic_eq = sym.Eq(x.subs(t, 0), x0)
print("Initial condition equation:", ic_eq)

A_val = sym.solve(ic_eq, A)[0]
print("A =", A_val)


x_particular = sym.simplify(x.subs(A, A_val))
print("Particular solution: x(t) =", x_particular)

display(Math(r'x(t) = ' + sym.latex(x_particular)))


lhs_p = sym.diff(x_particular, t)
rhs_p = x_particular * (1 - x_particular)
residual_p = sym.simplify(lhs_p - rhs_p)
print("Residual for particular solution:", residual_p)


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-logistic-ivp
#| fig-cap: "Solutions of $dx/dt = x(1-x)$ for various initial conditions $x_0$. All solutions with $x_0 \\in (0,1)$ increase toward the stable equilibrium $x=1$ (upper dashed line), while solutions with $x_0 > 1$ decrease toward it. The unstable equilibrium $x=0$ (lower dashed line) repels nearby solutions."

t_vals = np.linspace(-3, 6, 500)

# Initial conditions: below 0, near 0, spread through (0,1), above 1
x0_list = [-0.3, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 1.5, 2.5]

cmap = plt.cm.RdYlBu
colors = cmap(np.linspace(0.05, 0.95, len(x0_list)))

fig, ax = plt.subplots(figsize=(8, 5))

for x0_val, color in zip(x0_list, colors):
    # Avoid division by zero when x0 = 0 or x0 = 1
    denom = x0_val + (1 - x0_val) * np.exp(-t_vals)
    # Mask near-zero denominators for display stability
    with np.errstate(divide='ignore', invalid='ignore'):
        x_sol = np.where(np.abs(denom) > 1e-10, x0_val / denom, np.nan)
    ax.plot(t_vals, x_sol, color=color, lw=2,
            label=f'$x_0 = {x0_val}$')

# Equilibria
ax.axhline(1, color='black', linestyle='--', lw=1.2, label='Stable eq. $x=1$')
ax.axhline(0, color='gray',  linestyle='--', lw=1.2, label='Unstable eq. $x=0$')

ax.set_ylim(-0.6, 3.0)
ax.set_xlabel('$t$', fontsize=13)
ax.set_ylabel('$x(t)$', fontsize=13)
ax.set_title(r'Solutions of $\dfrac{dx}{dt} = x(1-x)$ for various $x_0$', fontsize=13)
ax.legend(fontsize=8, ncol=2, loc='upper left')
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys # sys for system-specific parameters and functions
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
