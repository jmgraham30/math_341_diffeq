# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

x_vals = np.linspace(-3, 3, 20)
y_vals = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Right-hand side: f(x,y) = y - x
dY = Y - X
dX = np.ones_like(dY)

# Normalise for uniform arrow length
length = np.sqrt(dX**2 + dY**2)
dX_n = dX / length
dY_n = dY / length

fig, ax = plt.subplots(figsize=(6, 6))
ax.quiver(X, Y, dX_n, dY_n,
          angles='xy', scale=22, width=0.003,
          color='steelblue', alpha=0.75)

# Equilibrium line y = x + 1
x_line = np.linspace(-3, 3, 200)
ax.plot(x_line, x_line + 1, 'k--', lw=1.5, label=r'$y = x + 1$ (equil.)')

# A few representative solution curves  y = C e^x + x + 1
for C in [-3, -1, -0.5, 0.5, 1, 3]:
    y_sol = C * np.exp(x_line) + x_line + 1
    mask = np.abs(y_sol) < 4
    ax.plot(x_line[mask], y_sol[mask], color='tomato', lw=1.2, alpha=0.8)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"Slope field for $y' = y - x$", fontsize=13)
ax.legend(fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

x = sym.Symbol('x')
y = sym.Function('y')   # y must be a Function, not a Symbol, for dsolve

# Define the ODE   dy/dx = x^2 / (1 - y^2)
ode = sym.Eq(y(x).diff(x), x**2 / (1 - y(x)**2))

# SymPy's dsolve handles separable equations
sol = sym.dsolve(ode, y(x))
display(Math(r'\text{General solution: }' + sym.latex(sol)))

# Apply the initial condition y(0) = 0
sol_ivp = sym.dsolve(ode, y(x), ics={y(0): 0})
display(Math(r'\text{Particular solution: }' + sym.latex(sol_ivp)))

x = sym.Symbol('x')
y = sym.Function('y')

ode3 = sym.Eq(y(x).diff(x) - y(x), x)

# General solution
gen_sol = sym.dsolve(ode3, y(x))
display(Math(r'\text{General solution: } ' + sym.latex(gen_sol)))

# Particular solution with y(0) = 2
part_sol = sym.dsolve(ode3, y(x), ics={y(0): 2})
display(Math(r'\text{Particular solution: } ' + sym.latex(part_sol)))

x_vals = np.linspace(-2, 2, 400)
fig, ax = plt.subplots(figsize=(7, 5))

for C in [-2, -1, 0, 1, 2]:
    y_sol = C * np.exp(x_vals) - x_vals - 1
    lw = 1.2
    color = 'steelblue'
    ax.plot(x_vals, y_sol, color=color, lw=lw, alpha=0.6)

# Highlight the particular solution C = 3
y_part = 3 * np.exp(x_vals) - x_vals - 1
ax.plot(x_vals, y_part, color='tomato', lw=2.2,
        label=r'$y = 3e^x - x - 1$  ($C=3$, $y(0)=2$)')
ax.plot(0, 2, 'ko', ms=6, zorder=5)

ax.set_xlim(-2, 2)
ax.set_ylim(-5, 12)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"Solutions of $y' - y = x$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

t, k = sym.symbols('t k', positive=True)
T    = sym.Function('T')
Ta   = sym.Integer(70)

ode4 = sym.Eq(T(t).diff(t), -k * (T(t) - Ta))

# General solution
gen4 = sym.dsolve(ode4, T(t), ics={T(0): 200})
display(Math(r'T(t) = ' + sym.latex(gen4.rhs)))

# Substitute T(10) = 150 to find k
k_val = sym.solve(gen4.rhs.subs(t, 10) - 150, k)[0]
display(Math(r'k = ' + sym.latex(k_val) + r'\approx ' +
             str(round(float(k_val), 5))))

# Full particular solution
T_sol = gen4.rhs.subs(k, k_val)
display(Math(r'T(t) = ' + sym.latex(sym.simplify(T_sol))))

# Time to reach 100°F
t_star = sym.solve(T_sol - 100, t)[0]
display(Math(r't^* = ' + sym.latex(t_star) +
             r'\approx ' + str(round(float(t_star), 2)) + r'\text{ min}'))

k_num  = float(k_val)
t_star_num = float(t_star)
t_plot = np.linspace(0, 80, 500)
T_plot = 70 + 130 * np.exp(-k_num * t_plot)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_plot, T_plot, color='tomato', lw=2, label=r'$T(t) = 70 + 130\,(8/13)^{t/10}$')
ax.axhline(70,  color='gray',  ls='--', lw=1, label=r'$T_a = 70^\circ$F')
ax.axhline(100, color='navy',  ls='--', lw=1, label=r'$100^\circ$F')
ax.axvline(t_star_num, color='navy', ls=':', lw=1)
ax.plot(t_star_num, 100, 'bo', ms=6, zorder=5)
ax.annotate(f'$t \\approx {t_star_num:.1f}$ min',
            xy=(t_star_num, 100), xytext=(t_star_num + 3, 108),
            fontsize=9, color='navy')
ax.set_xlabel(r'$t$ (min)', fontsize=12)
ax.set_ylabel(r'$T$ (°F)', fontsize=12)
ax.set_title("Newton's Law of Cooling", fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

t  = sym.Symbol('t', nonnegative=True)
P  = sym.Function('P')
r, H_val = sym.Rational(2, 5), sym.Integer(500)  # r = 0.4, H = 500

ode5 = sym.Eq(P(t).diff(t), r * P(t) - H_val)

sol5 = sym.dsolve(ode5, P(t), ics={P(0): 2000})
display(Math(r'P(t) = ' + sym.latex(sol5.rhs)))

t_plot = np.linspace(0, 6, 300)

fig, ax = plt.subplots(figsize=(7, 4))

for P0, color, lw, zord in [(500, 'steelblue', 1.4, 2),
                              (800, 'steelblue', 1.4, 2),
                              (1000,'steelblue', 1.4, 2),
                              (1250,'black',     1.4, 3),
                              (1500,'tomato',    1.4, 2),
                              (2000,'tomato',    2.2, 4)]:
    C_num = P0 - 1250
    P_sol = 1250 + C_num * np.exp(0.4 * t_plot)
    # Clip at 0 for display
    P_disp = np.maximum(P_sol, 0)
    label = f'$P(0)={P0}$' if P0 == 2000 else None
    ax.plot(t_plot, P_disp, color=color, lw=lw, zorder=zord, label=label)

ax.axhline(1250, color='black', ls='--', lw=1, label=r'$P^* = 1250$ (equil.)')
ax.set_xlabel(r'$t$ (years)', fontsize=12)
ax.set_ylabel(r'$P(t)$ (fish)', fontsize=12)
ax.set_title('Population growth with harvesting ($r=0.4$, $H=500$)', fontsize=12)
ax.set_ylim(-50, 5000)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

t = sym.Symbol('t', nonnegative=True)
I = sym.Function('I')
R_val, L_val = sym.Integer(4), sym.Rational(1, 10)

ode6 = sym.Eq(L_val * I(t).diff(t) + R_val * I(t),
              12 * sym.sin(10 * t))

sol6 = sym.dsolve(ode6, I(t), ics={I(0): 0})
I_expr = sym.simplify(sol6.rhs)
display(Math(r'I(t) = ' + sym.latex(I_expr)))

I_func = sym.lambdify(t, I_expr, 'numpy')
t_plot = np.linspace(0, 1.0, 1000)
I_plot = I_func(t_plot)

# Steady-state only
I_ss = (240/85) * np.sin(10 * t_plot) - (60/85) * np.cos(10 * t_plot)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_plot, I_plot, color='tomato', lw=2,   label=r'$I(t)$ (full solution)')
ax.plot(t_plot, I_ss,  color='steelblue', lw=1.5, ls='--',
        label=r'$I_{\mathrm{ss}}(t)$ (steady-state)')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$ (s)', fontsize=12)
ax.set_ylabel(r'$I$ (A)', fontsize=12)
ax.set_title('Current in a series RL circuit with sinusoidal forcing', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()