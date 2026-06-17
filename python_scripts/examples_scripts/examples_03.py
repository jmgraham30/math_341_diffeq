#| label: setup
#| code-fold: true
#| code-summary: "Show the code"

# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| label: ex1-sympy

t = sym.Symbol('t')
y = sym.Function('y')

ode1 = sym.Eq(y(t).diff(t, 2) - y(t).diff(t) - 6*y(t), 0)

# General solution
gen1 = sym.dsolve(ode1, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen1)))

# Particular solution with y(0)=1, y'(0)=-1
part1 = sym.dsolve(ode1, y(t), ics={y(0): 1, y(t).diff(t).subs(t, 0): -1})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part1)))


#| label: fig-ex1
#| fig-cap: "Solutions of $y'' - y' - 6y = 0$ for several initial conditions. The particular solution satisfying $y(0)=1$, $y'(0)=-1$ is highlighted in red."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(-0.4, 1.2, 500)
fig, ax = plt.subplots(figsize=(7, 4))

# Several solution curves via different (C1, C2) pairs
for C1, C2 in [(-0.5, 1.5), (0, 1), (0.3, 0.3), (0.5, -0.5), (-0.2, 0.8)]:
    y_sol = C1*np.exp(3*t_vals) + C2*np.exp(-2*t_vals)
    ax.plot(t_vals, y_sol, color='steelblue', lw=1.2, alpha=0.55)

# Particular solution C1=1/5, C2=4/5
y_part = (1/5)*np.exp(3*t_vals) + (4/5)*np.exp(-2*t_vals)
ax.plot(t_vals, y_part, color='tomato', lw=2.2,
        label=r'$y(0)=1,\; y\prime(0)=-1$')
ax.plot(0, 1, 'ko', ms=6, zorder=5)

ax.set_ylim(-2, 5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' - y' - 6y = 0$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


#| label: ex2-sympy

ode2 = sym.Eq(y(t).diff(t, 2) + 6*y(t).diff(t) + 9*y(t), 0)

gen2 = sym.dsolve(ode2, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen2)))

part2 = sym.dsolve(ode2, y(t), ics={y(0): 2, y(t).diff(t).subs(t, 0): -3})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part2)))


#| label: fig-ex2
#| fig-cap: "Solutions of $y'' + 6y' + 9y = 0$ (repeated root $r=-3$). All solutions decay to zero. The particular solution satisfying $y(0)=2$, $y'(0)=-3$ is highlighted in red."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 3, 500)
fig, ax = plt.subplots(figsize=(7, 4))

for C1, C2 in [(1, 0), (0, 1), (1, -4), (3, 1), (-1, 2)]:
    y_sol = (C1 + C2*t_vals)*np.exp(-3*t_vals)
    ax.plot(t_vals, y_sol, color='steelblue', lw=1.2, alpha=0.55)

y_part = (2 + 3*t_vals)*np.exp(-3*t_vals)
ax.plot(t_vals, y_part, color='tomato', lw=2.2,
        label=r'$y(0)=2,\; y\prime(0)=-3$')
ax.plot(0, 2, 'ko', ms=6, zorder=5)
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' + 6y' + 9y = 0$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


#| label: ex3-sympy

ode3 = sym.Eq(y(t).diff(t, 2) - 4*y(t).diff(t) + 13*y(t), 0)

gen3 = sym.dsolve(ode3, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen3)))

part3 = sym.dsolve(ode3, y(t), ics={y(0): 0, y(t).diff(t).subs(t, 0): 3})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part3)))


#| label: fig-ex3
#| fig-cap: "Solutions of $y'' - 4y' + 13y = 0$ (complex roots $r = 2\\pm 3i$). The growing exponential envelope $\\pm e^{2t}$ is shown dashed. The particular solution $y = e^{2t}\\sin(3t)$ is highlighted in red."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 2.2, 600)
fig, ax = plt.subplots(figsize=(7, 4))

for C1, C2 in [(1, 0), (0.5, 0.5), (-0.5, 1), (1, -1)]:
    y_sol = np.exp(2*t_vals)*(C1*np.cos(3*t_vals) + C2*np.sin(3*t_vals))
    ax.plot(t_vals, y_sol, color='steelblue', lw=1.2, alpha=0.55)

# Particular solution C1=0, C2=1
y_part = np.exp(2*t_vals)*np.sin(3*t_vals)
ax.plot(t_vals, y_part, color='tomato', lw=2.2,
        label=r'$y = e^{2t}\sin(3t)$')

# Envelope
ax.plot(t_vals,  np.exp(2*t_vals), 'k--', lw=1, label=r'$\pm e^{2t}$ (envelope)')
ax.plot(t_vals, -np.exp(2*t_vals), 'k--', lw=1)
ax.plot(0, 0, 'ko', ms=6, zorder=5)

ax.set_ylim(-20, 20)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' - 4y' + 13y = 0$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


#| label: ex4-sympy

ode4 = sym.Eq(y(t).diff(t, 2) + 4*y(t).diff(t) + 4*y(t), 3*t**2 - 2)

gen4 = sym.dsolve(ode4, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen4)))

part4 = sym.dsolve(ode4, y(t), ics={y(0): 1, y(t).diff(t).subs(t, 0): 0})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part4)))


#| label: ex5-sympy

ode5 = sym.Eq(y(t).diff(t, 2) - 3*y(t).diff(t) - 10*y(t), 6*sym.exp(2*t))

gen5 = sym.dsolve(ode5, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen5)))


#| label: ex6-sympy

ode6 = sym.Eq(y(t).diff(t, 2) - 3*y(t).diff(t) - 10*y(t), 6*sym.exp(-2*t))

gen6 = sym.dsolve(ode6, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen6)))


#| label: ex7-sympy

ode7 = sym.Eq(y(t).diff(t, 2) + 9*y(t), 5*sym.cos(2*t))

gen7 = sym.dsolve(ode7, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen7)))

part7 = sym.dsolve(ode7, y(t), ics={y(0): 1, y(t).diff(t).subs(t, 0): 0})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part7)))


#| label: fig-ex7
#| fig-cap: "Solution of $y'' + 9y = 5\\cos(2t)$ with $y(0)=1$, $y'(0)=0$. The particular (forced) response $\\cos(2t)$ and the homogeneous (natural) response are both shown. Here $C_1 = C_2 = 0$ so the full solution equals the forced response alone."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 4*np.pi, 800)

y_full = np.cos(2*t_vals)              # C1 = C2 = 0
y_hom  = np.zeros_like(t_vals)        # homogeneous part vanishes for this IVP
y_part_plot = np.cos(2*t_vals)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals, y_full,      color='tomato',    lw=2.2,
        label=r'$y(t) = \cos(2t)$ (full solution)')
ax.plot(t_vals, y_part_plot, color='steelblue', lw=1.4, ls='--',
        label=r'$y_p = \cos(2t)$ (particular)')
ax.axhline(0, color='gray', lw=0.8)
ax.plot(0, 1, 'ko', ms=6, zorder=5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' + 9y = 5\cos(2t)$, $\;y(0)=1$, $\;y'(0)=0$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


#| label: ex8-sympy

ode8 = sym.Eq(y(t).diff(t, 2) + 9*y(t), 5*sym.cos(3*t))

gen8 = sym.dsolve(ode8, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen8)))


#| label: fig-ex8
#| fig-cap: "General solution of $y'' + 9y = 5\\cos(3t)$ with $C_1 = C_2 = 0$, showing the resonant response $y_p = \\tfrac{5}{6}t\\sin(3t)$. The amplitude grows linearly with time."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 6*np.pi, 1000)
y_res = (5/6)*t_vals*np.sin(3*t_vals)
envelope = (5/6)*t_vals

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals, y_res,      color='tomato',    lw=1.8,
        label=r'$y_p = \frac{5}{6}\,t\sin(3t)$')
ax.plot(t_vals,  envelope,  color='steelblue', lw=1, ls='--',
        label=r'$\pm\frac{5}{6}\,t$ (linear envelope)')
ax.plot(t_vals, -envelope,  color='steelblue', lw=1, ls='--')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"Resonance: $y'' + 9y = 5\cos(3t)$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


#| label: session-info

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}'
                for m in globals().values()
                if getattr(m, '__version__', None)))
