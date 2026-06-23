# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

t, s = sym.symbols('t s', real=True)
x = sym.Function('x')

ode1 = sym.Eq(x(t).diff(t, 2) + 4*x(t).diff(t) + 4*x(t), 8*sym.exp(t))

# Particular solution
part1 = sym.dsolve(ode1, x(t), ics={x(0): 1, x(t).diff(t).subs(t, 0): 0})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part1)))

# Verify via Laplace transform
X = sym.Function('X')
Xs = sym.Symbol('X_s')

# Compute the Laplace transform of the solution and confirm
sol_expr = part1.rhs
lap = sym.laplace_transform(sol_expr, t, s, noconds=True)
display(Math(r'X(s) = ' + sym.latex(sym.simplify(lap))))

t_vals = np.linspace(0, 2.5, 500)
x_sol = (8/9)*np.exp(t_vals) + (1/9)*np.exp(-2*t_vals) - (2/3)*t_vals*np.exp(-2*t_vals)
x_hom = (1/9)*np.exp(-2*t_vals) - (2/3)*t_vals*np.exp(-2*t_vals)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_vals, x_sol, color='tomato',    lw=2.2, label=r'$x(t)$ (full solution)')
ax.plot(t_vals, x_hom, color='steelblue', lw=1.4, ls='--',
        label=r'Homogeneous part $\frac{1}{9}e^{-2t}-\frac{2}{3}te^{-2t}$')
ax.plot(0, 1, 'ko', ms=6, zorder=5)
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x'' + 4x' + 4x = 8e^t$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# Define Heaviside forcing
f2 = sym.Heaviside(t - 1) - sym.Heaviside(t - 4)
ode2 = sym.Eq(x(t).diff(t, 2) + 9*x(t), f2)

part2 = sym.dsolve(ode2, x(t), ics={x(0): 0, x(t).diff(t).subs(t, 0): 0})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part2)))

t_vals = np.linspace(0, 10, 1000)

def g(tau):
    return (1/9)*(1 - np.cos(3*tau))

x_sol = (g(t_vals - 1)*(t_vals >= 1).astype(float)
         - g(t_vals - 4)*(t_vals >= 4).astype(float))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals, x_sol, color='tomato', lw=2)
ax.axvspan(1, 4, alpha=0.12, color='steelblue', label='Forcing active')
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(1, color='steelblue', lw=1, ls='--')
ax.axvline(4, color='steelblue', lw=1, ls='--')
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x'' + 9x = H(t-1) - H(t-4)$, $\;x(0)=x'(0)=0$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

ode3 = sym.Eq(x(t).diff(t, 2) - 2*x(t).diff(t) + 5*x(t), 0)

part3 = sym.dsolve(ode3, x(t), ics={x(0): 0, x(t).diff(t).subs(t, 0): 4})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part3)))

t_vals = np.linspace(0, 3, 600)
x_sol  = 2*np.exp(t_vals)*np.sin(2*t_vals)
env    = 2*np.exp(t_vals)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_vals,  env, color='steelblue', lw=1, ls='--', label=r'$\pm 2e^t$ (envelope)')
ax.plot(t_vals, -env, color='steelblue', lw=1, ls='--')
ax.plot(t_vals, x_sol, color='tomato', lw=2, label=r'$x(t)=2e^t\sin 2t$')
ax.axhline(0, color='gray', lw=0.8)
ax.plot(0, 0, 'ko', ms=6, zorder=5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x'' - 2x' + 5x = 0$, $\;x(0)=0$, $\;x'(0)=4$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# Verify via inverse Laplace transform
expr4 = s / (s**2 + 4)**2
inv4  = sym.inverse_laplace_transform(expr4, s, t)
display(Math(r'\mathcal{L}^{-1}\!\left[\frac{s}{(s^2+4)^2}\right] = '
            + sym.latex(sym.simplify(inv4))))

# Verify via convolution
f4 = sym.cos(2*t)
g4 = sym.sin(2*t) / 2
conv4 = sym.integrate(f4.subs(t, sym.Symbol('tau'))
                      * g4.subs(t, t - sym.Symbol('tau')),
                      (sym.Symbol('tau'), 0, t))
display(Math(r'(f * g)(t) = ' + sym.latex(sym.simplify(conv4))))

t_vals = np.linspace(0, 5*np.pi, 800)
x_sol  = (t_vals/4)*np.sin(2*t_vals)
env    = t_vals/4

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals,  env, color='steelblue', lw=1, ls='--', label=r'$\pm t/4$ (envelope)')
ax.plot(t_vals, -env, color='steelblue', lw=1, ls='--')
ax.plot(t_vals, x_sol, color='tomato', lw=2,
        label=r'$\frac{t}{4}\sin 2t$')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'', fontsize=13)
ax.set_title(r"$\mathcal{L}^{-1}\!\left[s/(s^2+4)^2\right] = (t/4)\sin 2t$",
             fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# General case: impulse response
H_expr = 1 / (s**2 + 6*s + 10)
h_t = sym.inverse_laplace_transform(H_expr, s, t)
display(Math(r'h(t) = \mathcal{L}^{-1}[H(s)] = ' + sym.latex(sym.simplify(h_t))))

# Specific case f(t) = e^{-3t}
f5 = sym.exp(-3*t)
ode5 = sym.Eq(x(t).diff(t, 2) + 6*x(t).diff(t) + 10*x(t), f5)
part5 = sym.dsolve(ode5, x(t), ics={x(0): 0, x(t).diff(t).subs(t, 0): 0})
display(Math(r'\text{Solution for }f(t)=e^{-3t}:\quad' + sym.latex(part5)))

# Verify via convolution integral
tau = sym.Symbol('tau', positive=True)
conv5 = sym.integrate(
    sym.exp(-3*tau)*sym.sin(tau)*sym.exp(-3*(t - tau)),
    (tau, 0, t)
)
display(Math(r'\text{Via convolution: }x(t) = ' + sym.latex(sym.simplify(conv5))))

t_vals = np.linspace(0, 5, 600)
x_sol  = np.exp(-3*t_vals)*(1 - np.cos(t_vals))
h_vals = np.exp(-3*t_vals)*np.sin(t_vals)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_vals, h_vals, color='steelblue', lw=1.4, ls='--',
        label=r'Impulse response $h(t) = e^{-3t}\sin t$')
ax.plot(t_vals, x_sol, color='tomato', lw=2,
        label=r'$x(t) = e^{-3t}(1-\cos t)$')
ax.axhline(0, color='gray', lw=0.8)
ax.plot(0, 0, 'ko', ms=6, zorder=5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x'' + 6x' + 10x = e^{-3t}$, $\;x(0)=x'(0)=0$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

tau = sym.Symbol('tau', positive=True)
f6  = tau
g6  = sym.exp(2*(t - tau))

conv6 = sym.integrate(f6 * g6, (tau, 0, t))
conv6 = sym.simplify(conv6)
display(Math(r'(t * e^{2t})(t) = ' + sym.latex(conv6)))

# Cross-check via Laplace
lap_check = sym.laplace_transform(conv6, t, s, noconds=True)
display(Math(r'\mathcal{L}[(f*g)(t)] = ' + sym.latex(sym.simplify(lap_check))
            + r' \quad \text{(should equal } 1/(s^2(s-2))\text{)}'))

f7 = sym.DiracDelta(t - 3)
ode7 = sym.Eq(x(t).diff(t, 2) + 4*x(t).diff(t) + 5*x(t), f7)

part7 = sym.dsolve(ode7, x(t), ics={x(0): 1, x(t).diff(t).subs(t, 0): -2})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part7)))

t_vals = np.linspace(0, 10, 1000)

free_resp   = np.exp(-2*t_vals)*np.cos(t_vals)
impulse_resp = (np.exp(-2*(t_vals - 3))*np.sin(t_vals - 3)
                * (t_vals >= 3).astype(float))
x_full = free_resp + impulse_resp

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals, free_resp,    color='steelblue', lw=1.4, ls='--',
        label=r'Free response $e^{-2t}\cos t$')
ax.plot(t_vals, impulse_resp, color='seagreen',  lw=1.4, ls=':',
        label=r'Impulse response $e^{-2(t-3)}\sin(t-3)\,H(t-3)$')
ax.plot(t_vals, x_full,       color='tomato',    lw=2,
        label=r'Full solution $x(t)$')
ax.axvline(3, color='gray', lw=1, ls='--', alpha=0.6)
ax.annotate(r'Impulse at $t=3$', xy=(3, 0.08), xytext=(4, 0.35),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x'' + 4x' + 5x = \delta_3(t)$, $\;x(0)=1$, $\;x'(0)=-2$",
             fontsize=13)
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.show()

import sympy as sym

t = sym.Symbol('t', positive=True)
x = sym.Function('x')

f8 = sym.DiracDelta(t - 1) + sym.DiracDelta(t - 2)
ode8 = sym.Eq(x(t).diff(t, 2) + sym.pi**2 * x(t), f8)

part8 = sym.dsolve(ode8, x(t), ics={x(0): 0, x(t).diff(t).subs(t, 0): 0})
display(Math(r'\text{Particular solution: }\quad' + sym.latex(part8)))

t_vals = np.linspace(0, 5, 1000)
pi = np.pi

x_sol = (
    (1/pi)*np.sin(pi*(t_vals - 1)) * (t_vals >= 1).astype(float)
  + (1/pi)*np.sin(pi*(t_vals - 2)) * (t_vals >= 2).astype(float)
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals, x_sol, color='tomato', lw=2, label=r'$x(t)$')
ax.axvline(1, color='steelblue', lw=1.5, ls='--', alpha=0.7,
           label=r'Impulse at $t=1$')
ax.axvline(2, color='seagreen',  lw=1.5, ls='--', alpha=0.7,
           label=r'Impulse at $t=2$')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x'' + \pi^2 x = \delta_1(t)+\delta_2(t)$, $\;x(0)=x'(0)=0$",
             fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

t = sym.Symbol('t', nonnegative=True)
x = sym.Function('x')

# Part (b): impulse response
H_expr9 = 1 / (s**2 + 2*s + 2)
h9 = sym.inverse_laplace_transform(H_expr9, s, t)
display(Math(r'h(t) = ' + sym.latex(sym.simplify(h9))))

# Part (c): delayed step response
f9 = sym.Heaviside(t - 2)
ode9 = sym.Eq(x(t).diff(t, 2) + 2*x(t).diff(t) + 2*x(t), f9)
part9 = sym.dsolve(ode9, x(t), ics={x(0): 0, x(t).diff(t).subs(t, 0): 0})
display(Math(r'x(t) = ' + sym.latex(sym.simplify(part9.rhs))))

t_vals = np.linspace(0, 10, 800)

x_sol = np.where(
    t_vals < 2,
    0.0,
    0.5 - 0.5*np.exp(-(t_vals-2))*(np.sin(t_vals-2) + np.cos(t_vals-2))
)
f_vals = (t_vals >= 2).astype(float)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals, f_vals, color='steelblue', lw=1.4, ls='--',
        label=r'Input $f(t)=H(t-2)$')
ax.plot(t_vals, x_sol,  color='tomato',    lw=2,
        label=r'Response $x(t)$')
ax.axhline(0.5, color='seagreen', lw=1, ls=':', label=r'DC gain $= 1/2$')
ax.axvline(2, color='gray', lw=1, ls='--', alpha=0.5)
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x'' + 2x' + 2x = H(t-2)$, $\;x(0)=x'(0)=0$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()