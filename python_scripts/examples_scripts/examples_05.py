# This is a code cell that imports the necessary libraries for our session.
import sympy as sym                       # SymPy for symbolic mathematics
import numpy as np                        # NumPy for numerical computations
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Declare the symbolic variables used throughout
t, s = sym.symbols('t s', real=True)

f1 = 2 - 3*sym.exp(4*t) + t**2 * sym.exp(-t)
F1 = sym.laplace_transform(f1, t, s, noconds=True)
display(Math(
    r'\mathcal{L}\!\left[2 - 3e^{4t} + t^2 e^{-t}\right] = '
    + sym.latex(sym.simplify(F1))))

f2 = (t + 1)*sym.Heaviside(t - 2)
F2 = sym.laplace_transform(f2, t, s, noconds=True)
display(Math(
    r'\mathcal{L}[(t+1)H(t-2)] = '
    + sym.latex(sym.simplify(F2))))

F3 = sym.laplace_transform(sym.sin(3*t), t, s, noconds=True)
display(Math(r'\mathcal{L}[\sin 3t] = ' + sym.latex(F3)))

f4 = sym.cos(t - sym.pi/2)
F4 = sym.laplace_transform(f4, t, s, noconds=True)
display(Math(
    r'\mathcal{L}\!\left[\cos(t-\pi/2)\right] = '
    + sym.latex(sym.simplify(F4))))

f5 = 5*sym.exp(2*t)*sym.sinh(3*t)
F5 = sym.laplace_transform(f5, t, s, noconds=True)
display(Math(
    r'\mathcal{L}[5e^{2t}\sinh 3t] = '
    + sym.latex(sym.simplify(F5))))

f6 = sym.Heaviside(t - 2)*sym.sin(t - 2)
F6 = sym.laplace_transform(f6, t, s, noconds=True)
display(Math(
    r'\mathcal{L}[H(t-2)\sin(t-2)] = '
    + sym.latex(sym.simplify(F6))))

F7 = sym.Integer(4)/(s - 3)
f7 = sym.inverse_laplace_transform(F7, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{4}{s-3}\right] = '
    + sym.latex(f7)))

F8 = sym.Integer(5)/s + sym.Integer(3)/(s**2 + 4)
f8 = sym.inverse_laplace_transform(F8, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{5}{s}+\frac{3}{s^2+4}\right] = '
    + sym.latex(sym.simplify(f8))))

F9 = sym.Integer(6)/(s + 4)**3
f9 = sym.inverse_laplace_transform(F9, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{6}{(s+4)^3}\right] = '
    + sym.latex(f9)))

F10 = (sym.Integer(3)/s)*sym.exp(-2*s)
f10 = sym.inverse_laplace_transform(F10, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{3}{s}e^{-2s}\right] = '
    + sym.latex(f10)))

F11 = (sym.Integer(2)/(s*(s + 4)))*sym.exp(-3*s)
f11 = sym.inverse_laplace_transform(F11, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{2}{s(s+4)}e^{-3s}\right] = '
    + sym.latex(sym.simplify(f11))))

F12 = (4*s - 3)/(s**2 + 9)
f12 = sym.inverse_laplace_transform(F12, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{4s-3}{s^2+9}\right] = '
    + sym.latex(sym.simplify(f12))))

F13 = sym.Integer(5)/(3*s**2 + 12)
f13 = sym.inverse_laplace_transform(F13, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{5}{3s^2+12}\right] = '
    + sym.latex(sym.simplify(f13))))

F14 = sym.exp(-sym.pi*s/2)*(sym.Integer(4)/(s**2 + 4))
f14 = sym.inverse_laplace_transform(F14, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[e^{-\pi s/2}\frac{4}{s^2+4}\right] = '
    + sym.latex(sym.simplify(f14))))

F15 = (3*s + 7)/(s**2 + 6*s + 13)
f15 = sym.inverse_laplace_transform(F15, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{3s+7}{s^2+6s+13}\right] = '
    + sym.latex(sym.simplify(f15))))

t_vals  = np.linspace(0, 3, 600)
f_vals  = np.exp(-3*t_vals)*(3*np.cos(2*t_vals) - np.sin(2*t_vals))
env_pos =  np.sqrt(10)*np.exp(-3*t_vals)   # amplitude = sqrt(3^2 + 1^2)
env_neg = -env_pos

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_vals, f_vals,  color='tomato',    lw=2.2,
        label=r'$e^{-3t}[3\cos(2t)-\sin(2t)]$')
ax.plot(t_vals, env_pos, color='steelblue', lw=1.2, ls='--',
        label=r'$\pm\sqrt{10}\,e^{-3t}$ (envelope)')
ax.plot(t_vals, env_neg, color='steelblue', lw=1.2, ls='--')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$f(t)$', fontsize=13)
ax.set_title(
    r'$\mathcal{L}^{-1}\!\left[\frac{3s+7}{s^2+6s+13}\right]$',
    fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

F16 = (sym.Integer(5)/s**2)*sym.exp(-s)
f16 = sym.inverse_laplace_transform(F16, s, t)
display(Math(
    r'\mathcal{L}^{-1}\!\left[\frac{5}{s^2}e^{-s}\right] = '
    + sym.latex(sym.simplify(f16))))

t_vals     = np.linspace(0, 4, 500)
f_delayed  = 5*(t_vals - 1)*np.where(t_vals >= 1, 1, 0)
f_undelayed = 5*t_vals

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_vals, f_delayed,   color='tomato',    lw=2.2,
        label=r'$5(t-1)H(t-1)$')
ax.plot(t_vals, f_undelayed, color='steelblue', lw=1.4, ls='--',
        label=r'$5t$ (undelayed)')
ax.axvline(1, color='gray', lw=1, ls=':')
ax.annotate(r'$t = 1$', xy=(1, 0.15), xytext=(1.1, 0.5),
            fontsize=10, color='gray')
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$f(t)$', fontsize=13)
ax.set_title(
    r'$\mathcal{L}^{-1}\!\left[5e^{-s}/s^2\right]$ — delayed ramp',
    fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(-0.5, 16)
plt.tight_layout()
plt.show()