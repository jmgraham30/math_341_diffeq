# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

t = sym.Symbol('t')
y = sym.Function('y')

ode1 = sym.Eq(y(t).diff(t, 2) - 5*y(t).diff(t) + 6*y(t), sym.exp(t))

gen1 = sym.dsolve(ode1, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen1)))

ode2 = sym.Eq(y(t).diff(t, 2) + y(t), sym.sec(t))

gen2 = sym.dsolve(ode2, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen2)))

t_vals = np.linspace(-np.pi/2 + 0.04, np.pi/2 - 0.04, 600)
cos_t  = np.cos(t_vals)
sin_t  = np.sin(t_vals)
yp     = cos_t * np.log(cos_t) + t_vals * sin_t

fig, ax = plt.subplots(figsize=(7, 5))

for C1, C2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1)]:
    y_sol = C1*cos_t + C2*sin_t + yp
    ax.plot(t_vals, y_sol, color='steelblue', lw=1.2, alpha=0.55)

ax.plot(t_vals, yp, color='tomato', lw=2.2,
        label=r'$y_p = \cos t\,\ln(\cos t) + t\sin t$')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_ylim(-3, 3)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' + y = \sec t$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

ode3 = sym.Eq(y(t).diff(t, 2) - 2/t**2 * y(t), 3)

gen3 = sym.dsolve(ode3, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen3)))

y1, y2 = t**2, t**(-1)
f3     = sym.Integer(3)

W3 = sym.simplify(y1*sym.diff(y2,t) - y2*sym.diff(y1,t))
display(Math(r'W = ' + sym.latex(W3)))

u1p = sym.simplify(-y2*f3 / W3)
u2p = sym.simplify( y1*f3 / W3)
display(Math(r"u_1' = " + sym.latex(u1p) +
            r",\quad u_2' = " + sym.latex(u2p)))

u1 = sym.integrate(u1p, t)
u2 = sym.integrate(u2p, t)
display(Math(r'u_1 = ' + sym.latex(u1) +
            r',\quad u_2 = ' + sym.latex(u2)))

yp3 = sym.expand(sym.simplify(u1*y1 + u2*y2))
display(Math(r'y_p = ' + sym.latex(yp3)))

t_vals = np.linspace(0.05, 3.0, 500)
yp_vals = t_vals**2 * np.log(t_vals)

fig, ax = plt.subplots(figsize=(7, 5))

for C1, C2 in [(0.5, 0.2), (-0.3, 0.5), (0.2, -0.4), (-0.2, -0.3), (0.4, 0.5)]:
    y_sol = C1*t_vals**2 + C2/t_vals + yp_vals
    ax.plot(t_vals, y_sol, color='steelblue', lw=1.2, alpha=0.55)

ax.plot(t_vals, yp_vals, color='tomato', lw=2.2,
        label=r'$y_p = t^2\ln t$')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlim(0, 3)
ax.set_ylim(-2, 6)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' - 2t^{-2}\,y = 3$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

ode4 = sym.Eq(y(t).diff(t, 2) + 4*y(t), sym.csc(2*t))

gen4 = sym.dsolve(ode4, y(t))
display(Math(r'\text{General solution: }\quad' + sym.latex(gen4)))

t_vals = np.linspace(0.03, np.pi/2 - 0.03, 600)
yp4 = (-t_vals/2)*np.cos(2*t_vals) + (1/4)*np.sin(2*t_vals)*np.log(np.sin(2*t_vals))

fig, ax = plt.subplots(figsize=(7, 5))

for C1, C2 in [(1, 0), (-1, 0), (0, 1), (0.5, -0.5), (-0.5, 0.5)]:
    y_sol = C1*np.cos(2*t_vals) + C2*np.sin(2*t_vals) + yp4
    ax.plot(t_vals, y_sol, color='steelblue', lw=1.2, alpha=0.55)

ax.plot(t_vals, yp4, color='tomato', lw=2.2,
        label=r'$y_p$  ($C_1=C_2=0$)')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlim(0, np.pi/2)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' + 4y = \csc(2t)$", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# Homogeneous solutions and forcing function
y1  = sym.exp(t)
y2  = t * sym.exp(t)
f5  = sym.exp(t) / (t**2 + 1)

# Step 1 — Wronskian
W5 = sym.simplify(y1*sym.diff(y2, t) - y2*sym.diff(y1, t))
display(Math(r'W = ' + sym.latex(W5)))

# Step 2 — parameter derivatives
u1p = sym.simplify(-y2 * f5 / W5)
u2p = sym.simplify( y1 * f5 / W5)
display(Math(r"u_1' = " + sym.latex(u1p) +
            r",\quad u_2' = " + sym.latex(u2p)))

# Step 3 — integrate (SymPy handles arctan and log correctly here)
u1 = sym.integrate(u1p, t)
u2 = sym.integrate(u2p, t)
display(Math(r'u_1 = ' + sym.latex(u1)))
display(Math(r'u_2 = ' + sym.latex(u2)))

# Step 4 — particular solution
yp5 = sym.simplify(u1*y1 + u2*y2)
display(Math(r'y_p = ' + sym.latex(yp5)))

# Step 5 — general solution  y = C1*y1 + C2*y2 + yp
C1, C2 = sym.symbols('C1 C2')
y_gen = C1*y1 + C2*y2 + yp5
display(Math(r'y(t) = ' + sym.latex(y_gen)))

# Step 6 — apply initial conditions y(0)=0, y'(0)=1
ic1 = sym.Eq(y_gen.subs(t, 0), 0)
ic2 = sym.Eq(sym.diff(y_gen, t).subs(t, 0), 1)

consts = sym.solve([ic1, ic2], [C1, C2])
display(Math(r'C_1 = ' + sym.latex(consts[C1]) +
            r',\quad C_2 = ' + sym.latex(consts[C2])))

y_ivp = sym.simplify(y_gen.subs(consts))
display(Math(r'y(t) = ' + sym.latex(y_ivp)))

resid = sym.simplify(
    y_ivp.diff(t, 2) - 2*y_ivp.diff(t) + y_ivp - sym.exp(t)/(t**2 + 1)
)
display(Math(r'\text{ODE residual: }\quad' + sym.latex(resid)))
display(Math(r'y(0) = '  + sym.latex(sym.simplify(y_ivp.subs(t, 0)))))
display(Math(r"y'(0) = " + sym.latex(sym.simplify(y_ivp.diff(t).subs(t, 0)))))

t_vals   = np.linspace(0, 2.5, 500)
y_ivp_plot = np.exp(t_vals)*(t_vals + t_vals*np.arctan(t_vals)
                             - 0.5*np.log(t_vals**2 + 1))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(t_vals, y_ivp_plot,            color='tomato',    lw=2.2,
        label=r'$y(t)$ (IVP solution)')
ax.plot(t_vals, np.exp(t_vals),     color='steelblue', lw=1.2, ls='--',
        label=r'$e^t$')
ax.plot(t_vals, t_vals*np.exp(t_vals), color='gray',   lw=1.2, ls=':',
        label=r'$te^t$')
ax.plot(0, 0, 'ko', ms=6, zorder=5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"$y'' - 2y' + y = e^t/(t^2+1)$, $\;y(0)=0$, $\;y'(0)=1$",
             fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()