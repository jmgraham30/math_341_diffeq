#| code-fold: true
#| code-summary: "Show the code"

import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

H = lambda tv, a: np.where(np.asarray(tv, dtype=float) >= a, 1.0, 0.0)


#| code-fold: true
#| code-summary: "Show the code"

s_v = sym.Symbol('s')
t_v = sym.Symbol('t', positive=True)

examples = [
    (r"\frac{1}{(s+1)(s+2)}",       sym.Integer(1)/((s_v+1)*(s_v+2))),
    (r"\frac{1}{s^2+3s+6}",         sym.Integer(1)/(s_v**2+3*s_v+6)),
    (r"\frac{3-2s}{s^2+2s+10}",     (3-2*s_v)/(s_v**2+2*s_v+10)),
    (r"\frac{1}{(s-1)^2+1}",        sym.Integer(1)/((s_v-1)**2+1)),
    (r"\frac{2s+9}{(s+1)(s+3)}",    (2*s_v+9)/((s_v+1)*(s_v+3))),
]
print("Inverse Laplace transforms via SymPy:\n")
for label, expr in examples:
    x_t = sym.inverse_laplace_transform(expr, s_v, t_v)
    display(Math(r"\mathcal{L}^{-1}\!\left[" + label + r"\right] = " + sym.latex(sym.simplify(x_t))))
    print()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-complex-poles
#| fig-cap: "Solution of $x''-2x'+2x=0$, $x(0)=0$, $x'(0)=1$. The factor $e^t$ causes the amplitude to grow while $\\sin t$ provides the oscillation. Red dots confirm the Laplace transform result against a direct numerical solve."

t_plot = np.linspace(0, 5, 400)
x_anal = np.exp(t_plot)*np.sin(t_plot)

def ode_c(tv, y): return [y[1], 2*y[1]-2*y[0]]
sol_c = solve_ivp(ode_c, (0,5), [0,1], dense_output=True, max_step=0.01)
t_dots = np.linspace(0, 5, 20)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_plot, x_anal, color='steelblue', lw=2.5, label=r'$x(t)=e^t\sin t$')
ax.plot(t_dots, sol_c.sol(t_dots)[0], 'ro', markersize=6, label='Numerical check')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x''-2x'+2x=0$, $x(0)=0$, $x'(0)=1$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

s_v = sym.Symbol('s')
t_v = sym.Symbol('t', positive=True)

X_nh = sym.Integer(1)/((s_v-1)*(s_v+1)*(s_v+2))
pf = sym.apart(X_nh, s_v)
print("Partial fraction decomposition:")
display(Math(sym.latex(pf)))
x_nh = sym.inverse_laplace_transform(X_nh, s_v, t_v)
print("\nInverse transform:")
display(Math(r"x(t) = " + sym.latex(sym.simplify(x_nh))))


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-piecewise-f
#| fig-cap: "The piecewise function from Example 3.18: $f(t)=t$ for $0<t<1$, then $f(t)=2$ for $1\\leq t\\leq 3$, then $f(t)=0$ for $t>3$, expressed as $t+(2-t)H(t-1)-2H(t-3)$."

t_pl = np.linspace(-0.2, 5, 500)
def f318(tv):
    return np.where(tv<0, 0, np.where(tv<1, tv, np.where(tv<=3, 2, 0)))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(t_pl, f318(t_pl), color='steelblue', lw=2.5, label='$f(t)$')
ax.fill_between(t_pl, 0, f318(t_pl), alpha=0.15, color='steelblue')
ax.axvline(1, color='gray', ls=':', lw=1.2); ax.axvline(3, color='gray', ls=':', lw=1.2)
ax.text(0.5, 1.3, '$f=t$', fontsize=10, ha='center', color='steelblue')
ax.text(2.0, 2.2, '$f=2$', fontsize=10, ha='center', color='steelblue')
ax.text(4.0, 0.15, '$f=0$', fontsize=10, ha='center', color='steelblue')
ax.text(1, -0.15, '$t=1$', ha='center', fontsize=9, color='gray')
ax.text(3, -0.15, '$t=3$', ha='center', fontsize=9, color='gray')
ax.set_xlabel('$t$'); ax.set_ylabel('$f(t)$')
ax.set_title('Example 3.18: $f(t) = t + (2-t)H(t-1) - 2H(t-3)$')
ax.set_ylim(-0.3, 2.7); ax.axhline(0, color='k', lw=0.5)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex319
#| fig-cap: "Example 3.19: $x''+4x = \\sin t - H(t-\\pi)\\sin t$, $x(0)=x'(0)=0$. The forcing (top) is one arch of $\\sin t$ on $[0,\\pi]$, then zero. After the forcing turns off at $t=\\pi$ the oscillator continues with a modified amplitude and phase — the **free oscillation** driven by the residual energy stored during $[0,\\pi]$."

t_eval = np.linspace(0, 20, 2000)

def ode_319(tv, y):
    f = np.sin(tv) if tv <= np.pi else 0.0
    return [y[1], f - 4*y[0]]
sol_319 = solve_ivp(ode_319, (0,20), [0,0], t_eval=t_eval, max_step=0.01)

x_anal = (1/3*np.sin(t_eval) - 1/6*np.sin(2*t_eval)
          + H(t_eval, np.pi)*(-1/3*np.sin(t_eval) - 1/6*np.sin(2*t_eval)))
f_plot = np.where(t_eval <= np.pi, np.sin(t_eval), 0)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t_eval, f_plot, color='crimson', lw=2)
axes[0].fill_between(t_eval, 0, f_plot, alpha=0.2, color='crimson')
axes[0].axvline(np.pi, color='gray', ls='--', lw=1.2)
axes[0].set_ylabel('$f(t)$'); axes[0].set_ylim(-0.2, 1.4)
axes[0].set_title(r'Forcing: $\sin t$ on $[0,\pi]$, then off')

axes[1].plot(t_eval, x_anal, color='steelblue', lw=2, label='Analytical (3.8)')
axes[1].plot(t_eval[::40], sol_319.y[0][::40], 'ro', markersize=4, label='Numerical')
axes[1].axhline(0, color='k', lw=0.5); axes[1].axvline(np.pi, color='gray', ls='--', lw=1.2)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title(r'Solution via Laplace transform')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex321
#| fig-cap: "Example 3.21 (Logan, Figure 3.4): RC circuit charge $q(t)$ under a square-pulse voltage. The switch opens at $t=0$ (zero current), closes at $t=1$ (charge builds), opens again at $t=2$ (charge decays). The solution consists of two Heaviside-shifted exponential charging/discharging curves."

t_plot = np.linspace(0, 5, 600)
q_anal = (1/3*(1-np.exp(-3*(t_plot-1)))*H(t_plot,1)
         - 1/3*(1-np.exp(-3*(t_plot-2)))*H(t_plot,2))
f_plot = H(t_plot,1) - H(t_plot,2)

def ode_321(tv, y):
    f = 1.0 if 1 <= tv <= 2 else 0.0
    return [f - 3*y[0]]
sol_321 = solve_ivp(ode_321, (0,5), [0.0], t_eval=t_plot, max_step=0.005)

fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
axes[0].step(t_plot, f_plot, where='post', color='crimson', lw=2.5)
axes[0].fill_between(t_plot, 0, f_plot, alpha=0.2, color='crimson')
axes[0].set_ylabel('$f(t)=H(t-1)-H(t-2)$'); axes[0].set_ylim(-0.1, 1.4)
axes[0].set_title('Forcing: square pulse from $t=1$ to $t=2$')

axes[1].plot(t_plot, q_anal, color='steelblue', lw=2.5, label='Analytical')
axes[1].plot(t_plot[::25], sol_321.y[0][::25], 'ro', markersize=5, label='Numerical')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$q(t)$')
axes[1].set_title("Charge on capacitor (Logan Figure 3.4)")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-smoothing
#| fig-cap: "Smoothing by integration (Remark 3.20). The Heaviside input $f(t)=H(t-3)$ (red, discontinuous at $t=3$) produces a continuous but only piecewise-smooth $x'(t)=(t-3)H(t-3)$ (orange, kink at $t=3$), and a smoothly differentiable solution $x(t)=\\frac{1}{2}(t-3)^2H(t-3)$ (blue, parabola starting at $t=3$)."

t_plot = np.linspace(-0.5, 7, 400)
f_sm   = H(t_plot, 3)
xp_sm  = (t_plot-3)*H(t_plot, 3)
x_sm   = 0.5*(t_plot-3)**2*H(t_plot, 3)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_plot, f_sm,  color='crimson',    lw=2.5, label=r"$f(t)=H(t-3)$ (discontinuous)", zorder=3)
ax.plot(t_plot, xp_sm, color='darkorange', lw=2.5, label=r"$x'(t)=(t-3)H(t-3)$ (continuous, kink)")
ax.plot(t_plot, x_sm,  color='steelblue',  lw=2.5, label=r"$x(t)=\frac{1}{2}(t-3)^2H(t-3)$ (smooth)")
ax.axvline(3, color='gray', ls='--', lw=1.2, label='$t=3$ (jump point)')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('Function value')
ax.set_title(r'Smoothing: $x\prime\prime=H(t-3)$, $x(0)=x\prime(0)=0$')
ax.legend(fontsize=8.5); ax.set_ylim(-0.3, 4.5)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

s_v = sym.Symbol('s')
t_v = sym.Symbol('t', positive=True)

print("=" * 60)
print("Full Laplace workflow with SymPy")
print("Example: x'' + 4x = sin(t) - H(t-pi)*sin(t), x(0)=x'(0)=0")
print("=" * 60)

# Build X(s) manually
# F(s) = 1/(s^2+1) + e^{-pi*s}/(s^2+1)
# X(s) = F(s)/(s^2+4)
F_319a = sym.Integer(1)/(s_v**2+1)
F_319b = sym.exp(-sym.pi*s_v)/(s_v**2+1)
X_319  = (F_319a + F_319b)/(s_v**2+4)

print("\nX(s) =", X_319)
print("\nPartial fractions of 1/((s^2+1)(s^2+4)):")
pf = sym.apart(sym.Integer(1)/((s_v**2+1)*(s_v**2+4)), s_v)
display(Math(sym.latex(pf)))

# Invert
print("\nFull solution x(t):")
x_sol = sym.inverse_laplace_transform(X_319, s_v, t_v)
display(Math(r"x(t) = " + sym.latex(sym.simplify(x_sol))))


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m,'__version__',None)))
