import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

t_plot = np.linspace(0, 5, 400)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# ── Case I: Distinct real eigenvalues ───────────────────────
# x'' - 3x' + 2x = 0, lambda = 1, 2; x(0)=1, x'(0)=0
# x(0) = C1 + C2 = 1; x'(0) = C1 + 2*C2 = 0 => C2=-1, C1=2
C1, C2 = 2.0, -1.0
x_case1 = C1*np.exp(t_plot) + C2*np.exp(2*t_plot)
# Also show a stable case: x'' + 3x' + 2x = 0, lambda=-1,-2
# x(0)=1, x'(0)=0: C1+C2=1, -C1-2*C2=0 => C2=-1, C1=2... wait
# -C1-2*C2=0 => C1=-2C2; C1+C2=1 => -2C2+C2=1 => C2=-1, C1=2
C1s, C2s = 2.0, -1.0
x_stable = C1s*np.exp(-t_plot) + C2s*np.exp(-2*t_plot)
axes[0].plot(t_plot, x_case1, color='crimson', lw=2, label=r'$\lambda=1,2$ (unstable, $p=-3$)')
axes[0].plot(t_plot, x_stable, color='steelblue', lw=2, label=r'$\lambda=-1,-2$ (stable, $p=3$)')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title('Case I: Distinct real $\lambda$')
axes[0].legend(fontsize=8); axes[0].set_ylim(-2, 5)

# ── Case II: Repeated eigenvalue ────────────────────────────
# x'' + 2*gamma*x' + gamma^2*x = 0 for various gamma
for gamma_val, color, lbl in zip([0.5, 1.0, 2.0],
                                  ['steelblue','darkorange','crimson'],
                                  [r'$\lambda=-0.5$ (rep.)',r'$\lambda=-1$ (rep.)',r'$\lambda=-2$ (rep.)']):
    # x(0)=1, x'(0)=0: C1=1, C2*lambda + C1*lambda*0 => diff and apply
    # x=(C1+C2t)e^(lambda*t); x(0)=C1=1; x'=(C2+lambda(C1+C2t))e^lambda*t
    # x'(0)=C2+lambda*C1=0 => C2=-lambda=gamma_val
    C1r, C2r = 1.0, gamma_val
    x_rep = (C1r + C2r*t_plot)*np.exp(-gamma_val*t_plot)
    axes[1].plot(t_plot, x_rep, color=color, lw=2, label=lbl)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title('Case II: Repeated $\lambda$')
axes[1].legend(fontsize=8)

# ── Case III: Complex eigenvalues ───────────────────────────
# x'' + 2*gamma*x' + (gamma^2+beta^2)x=0; beta=2
beta = 2.0
for gamma_val, color, lbl in zip([0.0, 0.3, 0.8],
                                   ['steelblue','darkorange','crimson'],
                                   ['$\\gamma=0$ (undamped)','$\\gamma=0.3$ (underdamped)','$\\gamma=0.8$ (underdamped)']):
    # x(0)=1, x'(0)=0: C1=1, alpha*C1 + beta*C2=0 => C2=gamma/beta
    C1c = 1.0
    C2c = gamma_val / beta
    x_complex = np.exp(-gamma_val*t_plot)*(C1c*np.cos(beta*t_plot) + C2c*np.sin(beta*t_plot))
    axes[2].plot(t_plot, x_complex, color=color, lw=2, label=lbl)
# Envelope
axes[2].plot(t_plot,  np.exp(-0.3*t_plot), color='darkorange', lw=1, ls=':', alpha=0.8)
axes[2].plot(t_plot, -np.exp(-0.3*t_plot), color='darkorange', lw=1, ls=':', alpha=0.8)
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_xlabel('$t$'); axes[2].set_ylabel('$x(t)$')
axes[2].set_title('Case III: Complex $\\lambda = \\alpha\\pm i\\beta$')
axes[2].legend(fontsize=8)

plt.suptitle('Three Cases of the Characteristic Equation', fontsize=12)
plt.tight_layout()
plt.show()

omega0 = 1.0
t_plot = np.linspace(0, 12, 600)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── All three regimes ───────────────────────────────────────
def dho(t, y, gamma, omega0):
    return [y[1], -2*gamma*y[1] - omega0**2*y[0]]

params = [(0.1, 'steelblue',   'Underdamped $\\gamma=0.1$'),
          (1.0, 'darkorange',  'Critically damped $\\gamma=1.0$'),
          (2.0, 'crimson',     'Overdamped $\\gamma=2.0$')]

for gamma, color, lbl in params:
    sol = solve_ivp(dho, (0,12), [1.0, 0.0], args=(gamma, omega0),
                    t_eval=t_plot, max_step=0.02)
    axes[0].plot(sol.t, sol.y[0], color=color, lw=2, label=lbl)

axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title(r'Three damping regimes ($\omega_0=1$, $x(0)=1$, $x\'(0)=0$)')
axes[0].legend(fontsize=8.5)

# ── Phase portrait ──────────────────────────────────────────
for gamma, color, lbl in params:
    sol = solve_ivp(dho, (0,12), [1.0, 0.0], args=(gamma, omega0),
                    t_eval=t_plot, max_step=0.02)
    axes[1].plot(sol.y[0], sol.y[1], color=color, lw=2, label=lbl)
axes[1].plot(1, 0, 'ko', markersize=7, label='IC $(1,0)$', zorder=5)
axes[1].plot(0, 0, 'k*', markersize=10, label='Origin (equilibrium)', zorder=5)
axes[1].set_xlabel('$x$'); axes[1].set_ylabel("$x'$")
axes[1].set_title('Phase portrait')
axes[1].legend(fontsize=8)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

# Verify with SymPy
t_sym = sym.Symbol('t')
x_sym = sym.Function('x')
ode_ex4 = sym.Eq(x_sym(t_sym).diff(t_sym,2) + 2*x_sym(t_sym).diff(t_sym) + 5*x_sym(t_sym), 0)
sol_ex4 = sym.dsolve(ode_ex4, x_sym(t_sym),
                     ics={x_sym(0): 1, x_sym(t_sym).diff(t_sym).subs(t_sym,0): 2})
print("IVP solution:")
display(Math(sym.latex(sol_ex4)))

t_plot = np.linspace(0, 5, 400)
x_sol = np.exp(-t_plot)*(np.cos(2*t_plot) + 1.5*np.sin(2*t_plot))
envelope = np.sqrt(1 + 1.5**2) * np.exp(-t_plot)   # amplitude = sqrt(C1^2+C2^2)

def ode_num(t, y): return [y[1], -2*y[1] - 5*y[0]]
sol_num = solve_ivp(ode_num, (0,5), [1.0, 2.0], dense_output=True, max_step=0.02)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_plot, x_sol, color='steelblue', lw=2.5, label=r'$e^{-t}(\cos 2t+\frac{3}{2}\sin 2t)$')
ax.plot(t_plot,  envelope, color='gray', lw=1.2, ls=':', label='Envelope $\\pm\\sqrt{C_1^2+C_2^2}e^{-t}$')
ax.plot(t_plot, -envelope, color='gray', lw=1.2, ls=':')
t_dots = np.linspace(0, 5, 20)
ax.plot(t_dots, sol_num.sol(t_dots)[0], 'ro', markersize=5, label='Numerical check')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"IVP: $x''+2x'+5x=0$, $x(0)=1$, $x'(0)=2$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

t_sym = sym.Symbol('t')
x_sym = sym.Function('x')

cases = {
    r"x'' - 3x' + 2x = 0 \;\;(\Delta>0)":
        x_sym(t_sym).diff(t_sym,2) - 3*x_sym(t_sym).diff(t_sym) + 2*x_sym(t_sym),
    r"x'' - 4x' + 4x = 0 \;\;(\Delta=0)":
        x_sym(t_sym).diff(t_sym,2) - 4*x_sym(t_sym).diff(t_sym) + 4*x_sym(t_sym),
    r"x'' + 2x' + 5x = 0 \;\;(\Delta<0)":
        x_sym(t_sym).diff(t_sym,2) + 2*x_sym(t_sym).diff(t_sym) + 5*x_sym(t_sym),
}

for label, expr in cases.items():
    ode = sym.Eq(expr, 0)
    sol = sym.dsolve(ode, x_sym(t_sym))
    expanded = sym.expand(sol.rhs)
    print(f"ODE: ${label}$")
    display(Math(r"x(t) = " + sym.latex(expanded)))
    print()

t_sym = sym.Symbol('t')
x_sym = sym.Function('x')

# Example 5
ode5 = sym.Eq(x_sym(t_sym).diff(t_sym,2) + 2*x_sym(t_sym).diff(t_sym) + 5*x_sym(t_sym),
              3*sym.exp(t_sym))
sol5 = sym.dsolve(ode5, x_sym(t_sym))
print("Example 5 (exponential forcing):")
display(Math(r"x(t) = " + sym.latex(sol5.rhs)))
print()

# Example 6
ode6 = sym.Eq(x_sym(t_sym).diff(t_sym,2) + x_sym(t_sym).diff(t_sym) - 2*x_sym(t_sym),
              t_sym**2)
sol6 = sym.dsolve(ode6, x_sym(t_sym))
print("Example 6 (polynomial forcing):")
display(Math(r"x(t) = " + sym.latex(sol6.rhs)))

# SymPy verification for Example 8b
t_sym = sym.Symbol('t')
x_sym = sym.Function('x')
ode8b = sym.Eq(x_sym(t_sym).diff(t_sym,2) + 4*x_sym(t_sym).diff(t_sym) + 4*x_sym(t_sym),
               sym.exp(-2*t_sym))
sol8b = sym.dsolve(ode8b, x_sym(t_sym))
print("Example 8b (repeated eigenvalue, exponential forcing):")
display(Math(r"x(t) = " + sym.latex(sol8b.rhs)))

t_plot = np.linspace(0, 20, 1000)

def forced_osc(t, y, Omega):
    return [y[1], -4*y[0] + np.cos(Omega*t)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Non-resonant
sol_nr = solve_ivp(forced_osc, (0,20), [0,0], args=(1.0,), t_eval=t_plot, max_step=0.02)
axes[0].plot(sol_nr.t, sol_nr.y[0], color='steelblue', lw=2)
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title(r"Non-resonant: $\Omega=1$, $\omega_0=2$")
axes[0].set_ylim(-1, 1)

# Resonant
sol_r = solve_ivp(forced_osc, (0,20), [0,0], args=(2.0,), t_eval=t_plot, max_step=0.02)
axes[1].plot(sol_r.t, sol_r.y[0], color='crimson', lw=2, label='Solution $x(t)$')
axes[1].plot(t_plot,  t_plot/4, color='gray', lw=1.5, ls='--', label='Envelope $\\pm t/4$')
axes[1].plot(t_plot, -t_plot/4, color='gray', lw=1.5, ls='--')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title(r"Resonance: $\Omega=\omega_0=2$")
axes[1].legend(fontsize=9)

plt.suptitle(r"$x'' + 4x = \cos(\Omega t)$, $x(0)=x'(0)=0$", fontsize=12)
plt.tight_layout()
plt.show()