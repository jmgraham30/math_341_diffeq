import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import display, Math, Image
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False

def f_ex(t, x):
    return t - x

def euler(f, t0, T, x0, N):
    h = (T - t0) / N
    t = np.linspace(t0, T, N+1)
    X = np.zeros(N+1); X[0] = x0
    for n in range(N):
        X[n+1] = X[n] + h * f(t[n], X[n])
    return t, X

def exact_ex(t):
    return t - 1 + 2*np.exp(-t)

t0, T, x0 = 0.0, 2.0, 1.0
t_fine = np.linspace(t0, T, 400)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_fine, exact_ex(t_fine), 'steelblue', lw=2.8, label='Exact $x(t)=t-1+2e^{-t}$')

colors = ['crimson', 'darkorange', 'seagreen']
Nvals  = [4, 8, 16]
hvals  = [(T-t0)/N for N in Nvals]

for N, color, h in zip(Nvals, colors, hvals):
    t_e, X_e = euler(f_ex, t0, T, x0, N)
    err = abs(exact_ex(T) - X_e[-1])
    ax.plot(t_e, X_e, color=color, lw=1.8, marker='o', markersize=5,
            label=f'Euler $h={h}$, error at $t=2$: {err:.4f}')

ax.set_xlabel('$t$'); ax.set_ylabel('$x$')
ax.set_title("Example 6.3: Euler method convergence")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# Error table
print(f"{'h':>8}  {'X(2)':>8}  {'error':>8}")
print("-" * 28)
for N, h in zip(Nvals, hvals):
    _, X_e = euler(f_ex, t0, T, x0, N)
    err = exact_ex(T) - X_e[-1]
    print(f"{h:8.4f}  {X_e[-1]:8.4f}  {err:8.4f}")
print(f"{'exact':>8}  {exact_ex(T):8.4f}  {'0':>8}")

def modified_euler(f, t0, T, x0, N):
    h = (T - t0) / N
    t = np.linspace(t0, T, N+1)
    X = np.zeros(N+1); X[0] = x0
    for n in range(N):
        Xt = X[n] + h * f(t[n], X[n])
        X[n+1] = X[n] + (h/2) * (f(t[n], X[n]) + f(t[n+1], Xt))
    return t, X

def rk4(f, t0, T, x0, N):
    h = (T - t0) / N
    t = np.linspace(t0, T, N+1)
    X = np.zeros(N+1); X[0] = x0
    for n in range(N):
        k1 = f(t[n],       X[n])
        k2 = f(t[n]+h/2,   X[n]+h/2*k1)
        k3 = f(t[n]+h/2,   X[n]+h/2*k2)
        k4 = f(t[n]+h,     X[n]+h*k3)
        X[n+1] = X[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t, X

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: solution comparison with h=0.5
N_demo = 4
t_fine = np.linspace(0, 2, 400)
axes[0].plot(t_fine, exact_ex(t_fine), 'steelblue', lw=2.8,
             label='Exact')
for method, color, lbl in [
        (euler,          'crimson',    'Euler (RK1)'),
        (modified_euler, 'darkorange', 'Modified Euler (RK2)'),
        (rk4,            'seagreen',   'Runge–Kutta (RK4)')]:
    t_m, X_m = method(f_ex, 0, 2, 1, N_demo)
    axes[0].plot(t_m, X_m, marker='o', markersize=6, lw=2.0,
                 color=color, label=lbl)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x$')
axes[0].set_title('Solution comparison ($h=0.5$)')
axes[0].legend(fontsize=9)

# Right: convergence log-log
h_vals = [0.5, 0.25, 0.125, 0.0625, 0.03125]
for method, color, lbl, order in [
        (euler,          'crimson',    'Euler $O(h)$',    1),
        (modified_euler, 'darkorange', 'RK2 $O(h^2)$',   2),
        (rk4,            'seagreen',   'RK4 $O(h^4)$',   4)]:
    errs = []
    for h in h_vals:
        N_ = int(2/h)
        _, X_m = method(f_ex, 0, 2, 1, N_)
        errs.append(abs(exact_ex(2) - X_m[-1]))
    axes[1].loglog(h_vals, errs, marker='o', color=color, lw=2.0, label=lbl)
    # Reference slope
    h_arr = np.array(h_vals)
    axes[1].loglog(h_arr, errs[0]*(h_arr/h_vals[0])**order,
                   ls='--', color=color, lw=1.0, alpha=0.6)

axes[1].set_xlabel('Step size $h$'); axes[1].set_ylabel('Global error at $t=2$')
axes[1].set_title('Convergence (log–log)')
axes[1].legend(fontsize=9)
axes[1].grid(True, which='both', alpha=0.3)

plt.suptitle(r"Method comparison: $x'=t-x$, $x(0)=1$", fontsize=12)
plt.tight_layout()
plt.show()

# Verify the concrete error table values above (all method functions now defined)
_methods = [('Euler', euler), ('Modified Euler (RK2)', modified_euler), ('RK4', rk4)]
print(f"{'Method':<24} {'h=0.5 error':>12} {'h=0.25 error':>13} {'Ratio':>7}")
print("-" * 60)
for name, method in _methods:
    err_half    = abs(exact_ex(2) - method(f_ex, 0, 2, 1,  4)[-1][-1])
    err_quarter = abs(exact_ex(2) - method(f_ex, 0, 2, 1,  8)[-1][-1])
    ratio = err_half / err_quarter
    print(f"{name:<24} {err_half:>12.7f} {err_quarter:>13.7f} {ratio:>7.2f}")

from scipy.integrate import solve_ivp

# Define the right-hand side: must accept (t, y) where y is a 1-D array
def rhs(t, y):
    return [t - y[0]]       # y[0] is x(t)

# Solve
sol = solve_ivp(
    fun     = rhs,          # right-hand side f(t, y)
    t_span  = (0, 2),       # integration interval [t0, T]
    y0      = [1.0],        # initial condition (list or array)
    method  = 'RK45',       # solver (default; adaptive 4th/5th-order RK)
    dense_output = True,    # enable continuous interpolant sol.sol(t)
    rtol    = 1e-6,         # relative tolerance
    atol    = 1e-8,         # absolute tolerance
)

print(f"Success: {sol.success}")
print(f"Solver steps taken: {len(sol.t)}")
print(f"x(2) ≈ {sol.y[0, -1]:.8f}  (exact: {exact_ex(2):.8f})")

# Plot using the dense interpolant
t_plot = np.linspace(0, 2, 400)
x_interp = sol.sol(t_plot)[0]   # shape (1, 400) → take row 0

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_plot, exact_ex(t_plot), 'steelblue', lw=2.5, label='Exact')
ax.plot(t_plot, x_interp, 'seagreen', lw=2.0, ls='--', label='solve_ivp RK45')
ax.plot(sol.t, sol.y[0], 'go', markersize=5,
        label=f'Adaptive steps (N={len(sol.t)})')
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x'=t-x$, $x(0)=1$ via `solve_ivp`")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# Lotka-Volterra: x' = ax - bxy,  y' = -cy + dxy
a, b, c, d = 0.6, 0.5, 0.3, 0.4

def lv_rhs(t, z):
    x, y = z
    return [a*x - b*x*y,
            -c*y + d*x*y]

sol_lv = solve_ivp(
    fun          = lv_rhs,
    t_span       = (0, 60),
    y0           = [c/d + 0.4, a/b],   # perturbed from equilibrium
    method       = 'RK45',
    dense_output = True,
    rtol         = 1e-9,
    atol         = 1e-11,
    max_step     = 0.05,
)

t_p = np.linspace(0, 60, 1500)
z_p = sol_lv.sol(t_p)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(t_p, z_p[0], 'steelblue', lw=2.5, label='$x(t)$ prey')
axes[0].plot(t_p, z_p[1], 'crimson',   lw=2.5, label='$y(t)$ predator')
axes[0].axhline(c/d, color='steelblue', ls=':', lw=1, alpha=0.7)
axes[0].axhline(a/b, color='crimson',   ls=':', lw=1, alpha=0.7)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('Population')
axes[0].set_title('Lotka–Volterra time series')
axes[0].legend(fontsize=9)

axes[1].plot(z_p[0], z_p[1], 'seagreen', lw=1.8)
axes[1].plot(c/d, a/b, 'o', color='steelblue', markersize=9,
             label=f'Equilibrium $({c/d:.2f},{a/b:.2f})$')
axes[1].set_xlabel('$x$ (prey)'); axes[1].set_ylabel('$y$ (predator)')
axes[1].set_title('Phase portrait')
axes[1].legend(fontsize=9)

plt.suptitle(f'Lotka–Volterra ($a={a}$, $b={b}$, $c={c}$, $d={d}$)', fontsize=11)
plt.tight_layout()
plt.show()