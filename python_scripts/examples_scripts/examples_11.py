# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from scipy.integrate import solve_ivp     # SciPy ODE solver (for verification)
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

def euler(f, t0, x0, T, h):
    """Euler method for x' = f(t,x), returns (t_arr, X_arr)."""
    ts = [t0]; xs = [x0]
    t, x = t0, x0
    while t < T - 1e-12:
        x = x + h * f(t, x)
        t = t + h
        ts.append(t); xs.append(x)
    return np.array(ts), np.array(xs)

def modified_euler(f, t0, x0, T, h):
    """Modified Euler (Heun) method."""
    ts = [t0]; xs = [x0]
    t, x = t0, x0
    while t < T - 1e-12:
        k1 = f(t, x)
        k2 = f(t + h, x + h * k1)
        x  = x + (h / 2) * (k1 + k2)
        t  = t + h
        ts.append(t); xs.append(x)
    return np.array(ts), np.array(xs)

def rk4(f, t0, x0, T, h):
    """Classical Runge–Kutta (RK4) method."""
    ts = [t0]; xs = [x0]
    t, x = t0, x0
    while t < T - 1e-12:
        k1 = f(t,           x)
        k2 = f(t + h/2,     x + (h/2)*k1)
        k3 = f(t + h/2,     x + (h/2)*k2)
        k4 = f(t + h,       x + h*k3)
        x  = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t  = t + h
        ts.append(t); xs.append(x)
    return np.array(ts), np.array(xs)

# IVP definition
f1   = lambda t, x: x - 2*t
x_exact1 = lambda t: 2*t + 2 - np.exp(t)
t0, x0, T = 0.0, 1.0, 2.0

# Solve with h = 0.5 for comparison
h = 0.5
t_e,  X_e  = euler(f1, t0, x0, T, h)
t_me, X_me = modified_euler(f1, t0, x0, T, h)
t_rk, X_rk = rk4(f1, t0, x0, T, h)

print(f"Exact x(2)           = {x_exact1(2):.6f}")
print(f"Euler x(2)           = {X_e[-1]:.6f}  (error {abs(X_e[-1]-x_exact1(2)):.6f})")
print(f"Modified Euler x(2)  = {X_me[-1]:.6f}  (error {abs(X_me[-1]-x_exact1(2)):.6f})")
print(f"RK4 x(2)             = {X_rk[-1]:.6f}  (error {abs(X_rk[-1]-x_exact1(2)):.2e})")

# Error at t=2 for several step sizes
step_sizes = [0.1, 0.01, 0.001, 0.0001]
exact_T = x_exact1(T)

print(f"{'h':>8}  {'Euler err':>12}  {'Mod.Euler err':>14}  {'RK4 err':>12}")
print("-" * 54)
for h in step_sizes:
    _, Xe  = euler(f1, t0, x0, T, h)
    _, Xme = modified_euler(f1, t0, x0, T, h)
    _, Xrk = rk4(f1, t0, x0, T, h)
    print(f"{h:>8.4f}  {abs(Xe[-1]-exact_T):>12.2e}  "
          f"{abs(Xme[-1]-exact_T):>14.2e}  {abs(Xrk[-1]-exact_T):>12.2e}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: solution comparison
ax = axes[0]
t_fine = np.linspace(0, 2, 400)
ax.plot(t_fine, x_exact1(t_fine), 'k-',  lw=2.2, label='Exact', zorder=5)
ax.plot(t_e,  X_e,  'o--', color='tomato',    lw=1.5, ms=6, label='Euler ($h=0.5$)')
ax.plot(t_me, X_me, 's--', color='darkorange', lw=1.5, ms=6, label='Mod. Euler ($h=0.5$)')
ax.plot(t_rk, X_rk, '^--', color='steelblue',  lw=1.5, ms=6, label='RK4 ($h=0.5$)')
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x' = x - 2t$, $x(0) = 1$", fontsize=11)
ax.legend(fontsize=9)

# Right: error convergence
ax = axes[1]
hs  = np.array([0.2, 0.1, 0.05, 0.025, 0.01])
err_e = []; err_me = []; err_rk = []
for h in hs:
    _, Xe  = euler(f1, t0, x0, T, h)
    _, Xme = modified_euler(f1, t0, x0, T, h)
    _, Xrk = rk4(f1, t0, x0, T, h)
    err_e.append(abs(Xe[-1]  - exact_T))
    err_me.append(abs(Xme[-1] - exact_T))
    err_rk.append(abs(Xrk[-1] - exact_T))

ax.loglog(hs, err_e,  'o-', color='tomato',    lw=1.8, ms=7, label='Euler')
ax.loglog(hs, err_me, 's-', color='darkorange', lw=1.8, ms=7, label='Mod. Euler')
ax.loglog(hs, err_rk, '^-', color='steelblue',  lw=1.8, ms=7, label='RK4')

# Reference slopes
ax.loglog(hs, 3*hs,       'k:',  lw=1.2, label=r'$O(h)$')
ax.loglog(hs, 4*hs**2,    'k--', lw=1.2, label=r'$O(h^2)$')
ax.loglog(hs, 0.1*hs**4,  'k-.',  lw=1.2, label=r'$O(h^4)$')
ax.set_xlabel(r'Step size $h$', fontsize=13)
ax.set_ylabel(r'$|X_N - x(2)|$', fontsize=13)
ax.set_title('Error convergence at $t=2$', fontsize=11)
ax.legend(fontsize=8)

plt.suptitle(r"Example 1: $x'=x-2t$, $x(0)=1$", fontsize=11)
plt.tight_layout()
plt.show()

f2 = lambda t, x: -5*x + 5*np.cos(t) + np.sin(t)
x_exact2 = lambda t: np.cos(t) + 0.4*np.sin(t) - (12/13)*np.exp(-5*t)
t0, x0, T2 = 0.0, 0.0, 10.0

fig, ax = plt.subplots(figsize=(9, 5))
t_fine = np.linspace(0, T2, 1000)
ax.plot(t_fine, x_exact2(t_fine), 'k-', lw=2.5, label='Exact', zorder=10)

for N, col, alpha in [(30, 'tomato', 0.9), (100, 'darkorange', 0.8),
                       (300, 'steelblue', 0.75), (1000, 'seagreen', 0.7)]:
    h = T2 / N
    te, Xe = euler(f2, t0, x0, T2, h)
    ax.plot(te, Xe, lw=1.4, alpha=alpha, color=col,
            label=f'Euler $N={N}$ ($h={h:.3f}$)')

ax.set_xlim(0, T2)
ax.set_ylim(-1.8, 2.0)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$x$', fontsize=13)
ax.set_title(r"$x' = -5x + 5\cos t + \sin t$, $x(0)=0$", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

f3 = lambda t, x: 0.4*x*(1 - x/8) - 0.3*np.sin(np.pi*t/6)**2
t0, x0, T3 = 0.0, 3.0, 60.0

h = 0.5
te3, Xe3 = euler(f3, t0, x0, T3, h)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(te3, Xe3, color='steelblue', lw=1.8, label='Euler ($h=0.5$ months)')
ax.axhline(8.0, color='gray', ls='--', lw=1.2, label='Carrying capacity $K=8$')
ax.axhline(np.mean(Xe3[len(Xe3)//2:]), color='tomato', ls=':', lw=1.5,
           label=f'Long-term mean ≈ {np.mean(Xe3[len(Xe3)//2:]):.2f}')
ax.set_xlabel(r'$t$ (months)', fontsize=13)
ax.set_ylabel(r'Population (thousands)', fontsize=13)
ax.set_title(r'Logistic growth with seasonal harvesting: $x(0)=3$', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

def rk4_system(f, g, t0, x0, y0, T, h):
    """RK4 for the 2x2 system x'=f(t,x,y), y'=g(t,x,y)."""
    ts = [t0]; xs = [x0]; ys = [y0]
    t, x, y = t0, x0, y0
    while t < T - 1e-12:
        k11 = f(t, x, y);            k21 = g(t, x, y)
        k12 = f(t+h/2, x+h/2*k11, y+h/2*k21)
        k22 = g(t+h/2, x+h/2*k11, y+h/2*k21)
        k13 = f(t+h/2, x+h/2*k12, y+h/2*k22)
        k23 = g(t+h/2, x+h/2*k12, y+h/2*k22)
        k14 = f(t+h,   x+h*k13,   y+h*k23)
        k24 = g(t+h,   x+h*k13,   y+h*k23)
        x = x + h/6*(k11 + 2*k12 + 2*k13 + k14)
        y = y + h/6*(k21 + 2*k22 + 2*k23 + k24)
        t = t + h
        ts.append(t); xs.append(x); ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys)

f4 = lambda t, x, y: -x + y
g4 = lambda t, x, y: -y
x_ex4 = lambda t: (t + 2)*np.exp(-t)
y_ex4 = lambda t: np.exp(-t)

h = 0.5
ts4, Xs4, Ys4 = rk4_system(f4, g4, 0.0, 2.0, 1.0, 3.0, h)

print(f"{'t':>5}  {'X_n':>10}  {'x_exact':>10}  {'error_x':>10}")
for t, X, xe in zip(ts4, Xs4, x_ex4(ts4)):
    print(f"{t:>5.2f}  {X:>10.6f}  {xe:>10.6f}  {abs(X-xe):>10.2e}")

fig, ax = plt.subplots(figsize=(8, 5))
t_fine = np.linspace(0, 3, 300)
ax.plot(t_fine, x_ex4(t_fine), 'k-',  lw=2.0, label=r'Exact $x(t)=(t+2)e^{-t}$')
ax.plot(t_fine, y_ex4(t_fine), 'k--', lw=2.0, label=r'Exact $y(t)=e^{-t}$')
ax.plot(ts4, Xs4, 'o', color='steelblue', ms=7, zorder=5, label=r'RK4 $X_n$')
ax.plot(ts4, Ys4, 's', color='darkorange', ms=7, zorder=5, label=r'RK4 $Y_n$')
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'Solution', fontsize=13)
ax.set_title(r"System: $x'=-x+y$, $y'=-y$; $x(0)=2$, $y(0)=1$", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

f5 = lambda t, x, y: 0.5*x - 0.1*x*y
g5 = lambda t, x, y: -0.4*y + 0.05*x*y

ts5, Xs5, Ys5 = rk4_system(f5, g5, 0.0, 5.0, 2.0, 40.0, h=0.05)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Time series
ax = axes[0]
ax.plot(ts5, Xs5, color='steelblue',  lw=1.8, label='Prey $x(t)$ (rabbits)')
ax.plot(ts5, Ys5, color='darkorange', lw=1.8, label='Predator $y(t)$ (foxes)')
ax.axhline(8, color='steelblue',  ls=':', lw=1.0, alpha=0.5)
ax.axhline(5, color='darkorange', ls=':', lw=1.0, alpha=0.5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel('Population (hundreds)', fontsize=13)
ax.set_title('Predator–Prey populations', fontsize=11)
ax.legend(fontsize=9)

# Phase plane
ax = axes[1]
ax.plot(Xs5, Ys5, color='steelblue', lw=1.8)
ax.plot(Xs5[0], Ys5[0], 'D', color='black',   ms=8, zorder=5, label='IC $(5,2)$')
ax.plot(8, 5,             'o', color='seagreen', ms=10, zorder=6,
        label='Equil. $(8,5)$')
ax.set_xlabel(r'Prey $x$', fontsize=13)
ax.set_ylabel(r'Predator $y$', fontsize=13)
ax.set_title('Phase portrait', fontsize=11)
ax.legend(fontsize=9)

plt.suptitle('Example 5: Lotka–Volterra predator–prey system', fontsize=11)
plt.tight_layout()
plt.show()

f6 = lambda t, x, y: y
g6 = lambda t, x, y: -0.3*y - np.sin(x)

x0_6 = 2*np.pi/3
ts6, Xs6, Ys6 = rk4_system(f6, g6, 0.0, x0_6, 0.0, 20.0, h=0.05)

# Linearised solution
omega_d = np.sqrt(1 - 0.15**2)
A = x0_6
B = 0.15*A / omega_d
t_lin = np.linspace(0, 20, 800)
theta_lin = np.exp(-0.15*t_lin) * (A*np.cos(omega_d*t_lin) + B*np.sin(omega_d*t_lin))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(ts6, Xs6, color='steelblue',  lw=1.8, label=r'Nonlinear RK4 $\theta(t)$')
ax.plot(t_lin, theta_lin, '--', color='darkorange', lw=1.8,
        label='Linear approx.')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel(r'$t$ (s)', fontsize=13)
ax.set_ylabel(r'$\theta$ (rad)', fontsize=13)
ax.set_title(r'Damped pendulum: $\theta(0)=2\pi/3$, $\theta\'(0)=0$', fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(Xs6, Ys6, color='steelblue', lw=1.5)
ax.plot(x0_6, 0, 'D', color='black',   ms=8, zorder=5,
        label=r'IC $(2\pi/3, 0)$')
ax.plot(0,    0, 'o', color='seagreen', ms=9, zorder=6,
        label='Equilibrium $(0,0)$')
ax.set_xlabel(r'$\theta$ (rad)', fontsize=13)
ax.set_ylabel(r"$\theta'$ (rad/s)", fontsize=13)
ax.set_title('Phase portrait (damped spiral)', fontsize=11)
ax.legend(fontsize=9)

plt.suptitle(r"Example 6: $\theta''+0.3\theta'+\sin\theta=0$", fontsize=11)
plt.tight_layout()
plt.show()

f7 = lambda t, x, y: y
g7 = lambda t, x, y: -4*x - 2*y + 4*np.exp(-t)*np.cos(2*t)

ts7, Xs7, Ys7 = rk4_system(f7, g7, 0.0, 0.0, 0.0, 10.0, h=0.05)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(ts7, Xs7, color='steelblue',  lw=1.8, label=r'Charge $q(t)$')
ax.plot(ts7, Ys7, color='darkorange', lw=1.8, label=r'Current $I(t)=q\'(t)$')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel(r'$t$ (s)', fontsize=13)
ax.set_ylabel(r'$q$ (C) or $I$ (A)', fontsize=13)
ax.set_title(r'RLC circuit: $q(0)=0$, $I(0)=0$', fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(Xs7, Ys7, color='steelblue', lw=1.5)
ax.plot(Xs7[0], Ys7[0], 'D', color='black',   ms=8, zorder=5, label='IC $(0,0)$')
ax.plot(0,      0,       'o', color='seagreen', ms=9, zorder=6,
        label='Equilibrium $(0,0)$')
ax.set_xlabel(r'$q$ (C)', fontsize=13)
ax.set_ylabel(r'$I$ (A)', fontsize=13)
ax.set_title('Phase portrait', fontsize=11)
ax.legend(fontsize=9)

plt.suptitle(r"Example 7: $q''+2q'+4q=4e^{-t}\cos2t$", fontsize=11)
plt.tight_layout()
plt.show()

from scipy.integrate import solve_ivp as sp_ivp

def sys8_scipy(t, s):
    x, y = s
    return [y - x**2, -x]

ref = sp_ivp(sys8_scipy, [0, 3], [0.0, 1.0],
             method='RK45', rtol=1e-11, atol=1e-13, dense_output=True)
x_ref, y_ref = ref.sol(3.0)
print(f"High-accuracy reference: x(3) ≈ {x_ref:.10f},  y(3) ≈ {y_ref:.10f}")

def euler_system(f, g, t0, x0, y0, T, h):
    """Euler method for 2x2 system."""
    t, x, y = t0, x0, y0
    while t < T - 1e-12:
        xn = x + h*f(t, x, y)
        yn = y + h*g(t, x, y)
        x, y, t = xn, yn, t + h
    return x, y

def heun_system(f, g, t0, x0, y0, T, h):
    """Modified Euler (Heun) for 2x2 system."""
    t, x, y = t0, x0, y0
    while t < T - 1e-12:
        k1x = f(t, x, y);      k1y = g(t, x, y)
        k2x = f(t+h, x+h*k1x, y+h*k1y)
        k2y = g(t+h, x+h*k1x, y+h*k1y)
        x = x + h/2*(k1x + k2x)
        y = y + h/2*(k1y + k2y)
        t += h
    return x, y

f8 = lambda t, x, y: y - x**2
g8 = lambda t, x, y: -x
T8 = 3.0

print(f"{'h':>6}  {'Euler |err_x|':>15}  {'Heun |err_x|':>14}  {'RK4 |err_x|':>13}")
print("-" * 55)
for h in [0.5, 0.25, 0.1, 0.05]:
    xe, ye   = euler_system(f8, g8, 0, 0, 1, T8, h)
    xm, ym   = heun_system(f8, g8, 0, 0, 1, T8, h)
    _, Xrk, _ = rk4_system(f8, g8, 0.0, 0.0, 1.0, T8, h)
    Xrk_T = Xrk[-1]
    print(f"{h:>6.2f}  {abs(xe-x_ref):>15.2e}  {abs(xm-x_ref):>14.2e}  {abs(Xrk_T-x_ref):>13.2e}")

hs8 = np.array([0.5, 0.25, 0.1, 0.05, 0.025])
err_e8 = []; err_m8 = []; err_rk8 = []
for h in hs8:
    xe, _  = euler_system(f8, g8, 0, 0, 1, T8, h)
    xm, _  = heun_system(f8, g8, 0, 0, 1, T8, h)
    _, Xrk, _ = rk4_system(f8, g8, 0.0, 0.0, 1.0, T8, h)
    err_e8.append(abs(xe - x_ref))
    err_m8.append(abs(xm - x_ref))
    err_rk8.append(abs(Xrk[-1] - x_ref))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.loglog(hs8, err_e8,  'o-', color='tomato',    lw=1.8, ms=7, label='Euler')
ax.loglog(hs8, err_m8,  's-', color='darkorange', lw=1.8, ms=7, label='Mod. Euler')
ax.loglog(hs8, err_rk8, '^-', color='steelblue',  lw=1.8, ms=7, label='RK4')
ax.loglog(hs8, 2.0*hs8,       'k:',  lw=1.2, label=r'$O(h)$')
ax.loglog(hs8, 1.5*hs8**2,    'k--', lw=1.2, label=r'$O(h^2)$')
ax.loglog(hs8, 0.8*hs8**4,    'k-.', lw=1.2, label=r'$O(h^4)$')
ax.set_xlabel(r'Step size $h$', fontsize=13)
ax.set_ylabel(r'$|X_N - x_{\rm ref}(3)|$', fontsize=13)
ax.set_title('Error in $x(3)$ for the nonlinear system', fontsize=11)
ax.legend(fontsize=8)

ax = axes[1]
t_dense = np.linspace(0, T8, 500)
xy_dense = ref.sol(t_dense)
ax.plot(xy_dense[0], xy_dense[1], color='steelblue', lw=1.8)
ax.plot(0, 1, 'D', color='black',   ms=8, zorder=5, label='IC $(0,1)$')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title('Reference phase portrait', fontsize=11)
ax.legend(fontsize=9)

plt.suptitle(r"Example 8: $x'=y-x^2$, $y'=-x$", fontsize=11)
plt.tight_layout()
plt.show()