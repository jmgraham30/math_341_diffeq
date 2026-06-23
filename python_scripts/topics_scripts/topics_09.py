import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jn_zeros
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

x = sym.Symbol('x', positive=True)
nu = sym.Symbol('nu', nonnegative=True)

# ---- Symbolic Bessel functions ----
J0_sym = sym.besselj(0, x)
J1_sym = sym.besselj(1, x)
J2_sym = sym.besselj(2, x)

print("Bessel functions of the first kind (SymPy):")
display(Math(r'J_0(x) = ' + sym.latex(J0_sym)))
display(Math(r'J_1(x) = ' + sym.latex(J1_sym)))

# Series expansion of J_0 about x=0
J0_series = sym.series(J0_sym, x, 0, n=10)
print("\nPower series for J_0(x) about x = 0 (first 5 terms):")
display(Math(r'J_0(x) = ' + sym.latex(J0_series)))

# ---- Verify the ODE: x^2 y'' + x y' + (x^2 - nu^2) y = 0 ----
print("Verifying that J_0 satisfies Bessel's equation of order 0:")
y = J0_sym
ode_lhs = x**2 * sym.diff(y, x, 2) + x * sym.diff(y, x) + x**2 * y
residual = sym.simplify(ode_lhs)
display(Math(r"x^2 J_0'' + x J_0' + x^2 J_0 = " + sym.latex(residual)))

print("\nVerifying that J_1 satisfies Bessel's equation of order 1:")
y1 = J1_sym
ode_lhs1 = x**2 * sym.diff(y1, x, 2) + x * sym.diff(y1, x) + (x**2 - 1) * y1
residual1 = sym.simplify(ode_lhs1)
display(Math(r"x^2 J_1'' + x J_1' + (x^2-1) J_1 = " + sym.latex(residual1)))

Y0_sym = sym.bessely(0, x)
Y1_sym = sym.bessely(1, x)

print("Bessel functions of the second kind (SymPy):")
display(Math(r'Y_0(x) = ' + sym.latex(Y0_sym)))
display(Math(r'Y_1(x) = ' + sym.latex(Y1_sym)))

# Verify Y_0 satisfies Bessel's equation of order 0
y_Y0 = Y0_sym
ode_Y0 = x**2 * sym.diff(y_Y0, x, 2) + x * sym.diff(y_Y0, x) + x**2 * y_Y0
display(Math(r"x^2 Y_0'' + x Y_0' + x^2 Y_0 = "
             + sym.latex(sym.simplify(ode_Y0))))

# Use SymPy lambdify to convert symbolic Bessel functions to fast numerical callables.
# We supply a custom module dictionary so that lambdify maps SymPy's besselj/bessely
# to scipy.special.jv/yv, which numpy alone does not provide.
x_sym = sym.Symbol('x', positive=True)

from scipy.special import jv as _jv, yv as _yv
bessel_modules = [{'besselj': _jv, 'bessely': _yv}, 'numpy']

orders = [0, 1, 2, 3]
colors = plt.cm.tab10(np.arange(len(orders)))

x_vals = np.linspace(0.01, 20, 1000)

# Build lambdified functions from SymPy
J_funcs = [sym.lambdify(x_sym, sym.besselj(n, x_sym), bessel_modules) for n in orders]
Y_funcs = [sym.lambdify(x_sym, sym.bessely(n, x_sym), bessel_modules) for n in orders]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for n, color, Jf, Yf in zip(orders, colors, J_funcs, Y_funcs):
    axes[0].plot(x_vals, Jf(x_vals), color=color, lw=2, label=f'$J_{n}(x)$')
    y_vals = Yf(x_vals)
    # Mask values below display range to keep the plot clean near x=0
    y_vals_masked = np.where(y_vals < -1.5, np.nan, y_vals)
    axes[1].plot(x_vals, y_vals_masked, color=color, lw=2, label=f'$Y_{n}(x)$')

for ax in axes:
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('$x$')
    ax.legend(fontsize=9)
    ax.set_ylim(-1.5, 1.2)

axes[0].set_title('Bessel functions of the first kind $J_\\nu(x)$')
axes[0].set_ylabel('$J_\\nu(x)$')
axes[1].set_title('Bessel functions of the second kind $Y_\\nu(x)$')
axes[1].set_ylabel('$Y_\\nu(x)$')

plt.tight_layout()
plt.show()

x_check = np.linspace(0.5, 15, 500)

J0_sp = sym.lambdify(x_sym, sym.besselj(0, x_sym), bessel_modules)
J1_sp = sym.lambdify(x_sym, sym.besselj(1, x_sym), bessel_modules)

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for col, (order, Jf_sym, label) in enumerate([
        (0, J0_sp, '$J_0$'),
        (1, J1_sp, '$J_1$')]):

    y_sym   = Jf_sym(x_check)
    y_scipy = jv(order, x_check)

    axes[0, col].plot(x_check, y_sym,   lw=2.5, color='steelblue',
                      label='SymPy (lambdified)')
    axes[0, col].plot(x_check, y_scipy, lw=1.2, color='crimson',
                      ls='--', label='SciPy `jv`')
    axes[0, col].set_title(f'{label}(x): SymPy vs SciPy')
    axes[0, col].set_ylabel(f'{label}(x)')
    axes[0, col].legend(fontsize=9)
    axes[0, col].axhline(0, color='gray', lw=0.7, ls=':')

    diff = np.abs(y_sym - y_scipy)
    axes[1, col].semilogy(x_check, diff + 1e-18, color='seagreen', lw=1.8)
    axes[1, col].set_title(f'Absolute difference |SymPy $-$ SciPy| for {label}')
    axes[1, col].set_xlabel('$x$')
    axes[1, col].set_ylabel('Absolute error')

plt.tight_layout()
plt.show()

x = sym.Symbol('x', positive=True)

for nu_val in [1, 2, sym.Rational(3, 2)]:
    Jnm1 = sym.besselj(nu_val - 1, x)
    Jn   = sym.besselj(nu_val,     x)
    Jnp1 = sym.besselj(nu_val + 1, x)

    lhs_R1 = sym.simplify(Jnm1 + Jnp1 - 2*nu_val/x * Jn)
    lhs_R2 = sym.simplify(Jnm1 - Jnp1 - 2*sym.diff(Jn, x))

    print(f"nu = {nu_val}:")
    display(Math(r"J_{\nu-1}+J_{\nu+1}-\tfrac{2\nu}{x}J_\nu = "
                 + sym.latex(lhs_R1)))
    display(Math(r"J_{\nu-1}-J_{\nu+1}-2J_\nu' = "
                 + sym.latex(lhs_R2)))
    print()

print("First five positive zeros of J_nu:\n")
print(f"{'nu':>4}  {'j(nu,1)':>10}  {'j(nu,2)':>10}  {'j(nu,3)':>10}  "
      f"{'j(nu,4)':>10}  {'j(nu,5)':>10}")
print("-" * 60)
for n in [0, 1, 2, 3]:
    z = jn_zeros(n, 5)
    print(f"  {n:>2}  " + "  ".join(f"{zi:10.6f}" for zi in z))

x_vals = np.linspace(0.01, 20, 2000)
J_sym_funcs = [sym.lambdify(x_sym, sym.besselj(n, x_sym), bessel_modules)
               for n in range(4)]

fig, ax = plt.subplots(figsize=(11, 4.5))

for n, color, Jf in zip(range(4), plt.cm.tab10(np.arange(4)), J_sym_funcs):
    ax.plot(x_vals, Jf(x_vals), color=color, lw=2, label=f'$J_{n}(x)$')
    zeros = jn_zeros(n, 5)
    ax.scatter(zeros, np.zeros_like(zeros), color=color, s=55,
               zorder=5, edgecolors='black', linewidths=0.6)

ax.axhline(0, color='gray', lw=0.9, ls='--')
ax.set_xlabel('$x$')
ax.set_ylabel('$J_\\nu(x)$')
ax.set_title('Zeros of Bessel functions $J_0, J_1, J_2, J_3$ (first five each)')
ax.set_ylim(-0.55, 1.1)
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.show()

x_a = np.linspace(1, 30, 1000)
J0_exact = jv(0, x_a)
J0_asymp = np.sqrt(2 / (np.pi * x_a)) * np.cos(x_a - np.pi / 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x_a, J0_exact, color='steelblue', lw=2.2, label='$J_0(x)$ (exact)')
axes[0].plot(x_a, J0_asymp, color='darkorange', lw=1.5, ls='--',
             label=r'$\sqrt{2/\pi x}\cos(x-\pi/4)$ (asymptotic)')
axes[0].axhline(0, color='gray', lw=0.7, ls=':')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$J_0(x)$')
axes[0].set_title('$J_0(x)$ vs large-$x$ asymptotic approximation')
axes[0].legend(fontsize=9)

rel_err = np.abs(J0_exact - J0_asymp) / (np.abs(J0_exact) + 1e-12)
axes[1].semilogy(x_a, rel_err, color='seagreen', lw=2)
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('Relative error')
axes[1].set_title('Relative error of asymptotic approximation for $J_0$')

plt.tight_layout()
plt.show()

# Numerical verification of orthogonality for J_0
from scipy.integrate import quad

nu_orth = 0
print(f"Numerical verification of orthogonality for J_{nu_orth}:\n")
print(f"  (m, n)     integral        expected")
print("  " + "-"*45)

for m in range(1, 4):
    for n in range(1, 4):
        zm = jn_zeros(nu_orth, m)[-1]
        zn = jn_zeros(nu_orth, n)[-1]

        integrand = lambda t: t * jv(nu_orth, zm * t) * jv(nu_orth, zn * t)
        result, _ = quad(integrand, 0, 1, limit=200)

        if m == n:
            expected = 0.5 * jv(nu_orth + 1, zm)**2
        else:
            expected = 0.0

        print(f"  ({m}, {n})       {result:+.6f}       {expected:+.6f}")

R = 1.0   # drum radius
zeros_J0 = jn_zeros(0, 3)   # j_{0,1}, j_{0,2}, j_{0,3}

r_vals = np.linspace(0, R, 500)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
colors_mode = ['steelblue', 'darkorange', 'seagreen']

# Left panel: radial profiles
ax0 = axes[0]
for m, (zm, color) in enumerate(zip(zeros_J0, colors_mode), start=1):
    ax0.plot(r_vals, jv(0, zm * r_vals / R), color=color, lw=2.2,
             label=f'$m={m}$, $j_{{0,{m}}}={zm:.3f}$')
ax0.axhline(0, color='gray', lw=0.7, ls='--')
ax0.set_xlabel('$r/R$'); ax0.set_ylabel('$J_0(j_{0,m}\\,r/R)$')
ax0.set_title('Radial profiles')
ax0.legend(fontsize=8)

# 2D mode shape plots
theta = np.linspace(0, 2*np.pi, 300)
r_2d  = np.linspace(0, R, 300)
R_grid, T_grid = np.meshgrid(r_2d, theta)
X = R_grid * np.cos(T_grid)
Y = R_grid * np.sin(T_grid)

for ax, (zm, color, m) in zip(axes[1:], zip(zeros_J0, colors_mode, range(1, 4))):
    Z = jv(0, zm * R_grid / R)
    cp = ax.contourf(X, Y, Z, levels=40, cmap='RdBu_r')
    # Draw nodal circles (zero-crossings of J_0 inside the drum)
    inner_zeros = zeros_J0[:m-1]
    for iz in inner_zeros:
        r_node = iz / zm * R
        circle = plt.Circle((0, 0), r_node, color='black', fill=False,
                             lw=1.2, ls='--')
        ax.add_patch(circle)
    ax.set_aspect('equal'); ax.set_axis_off()
    ax.set_title(f'Mode $m={m}$\n$\\omega_{{0,{m}}} = c\\cdot{zm:.3f}/R$', fontsize=9)

plt.tight_layout()
plt.show()

x = sym.Symbol('x', positive=True)

print("Verifying derivative formulas for Bessel functions:\n")
for nu_val in [0, 1, 2]:
    Jnu  = sym.besselj(nu_val, x)
    Jnup1 = sym.besselj(nu_val + 1, x)
    Jnum1 = sym.besselj(nu_val - 1, x)

    # d/dx [x^nu J_nu] = x^nu J_{nu-1}
    lhs1 = sym.diff(x**nu_val * Jnu, x)
    rhs1 = x**nu_val * Jnum1
    res1 = sym.simplify(lhs1 - rhs1)
    display(Math(
        rf"\frac{{d}}{{dx}}\left[x^{nu_val} J_{nu_val}(x)\right] - "
        rf"x^{nu_val} J_{{{nu_val-1}}}(x) = " + sym.latex(res1)
    ))

    # d/dx [x^{-nu} J_nu] = -x^{-nu} J_{nu+1}
    lhs2 = sym.diff(x**(-nu_val) * Jnu, x)
    rhs2 = -x**(-nu_val) * Jnup1
    res2 = sym.simplify(lhs2 - rhs2)
    display(Math(
        rf"\frac{{d}}{{dx}}\left[x^{{-{nu_val}}} J_{nu_val}(x)\right] + "
        rf"x^{{-{nu_val}}} J_{{{nu_val+1}}}(x) = " + sym.latex(res2)
    ))
    print()