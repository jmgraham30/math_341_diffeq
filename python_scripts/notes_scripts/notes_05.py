import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

t_plot = np.linspace(0, 1.4, 400)

# Particular solution (no constants)
xp1 = np.cos(t_plot)*np.log(np.cos(t_plot)) + t_plot*np.sin(t_plot)
# IVP: xp(0)=0, xp'(0)=0, so x(0)=C1=1, x'(0)=C2=0
xh1 = np.cos(t_plot)          # C1=1, C2=0
x1_full = xh1 + xp1

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_plot, x1_full, color='steelblue', lw=2.5, label='Full solution $x(t)$')
ax.plot(t_plot, xh1,     color='darkorange', lw=1.8, ls='--', label='Homogeneous $x_h=\\cos t$')
ax.plot(t_plot, xp1,     color='seagreen',   lw=1.8, ls='-.',
        label='Particular $x_p=\\cos t\\ln|\\cos t|+t\\sin t$')

# Numerical check
def ode1(t, y):
    return [y[1], -y[0] + 1/np.cos(t)]
sol1 = solve_ivp(ode1, (0, 1.4), [1.0, 0.0], dense_output=True, max_step=0.005)
t_dots = np.linspace(0, 1.4, 18)
ax.plot(t_dots, sol1.sol(t_dots)[0], 'ro', markersize=5, label='Numerical check')

ax.axvline(np.pi/2, color='gray', ls=':', lw=1.2, label=r'Singularity $t=\pi/2$')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x''+x=\sec t$, $x(0)=1$, $x'(0)=0$")
ax.legend(fontsize=8.5); ax.set_ylim(-0.5, 2.5)
plt.tight_layout(); plt.show()

# SymPy verification
t_sym = sym.Symbol('t', positive=True)
x_sym = sym.Function('x')
ode2_sym = sym.Eq(x_sym(t_sym).diff(t_sym,2) - 2*x_sym(t_sym).diff(t_sym) + x_sym(t_sym),
                  sym.exp(t_sym)/t_sym)
sol2_sym = sym.dsolve(ode2_sym, x_sym(t_sym))
print("SymPy solution:")
display(Math(sym.latex(sol2_sym)))

t_plot2 = np.linspace(1.0, 2.8, 400)

# IVP: x(1)=0, x'(1)=0
# Solving: C1 + C2 - 1 = 0 and C1 + 2*C2 + 1 = 0  =>  C1=2, C2=-1
C1_2, C2_2 = 2.0, -1.0
xp2  = t_plot2 * np.exp(t_plot2) * (np.log(t_plot2) - 1)
xh2  = C1_2*np.exp(t_plot2) + C2_2*t_plot2*np.exp(t_plot2)   # 2e^t - te^t
x2_full = xh2 + xp2

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_plot2, x2_full, color='steelblue', lw=2.5, label='Full solution $x(t)$')
ax.plot(t_plot2, xh2,     color='darkorange', lw=1.8, ls='--', label='$x_h = 2e^t - te^t$ (with $C_1=2$, $C_2=-1$)')
ax.plot(t_plot2, xp2,     color='seagreen',   lw=1.8, ls='-.', label='$x_p = te^t(\\ln t-1)$')

def ode2(t, y): return [y[1], 2*y[1] - y[0] + np.exp(t)/t]
sol2n = solve_ivp(ode2, (1.0, 2.8), [0.0, 0.0], dense_output=True, max_step=0.005)
t_dots2 = np.linspace(1.0, 2.8, 18)
ax.plot(t_dots2, sol2n.sol(t_dots2)[0], 'ro', markersize=5, label='Numerical check')

ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x''-2x'+x=e^t/t$, $x(1)=0$, $x'(1)=0$")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

# SymPy verification
t_sym = sym.Symbol('t', positive=True)
x_sym = sym.Function('x')
ode3_sym = sym.Eq(t_sym**2*x_sym(t_sym).diff(t_sym,2)
                  - 2*t_sym*x_sym(t_sym).diff(t_sym)
                  + 2*x_sym(t_sym),
                  t_sym**3*sym.sin(t_sym))
sol3_sym = sym.dsolve(ode3_sym, x_sym(t_sym))
print("SymPy solution:")
display(Math(sym.latex(sol3_sym)))

t_plot3 = np.linspace(np.pi, 3*np.pi, 500)

# IVP: x(pi) = pi^2, x'(pi) = 2*pi
# x = C1*t + C2*t^2 - t*sin(t)
# x(pi): C1*pi + C2*pi^2 - pi*0 = pi*(C1 + C2*pi) = pi^2 => C1 + C2*pi = pi => C1 = pi - C2*pi
# x'(t) = C1 + 2*C2*t - sin(t) - t*cos(t)
# x'(pi): C1 + 2*C2*pi - 0 - pi*(-1) = C1 + 2*C2*pi + pi = 2*pi
# => C1 + 2*C2*pi = pi; with C1 = pi - C2*pi: pi - C2*pi + 2*C2*pi = pi + C2*pi = pi => C2=0, C1=pi

xp3   = -t_plot3 * np.sin(t_plot3)
xh3   = np.pi * t_plot3                        # C1=pi, C2=0
x3_full = xh3 + xp3

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_plot3, x3_full, color='steelblue', lw=2.5, label='Full solution $x(t)$')
ax.plot(t_plot3, xh3,     color='darkorange', lw=1.8, ls='--', label='$x_h = \\pi t$ (with $C_1=\\pi$, $C_2=0$)')
ax.plot(t_plot3, xp3,     color='seagreen',   lw=1.8, ls='-.', label='$x_p = -t\\sin t$')

def ode3(t, y):
    return [y[1], (2*t*y[1] - 2*y[0] + t**3*np.sin(t))/t**2]
x0_3  = np.pi**2
xp0_3 = 2*np.pi
sol3n = solve_ivp(ode3, (np.pi, 3*np.pi), [x0_3, xp0_3], dense_output=True, max_step=0.01)
t_dots3 = np.linspace(np.pi, 3*np.pi, 20)
ax.plot(t_dots3, sol3n.sol(t_dots3)[0], 'ro', markersize=5, label='Numerical check')

ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"Euler eq: $t^2x''-2tx'+2x=t^3\sin t$, $x(\pi)=\pi^2$, $x'(\pi)=2\pi$")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

t_plot4 = np.linspace(0.1, np.pi - 0.05, 400)  # between singularities

xp4   = -t_plot4*np.cos(t_plot4) + np.sin(t_plot4)*np.log(np.sin(t_plot4))
# IVP: x(pi/2)=pi/2, x'(pi/2)=0
# x = C1*cos(t) + C2*sin(t) + xp
# xp(pi/2) = -pi/2*0 + 1*ln(1) = 0
# xp'(t) = -cos(t) + t*sin(t) + cos(t)*ln(sin(t)) + sin(t)*cos(t)/sin(t)
#         = -cos(t) + t*sin(t) + cos(t)*ln(sin(t)) + cos(t)
#         = t*sin(t) + cos(t)*ln(sin(t))
# xp'(pi/2) = pi/2*1 + 0*ln(1) = pi/2
# x(pi/2): C1*0 + C2*1 + 0 = pi/2 => C2 = pi/2
# x'(pi/2): -C1*0 + C2*0 + pi/2 = ... wait
# x'(t) = -C1*sin(t) + C2*cos(t) + xp'(t)
# x'(pi/2) = -C1*1 + C2*0 + pi/2 = 0 => C1 = pi/2
C1_4, C2_4 = np.pi/2, np.pi/2
xh4   = C1_4*np.cos(t_plot4) + C2_4*np.sin(t_plot4)
x4_full = xh4 + xp4

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_plot4, x4_full, color='steelblue', lw=2.5, label='Full solution $x(t)$')
ax.plot(t_plot4, xh4,     color='darkorange', lw=1.8, ls='--',
        label=f'$x_h = \\frac{{\\pi}}{{2}}\\cos t + \\frac{{\\pi}}{{2}}\\sin t$')
ax.plot(t_plot4, xp4,     color='seagreen',   lw=1.8, ls='-.', label='$x_p=-t\\cos t+\\sin t\\ln|\\sin t|$')

def ode4(t, y):
    return [y[1], -y[0] + 1/np.sin(t)]
sol4n = solve_ivp(ode4, (np.pi/2, np.pi-0.05), [np.pi/2, 0.0], dense_output=True, max_step=0.005)
t_dots4 = np.linspace(np.pi/2, np.pi-0.08, 10)
ax.plot(t_dots4, sol4n.sol(t_dots4)[0], 'ro', markersize=5, label='Numerical check')

ax.axhline(0, color='k', lw=0.5)
ax.axvline(np.pi, color='gray', ls=':', lw=1.2, label=r'Singularity at $t=\pi$')
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x''+x=\csc t$, $x(\pi/2)=\pi/2$, $x'(\pi/2)=0$, on $(0,\pi)$")
ax.legend(fontsize=8); ax.set_ylim(-1.5, 3)
plt.tight_layout(); plt.show()

t_sym = sym.Symbol('t', positive=True)
x_sym = sym.Function('x')

examples = {
    r"x''+x=\sec t":
        sym.Eq(x_sym(t_sym).diff(t_sym,2) + x_sym(t_sym),  sym.sec(t_sym)),
    r"x''-2x'+x=e^t/t":
        sym.Eq(x_sym(t_sym).diff(t_sym,2) - 2*x_sym(t_sym).diff(t_sym) + x_sym(t_sym),
               sym.exp(t_sym)/t_sym),
    r"t^2x''-2tx'+2x=t^3\sin t":
        sym.Eq(t_sym**2*x_sym(t_sym).diff(t_sym,2) - 2*t_sym*x_sym(t_sym).diff(t_sym)
               + 2*x_sym(t_sym),  t_sym**3*sym.sin(t_sym)),
    r"x''+x=\csc t":
        sym.Eq(x_sym(t_sym).diff(t_sym,2) + x_sym(t_sym),  sym.csc(t_sym)),
}

for label, ode in examples.items():
    print(f"${label}$")
    sol = sym.dsolve(ode, x_sym(t_sym))
    display(Math(r"x(t)=" + sym.latex(sol.rhs)))
    print()