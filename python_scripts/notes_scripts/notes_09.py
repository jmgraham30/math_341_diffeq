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


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-hamiltonian
#| fig-cap: "Hamiltonian view of the SHO $x'=y$, $y'=-4x$ ($m=1$, $k=4$). Left: phase-plane orbits are level curves of $H = y^2/2 + 2x^2$ (ellipses). The vector field $(y,-4x)$ is tangent to every orbit — confirming $H$ is conserved. Right: component plots $x(t)=\\cos 2t$ and $y(t)=-2\\sin 2t$ for the initial condition $(x_0,y_0)=(1,0)$."

t_plot = np.linspace(0, 2*np.pi, 400)
x_sol  = np.cos(2*t_plot)
y_sol  = -2*np.sin(2*t_plot)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Phase plane — level curves of H
x_grid = np.linspace(-2.2, 2.2, 300)
y_grid = np.linspace(-4.5, 4.5, 300)
X_g, Y_g = np.meshgrid(x_grid, y_grid)
H_grid   = Y_g**2/2 + 2*X_g**2
levels   = [0.5, 1.0, 2.0, 4.0, 6.0]
cs = axes[0].contour(X_g, Y_g, H_grid, levels=levels, colors=plt.cm.viridis(np.linspace(0.1,0.85,5)))
axes[0].clabel(cs, fmt={v: f'$H={v}$' for v in levels}, fontsize=8)

# Vector field
Xq = np.linspace(-2,2,12); Yq = np.linspace(-4,4,12)
Xqm, Yqm = np.meshgrid(Xq, Yq)
dX, dY = Yqm, -4*Xqm
speed = np.sqrt(dX**2 + dY**2) + 1e-8
axes[0].quiver(Xqm, Yqm, dX/speed, dY/speed, alpha=0.35, color='gray', scale=25, width=0.003)
axes[0].plot(x_sol, y_sol, color='steelblue', lw=2.5, label='$(x_0,y_0)=(1,0)$')
axes[0].plot(1, 0, 'ko', markersize=8, zorder=5, label='$t=0$')
axes[0].set_xlabel('$x$ (position)'); axes[0].set_ylabel('$y$ (momentum)')
axes[0].set_title(r'Phase plane: level curves of $H=y^2/2+2x^2$')
axes[0].legend(fontsize=9); axes[0].set_aspect('equal')

# Component plots
axes[1].plot(t_plot, x_sol, color='steelblue', lw=2, label='$x(t)=\\cos 2t$')
axes[1].plot(t_plot, y_sol, color='crimson',   lw=2, label='$y(t)=-2\\sin 2t$')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t),\\ y(t)$')
axes[1].set_title('Component plots (time series)')
axes[1].legend(fontsize=9)
axes[1].set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])
axes[1].set_xticklabels(['$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])

plt.suptitle(r"Simple Harmonic Oscillator as a Hamiltonian System: $H=y^2/2+2x^2$", fontsize=11)
plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-two-systems
#| fig-cap: "Phase-plane portraits and component plots for two systems solved by elimination. Left: $x'=y$, $y'=-4x$ — elliptic orbits (pure imaginary eigenvalues, neutral oscillation). Right: $x'=y$, $y'=4x$ — hyperbolic orbits (real eigenvalues, saddle point), with the IVP solution $c_1=1/4$, $c_2=-1/4$ highlighted."

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
t_long = np.linspace(0, 2*np.pi, 400)
t_saddle = np.linspace(0, 1.2, 300)

# --- System 1: x'=y, y'=-4x (ellipses) ---
def sys1(t, s): return [s[1], -4*s[0]]
for ic, color in [((1,0),'steelblue'),((0.5,0),'darkorange'),((1.5,0),'seagreen')]:
    sol = solve_ivp(sys1,(0,2*np.pi),list(ic),dense_output=True,max_step=0.01)
    t_p = np.linspace(0,2*np.pi,400)
    axes[0].plot(sol.sol(t_p)[0], sol.sol(t_p)[1], color=color, lw=2)

# Nullclines
axes[0].axhline(0, color='crimson',   lw=1.5, ls='--', label='$x$-nullcline: $y=0$')
axes[0].axvline(0, color='seagreen',  lw=1.5, ls='--', label='$y$-nullcline: $x=0$')
axes[0].set_xlim(-2,2); axes[0].set_ylim(-4.5,4.5)
axes[0].set_aspect('equal')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
axes[0].set_title(r"$x'=y$, $y'=-4x$: elliptic orbits (center)")
axes[0].legend(fontsize=8)

# --- System 2: x'=y, y'=4x (saddle) ---
def sys2(t, s): return [s[1], 4*s[0]]
# Multiple ICs
for ic, color in [((0.1,0.3),'steelblue'),((0.5,1.1),'darkorange'),
                  ((-0.1,0.3),'seagreen'),((-0.5,1.1),'purple')]:
    for ic, color in [((0.1,0.3),'steelblue'),((0.5,1.1),'darkorange'),
                  ((-0.1,0.3),'seagreen'),((-0.5,1.1),'purple')]:
        for sign in [1, -1]:
          ic_s = (ic[0]*sign, ic[1]*sign)

          def stop_event(t, y):
              return 5.0 - np.max(np.abs(y))
          stop_event.terminal  = True
          stop_event.direction = -1

          sol = solve_ivp(sys2, (0, 1.5), list(ic_s),
                        dense_output=True, max_step=0.005,
                        events=stop_event)
          t_p = np.linspace(0, sol.t[-1], 200)
          axes[1].plot(sol.sol(t_p)[0], sol.sol(t_p)[1], color=color, lw=1.5)

# IVP solution c1=1/4, c2=-1/4
t_ivp = np.linspace(0,1.2,200)
x_ivp = 0.25*np.exp(2*t_ivp) - 0.25*np.exp(-2*t_ivp)
y_ivp = 0.5*np.exp(2*t_ivp) + 0.5*np.exp(-2*t_ivp)
axes[1].plot(x_ivp, y_ivp, 'k-', lw=3, label='IVP: $x(0)=0$, $y(0)=1$')
axes[1].plot(0, 1, 'ko', markersize=8, zorder=5)

# Eigenvectors (unstable/stable manifolds)
t_ev = np.linspace(-2,2,100)
axes[1].plot(t_ev, 2*t_ev, 'r--', lw=1.5, label='Eigenvector $\\lambda=2$')
axes[1].plot(t_ev,-2*t_ev, 'm--', lw=1.5, label='Eigenvector $\\lambda=-2$')
axes[1].set_xlim(-2,2); axes[1].set_ylim(-4,4)
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
axes[1].set_title(r"$x'=y$, $y'=4x$: hyperbolic orbits (saddle)")
axes[1].legend(fontsize=7.5)

plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"

print("2nd-order companion: x'' + (b/a)x' + (c/a)x = 0, y = x'")
a_v, b_v, c_v = 1, 3, 2  # x'' + 3x' + 2x = 0
A_2nd = sym.Matrix([[0,1],[-c_v/a_v, -b_v/a_v]])
lam = sym.Symbol('lambda')
char2 = (lam*sym.eye(2) - A_2nd).det()
print(f"  A = {A_2nd.tolist()}")
print(f"  char poly = {sym.expand(char2)}")
print(f"  eigenvalues = {sym.solve(char2, lam)}")

print("\n3rd-order companion: x''' + 2x'' - x' + 3x = 0")
A_3rd = sym.Matrix([[0,1,0],[0,0,1],[-3,1,-2]])
char3 = (lam*sym.eye(3) - A_3rd).det()
print(f"  char poly = {sym.expand(char3)}")


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-compartment
#| fig-cap: "Compartmental pesticide model with $\\alpha=0.3$, $\\beta=0.5$, $\\gamma=0.1$, initial conditions $x(0)=0$, $y(0)=10$ (pesticide initially only in soil). The pesticide gradually transfers into the crop, reaches a peak, and both compartments eventually decay to zero as the pesticide degrades."

alpha_v, beta_v, gamma_v = 0.3, 0.5, 0.1

def comp_sys(t, state):
    x_c, y_c = state
    return [-beta_v*x_c + alpha_v*y_c,
             beta_v*x_c - (alpha_v+gamma_v)*y_c]

t_eval = np.linspace(0, 25, 500)
sol_c = solve_ivp(comp_sys, (0,25), [0.0, 10.0], t_eval=t_eval, max_step=0.05)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(t_eval, sol_c.y[0], color='steelblue', lw=2, label='$x(t)$ — crop')
axes[0].plot(t_eval, sol_c.y[1], color='seagreen',  lw=2, label='$y(t)$ — soil')
axes[0].set_xlabel('$t$ (days)'); axes[0].set_ylabel('Pesticide amount')
axes[0].set_title('Component plots')
axes[0].legend(fontsize=9)

axes[1].plot(sol_c.y[0], sol_c.y[1], color='darkorange', lw=2.5)
axes[1].plot(0, 10, 'ko', markersize=8, label='$t=0$: $(0,10)$', zorder=5)
axes[1].set_xlabel('$x$ (crop)'); axes[1].set_ylabel('$y$ (soil)')
axes[1].set_title('Phase-plane orbit')
axes[1].legend(fontsize=9)

plt.suptitle(r'Compartmental Model: $dx/dt=-\beta x+\alpha y$, $dy/dt=\beta x-(\alpha+\gamma)y$', fontsize=11)
plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"

print("=== Examples 4.16 and 4.17 (Logan) ===")
A = sym.Matrix([[1,2],[3,-4]]); B = sym.Matrix([[0,-2],[7,-4]])
x_v = sym.Matrix([-4,6]);      y_v = sym.Matrix([5,1])
print(f"A+B =\n{A+B}")
print(f"-3B =\n{-3*B}")
print(f"5x = {(5*x_v).T}")
print(f"x+2y = {(x_v+2*y_v).T}")

A2 = sym.Matrix([[2,3],[-1,0]]); B2 = sym.Matrix([[1,4],[5,2]])
print(f"\nAB =\n{A2*B2}")
print(f"A^2 =\n{A2**2}")

print("\n=== Non-commutativity: AB ≠ BA ===")
Anc = sym.Matrix([[1,1],[0,0]]); Bnc = sym.Matrix([[0,0],[1,1]])
print(f"A =\n{Anc}\nB =\n{Bnc}")
print(f"AB =\n{Anc*Bnc}")
print(f"BA =\n{Bnc*Anc}")
print(f"AB == BA? {Anc*Bnc == Bnc*Anc}")

print("\n=== Example 4.18 ===")
A18 = sym.Matrix([[1,2],[4,3]])
print(f"det(A) = {A18.det()},  A^(-1) =\n{A18.inv()}")
print(f"A*A^(-1) = {A18*A18.inv()}")


#| code-fold: true
#| code-summary: "Show the code"

print("=== Example 4.20 (Logan): singular matrix ===")
A20 = sym.Matrix([[4,1],[8,2]])
print(f"det(A) = {A20.det()} (singular)")
print(f"Nullspace: {A20.nullspace()}")
print(f"All solutions: x = alpha*(1,-4)^T for alpha in R")

print("\n=== Example 4.22 (Logan): Cramer's rule ===")
A22 = sym.Matrix([[2,-3],[4,-1]]); b22 = sym.Matrix([4,-1])
print(f"det(A) = {A22.det()}")
A_x = sym.Matrix([[4,-3],[-1,-1]]); A_y = sym.Matrix([[2,4],[4,-1]])
x_cr = A_x.det()/A22.det(); y_cr = A_y.det()/A22.det()
print(f"x = {x_cr}, y = {y_cr}")
print(f"Verify: 2x-3y = {2*x_cr - 3*y_cr}, 4x-y = {4*x_cr - y_cr}")


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-equilibria
#| fig-cap: "Equilibria of two systems. Left: $\\mathbf{x}'=A\\mathbf{x}$ with $A=\\bigl[\\begin{smallmatrix}0&1\\\\-4&0\\end{smallmatrix}\\bigr]$, $\\det A=4\\neq 0$ — unique isolated equilibrium at origin (stable center). Right: $\\mathbf{x}'=A\\mathbf{x}+\\mathbf{b}$ with $\\mathbf{b}=(-2,3)^T$ — the equilibrium shifts to $\\mathbf{x}^*=-A^{-1}\\mathbf{b}$."

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# System 1: x'=Ax, A=[[0,1],[-4,0]], unique eq at origin
def sys_center(t,s): return [s[1], -4*s[0]]
for ic, color in [((1,0),'steelblue'),((0.5,0),'darkorange'),((1.5,0),'seagreen'),
                  ((0,1),'crimson')]:
    sol = solve_ivp(sys_center,(0,2*np.pi),list(ic),dense_output=True,max_step=0.01)
    t_p = np.linspace(0,2*np.pi,400)
    axes[0].plot(sol.sol(t_p)[0],sol.sol(t_p)[1],color=color,lw=2)
axes[0].plot(0,0,'ko',markersize=10,zorder=5,label='Eq: $(0,0)$')
axes[0].axhline(0,color='crimson',lw=1,ls='--',alpha=0.5)
axes[0].axvline(0,color='seagreen',lw=1,ls='--',alpha=0.5)
axes[0].set_aspect('equal'); axes[0].set_xlim(-2.2,2.2); axes[0].set_ylim(-4.5,4.5)
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
axes[0].set_title(r"$\mathbf{x}'=A\mathbf{x}$: unique eq at origin ($\det A=4$)")
axes[0].legend(fontsize=9)

# System 2: x'=Ax+b, equilibrium shifted
A_nh = np.array([[0,1],[-4,0]]); b_nh = np.array([-2.0,3.0])
x_star = -np.linalg.inv(A_nh)@b_nh
print(f"x* = -A^(-1)b = {x_star}")

def sys_nh(t,s):
    return [A_nh[0,0]*s[0]+A_nh[0,1]*s[1]+b_nh[0],
            A_nh[1,0]*s[0]+A_nh[1,1]*s[1]+b_nh[1]]
for ic_offset, color in [(np.array([1,0]),'steelblue'),(np.array([0.5,0]),'darkorange'),
                          (np.array([-1,0]),'seagreen'),(np.array([0,1]),'crimson')]:
    ic = x_star + ic_offset
    sol = solve_ivp(sys_nh,(0,2*np.pi),list(ic),dense_output=True,max_step=0.01)
    t_p = np.linspace(0,2*np.pi,400)
    axes[1].plot(sol.sol(t_p)[0],sol.sol(t_p)[1],color=color,lw=2)
axes[1].plot(*x_star,'k*',markersize=14,zorder=5,label=f'Eq: $\\mathbf{{x}}^*=({x_star[0]:.2f},{x_star[1]:.2f})$')
axes[1].set_aspect('equal')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
axes[1].set_title(r"$\mathbf{x}'=A\mathbf{x}+\mathbf{b}$: eq shifted to $\mathbf{x}^*=-A^{-1}\mathbf{b}$")
axes[1].legend(fontsize=9)

plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m,'__version__',None)))
