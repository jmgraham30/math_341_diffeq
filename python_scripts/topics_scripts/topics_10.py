import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

g   = 9.81
A_t = 0.20      # m^2  (tank cross-section, ~50 cm diameter)
a_t = 1.0e-4    # m^2  (orifice, ~1.1 cm diameter hole)
Cd  = 0.6
h0  = 1.0       # m

k_t = Cd*a_t*np.sqrt(2*g)/A_t
T_t = 2*np.sqrt(h0)/k_t
print(f"k = {k_t:.6f} m^0.5/s,  drain time T = {T_t:.1f} s = {T_t/60:.2f} min")

def h_analytic(t):
    return np.maximum(np.sqrt(h0) - k_t*t/2, 0.0)**2

def tank_ode(t, h):
    return [-k_t*np.sqrt(max(h[0], 0.0))]

t_num = np.linspace(0, T_t, 400)
sol = solve_ivp(tank_ode, (0, T_t), [h0], t_eval=t_num, max_step=1.0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Physical solution vs. numerical check ───────────────────────
t_plot = np.linspace(0, T_t, 400)
axes[0].plot(t_plot/60, h_analytic(t_plot), color='steelblue', lw=2.5, label='Analytic $h(t)$')
axes[0].plot(sol.t[::20]/60, sol.y[0][::20], 'o', color='darkorange',
             ms=5, label='Numerical (solve\\_ivp)', zorder=5)
axes[0].axvline(T_t/60, color='crimson', ls=':', lw=1.5, label=f'$T={T_t/60:.1f}$ min')
axes[0].set_xlabel('Time (min)'); axes[0].set_ylabel('$h(t)$ (m)')
axes[0].set_title('Physical solution: tank empties at $t=T$')
axes[0].legend(fontsize=8.5); axes[0].set_ylim(-0.05, 1.05)

# ── Domain-of-validity issue ─────────────────────────────────
t_ext = np.linspace(0, 1.6*T_t, 400)
h_naive = (np.sqrt(h0) - k_t*t_ext/2)**2       # never clipped at 0
axes[1].plot(t_ext/60, h_naive, color='gray', ls='--', lw=2, label='Naive formula (unclipped)')
axes[1].plot(t_plot/60, h_analytic(t_plot), color='steelblue', lw=2.5, label='Physical solution')
axes[1].axvline(T_t/60, color='crimson', ls=':', lw=1.5, label=f'$T={T_t/60:.1f}$ min')
axes[1].set_xlabel('Time (min)'); axes[1].set_ylabel('$h(t)$ (m)')
axes[1].set_title('Formula continued past $T$: unphysical rebound')
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.show()

R_cone = 0.3
H_cone = 1.0
k0 = Cd*a_t*np.sqrt(2*g)

def h_cone(t):
    coef = np.pi*(R_cone/H_cone)**2*(2/5)
    val = np.maximum(H_cone**2.5 - (k0/coef)*t, 0.0)
    return val**(2/5)

T_cone = np.pi*(R_cone/H_cone)**2*(2/5)*H_cone**2.5/k0
print(f"Cylinder drain time: {T_t/60:.2f} min   |   Cone drain time: {T_cone/60:.2f} min")

t_plot2 = np.linspace(0, T_t, 400)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(t_plot2/60, h_analytic(t_plot2), color='steelblue', lw=2.5,
        label=f'Cylinder ($T={T_t/60:.1f}$ min)')
ax.plot(t_plot2/60, h_cone(t_plot2), color='darkorange', lw=2.5,
        label=f'Cone, apex down ($T={T_cone/60:.1f}$ min)')
ax.axvline(T_cone/60, color='darkorange', ls=':', lw=1)
ax.axvline(T_t/60, color='steelblue', ls=':', lw=1)
ax.set_xlabel('Time (min)'); ax.set_ylabel('$h(t)$ (m)')
ax.set_title('Tank shape changes the drain-time scale')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

A1 = A2 = 1.0     # m^2
R12 = 5.0         # s/m^2
R2  = 10.0        # s/m^2
qin = 0.1         # m^3/s

Amat = np.array([[-1/(A1*R12), 1/(A1*R12)],
                  [1/(A2*R12), -1/(A2*R12) - 1/(A2*R2)]])
bvec = np.array([qin/A1, 0.0])

eigval, eigvec = np.linalg.eig(Amat)
h_eq = np.linalg.solve(-Amat, bvec)

print("A =\n", Amat)
print("eigenvalues:", eigval)
print("equilibrium (h1_eq, h2_eq):", h_eq)

def linear_tanks(t, h):
    return Amat @ h + bvec

t_span = (0, 150)
t_eval = np.linspace(*t_span, 400)
sol_lin = solve_ivp(linear_tanks, t_span, [0.0, 0.0], t_eval=t_eval)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(sol_lin.t, sol_lin.y[0], color='steelblue', lw=2.5, label='$h_1(t)$')
axes[0].plot(sol_lin.t, sol_lin.y[1], color='crimson', lw=2.5, label='$h_2(t)$')
axes[0].axhline(h_eq[0], color='steelblue', ls=':', lw=1.2, label=f'$h_{{1,eq}}={h_eq[0]:.2f}$ m')
axes[0].axhline(h_eq[1], color='crimson', ls=':', lw=1.2, label=f'$h_{{2,eq}}={h_eq[1]:.2f}$ m')
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Height (m)')
axes[0].set_title('Approach to equilibrium (no overshoot)')
axes[0].legend(fontsize=8)

axes[1].plot(sol_lin.y[0], sol_lin.y[1], color='steelblue', lw=2.5)
axes[1].plot(0, 0, 'ko', ms=7, label='Start', zorder=5)
axes[1].plot(h_eq[0], h_eq[1], 'g*', ms=13, label='Equilibrium', zorder=5)
# eigenvector directions through the equilibrium
for lam, vec, color, lbl in zip(eigval, eigvec.T, ['gray', 'darkorange'],
                                 ['Fast eigendirection', 'Slow eigendirection']):
    pts = np.array([h_eq - 0.6*vec, h_eq + 0.6*vec])
    axes[1].plot(pts[:,0], pts[:,1], color=color, ls='--', lw=1.3, label=lbl)
axes[1].set_xlabel('$h_1$ (m)'); axes[1].set_ylabel('$h_2$ (m)')
axes[1].set_title('Phase portrait: a stable node')
axes[1].legend(fontsize=7.5)

plt.tight_layout()
plt.show()

qin  = 0.1
c12  = 0.1
c2   = 0.1

def tank_rhs(h):
    h1, h2 = h
    Q12 = c12*np.sign(h1-h2)*np.sqrt(abs(h1-h2))
    Q2  = c2*np.sqrt(max(h2, 0.0))
    return np.array([qin - Q12, Q12 - Q2])

h_eq_nl = fsolve(tank_rhs, [1.5, 1.0])
print("Nonlinear equilibrium (h1_eq, h2_eq):", h_eq_nl)

# Jacobian, evaluated symbolically then substituted at equilibrium
h1s, h2s = sym.symbols('h1 h2', positive=True)
Q12s = c12*sym.sqrt(h1s - h2s)
Q2s  = c2*sym.sqrt(h2s)
f1 = qin - Q12s
f2 = Q12s - Q2s
Jsym = sym.Matrix([f1, f2]).jacobian([h1s, h2s])
Jnum = np.array(Jsym.subs({h1s: h_eq_nl[0], h2s: h_eq_nl[1]})).astype(float)
print("Jacobian at equilibrium:\n", Jnum)
print("eigenvalues of Jacobian:", np.linalg.eigvals(Jnum))

def rhs_t(t, h):
    h1, h2 = max(h[0], -10), max(h[1], -10)
    Q12 = c12*np.sign(h1-h2)*np.sqrt(abs(h1-h2))
    Q2  = c2*np.sqrt(abs(h2)) * (1 if h2 >= 0 else -1)
    return [qin - Q12, Q12 - Q2]

t_final = 400
t_fine = np.linspace(0, t_final, 2000)
sol_ref = solve_ivp(rhs_t, (0, t_final), [0.0, 0.0], t_eval=t_fine, max_step=0.1)

def euler_fixed(dt):
    n = int(t_final/dt)
    h = np.zeros((n+1, 2))
    t = np.zeros(n+1)
    for i in range(n):
        h[i+1] = h[i] + dt*np.array(rhs_t(t[i], h[i]))
        t[i+1] = t[i] + dt
    return t, h

t_e10, h_e10 = euler_fixed(10.0)
t_e20, h_e20 = euler_fixed(20.0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(sol_ref.t, sol_ref.y[1], color='steelblue', lw=2.5, label='RK45 reference')
axes[0].plot(t_e10, h_e10[:,1], 'o-', color='darkorange', ms=3, lw=1, label='Euler, $\\Delta t=10$ s')
axes[0].plot(t_e20, h_e20[:,1], 's-', color='crimson', ms=4, lw=1, label='Euler, $\\Delta t=20$ s')
axes[0].axhline(0, color='k', lw=0.7)
axes[0].axhline(h_eq_nl[1], color='gray', ls=':', lw=1, label=f'$h_{{2,eq}}={h_eq_nl[1]:.1f}$ m')
axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('$h_2(t)$ (m)')
axes[0].set_title('Coarse Euler drives $h_2$ negative')
axes[0].legend(fontsize=8)

axes[1].plot(sol_ref.y[0], sol_ref.y[1], color='steelblue', lw=2.5, label='RK45 reference')
axes[1].plot(h_e10[:,0], h_e10[:,1], 'o-', color='darkorange', ms=3, lw=1, label='Euler, $\\Delta t=10$ s')
axes[1].plot(h_e20[:,0], h_e20[:,1], 's-', color='crimson', ms=4, lw=1, label='Euler, $\\Delta t=20$ s')
axes[1].plot(h_eq_nl[0], h_eq_nl[1], 'g*', ms=13, label='Equilibrium', zorder=5)
axes[1].set_xlabel('$h_1$ (m)'); axes[1].set_ylabel('$h_2$ (m)')
axes[1].set_title('Phase plane: coarse Euler overshoots the node')
axes[1].legend(fontsize=7.5)

plt.tight_layout()
plt.show()

g = 9.81

# ── Stokes drag: steel ball bearing in glycerin ──────────────
r_ball   = 1.5e-3      # m
rho_steel, rho_gly, mu = 7800.0, 1260.0, 1.4
m_ball   = (4/3)*np.pi*r_ball**3*rho_steel
Vol_ball = (4/3)*np.pi*r_ball**3
net_grav = m_ball*g - rho_gly*Vol_ball*g   # weight minus buoyancy
c_stokes = 6*np.pi*mu*r_ball
v_term_s = net_grav/c_stokes
tau_s    = m_ball/c_stokes
Re_s     = rho_gly*v_term_s*(2*r_ball)/mu
print(f"Stokes: v_term = {v_term_s*100:.2f} cm/s, tau = {tau_s*1e3:.2f} ms, Re = {Re_s:.3f}")

# ── Quadratic drag: skydiver ──────────────────────────────────
m_sky, rho_air, Cd_sky, Area = 75.0, 1.225, 1.0, 0.7
k_quad   = 0.5*rho_air*Cd_sky*Area
v_term_q = np.sqrt(m_sky*g/k_quad)
t95      = v_term_q/g*np.arctanh(0.95)
print(f"Quadratic: v_term = {v_term_q:.2f} m/s = {v_term_q*3.6:.0f} km/h, t95% = {t95:.2f} s")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

t_s = np.linspace(0, 7*tau_s, 300)
v_s = v_term_s*(1-np.exp(-t_s/tau_s))
axes[0].plot(t_s*1e3, v_s*100, color='steelblue', lw=2.5)
axes[0].axhline(v_term_s*100, color='crimson', ls='--', lw=1.2, label=f'$v_{{term}}={v_term_s*100:.2f}$ cm/s')
axes[0].axvline(tau_s*1e3, color='gray', ls=':', lw=1.2, label=f'$\\tau={tau_s*1e3:.2f}$ ms')
axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('$v(t)$ (cm/s)')
axes[0].set_title('Stokes drag: ball bearing in glycerin')
axes[0].legend(fontsize=8.5)

t_q = np.linspace(0, 30, 300)
v_q = v_term_q*np.tanh(g*t_q/v_term_q)
axes[1].plot(t_q, v_q, color='darkorange', lw=2.5)
axes[1].axhline(v_term_q, color='crimson', ls='--', lw=1.2, label=f'$v_{{term}}={v_term_q:.1f}$ m/s')
axes[1].axvline(t95, color='gray', ls=':', lw=1.2, label=f'95% at $t={t95:.1f}$ s')
axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('$v(t)$ (m/s)')
axes[1].set_title('Quadratic drag: skydiver in free fall')
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.show()

t, s = sym.symbols('t s', positive=True)
zeta_s, wn_s, zss_s = sym.symbols('zeta omega_n z_ss', positive=True)

Zs = wn_s**2*zss_s/(s*(s**2 + 2*zeta_s*wn_s*s + wn_s**2))
z_of_t = sym.inverse_laplace_transform(Zs, s, t)
display(Math("Z(s) = " + sym.latex(Zs)))
display(Math("z(t) = " + sym.latex(sym.simplify(z_of_t))))

zeta_v, wn_v, zss_v = 0.15, 0.05, 2.0
wd_v = wn_v*np.sqrt(1-zeta_v**2)
phi_v = np.arccos(zeta_v)
tp_v  = np.pi/wd_v
OS_v  = np.exp(-zeta_v*np.pi/np.sqrt(1-zeta_v**2))
ts_2pct = 4/(zeta_v*wn_v)
print(f"Peak time = {tp_v:.1f} s = {tp_v/60:.2f} min,  overshoot = {OS_v*100:.1f}%")
print(f"Peak level = {zss_v*(1+OS_v):.2f} m,  2%-settling time (est.) = {ts_2pct:.0f} s = {ts_2pct/60:.1f} min")

t_plot = np.linspace(0, 600, 1000)
z_t = zss_v*(1 - np.exp(-zeta_v*wn_v*t_plot)/np.sqrt(1-zeta_v**2)*np.sin(wd_v*t_plot + phi_v))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(t_plot, z_t, color='steelblue', lw=2.5)
ax.axhline(zss_v, color='k', ls='--', lw=1, label=f'New steady level $z_{{ss}}={zss_v}$ m')
ax.plot(tp_v, zss_v*(1+OS_v), 'ro', ms=7, zorder=5,
        label=f'Peak: {zss_v*(1+OS_v):.2f} m at $t={tp_v:.0f}$ s')
ax.fill_between(t_plot, zss_v*0.98, zss_v*1.02, color='gray', alpha=0.15, label='$\\pm 2\\%$ band')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Surge level $z(t)$ (m)')
ax.set_title('Surge-tank oscillation following a valve closure')
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.show()