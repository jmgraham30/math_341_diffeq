import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

def alpha_m(V): return 0.1*(V+40.0)/(1-np.exp(-(V+40.0)/10.0))
def beta_m(V):  return 4.0*np.exp(-(V+65.0)/18.0)
def alpha_h(V): return 0.07*np.exp(-(V+65.0)/20.0)
def beta_h(V):  return 1.0/(1+np.exp(-(V+35.0)/10.0))
def alpha_n(V): return 0.01*(V+55.0)/(1-np.exp(-(V+55.0)/10.0))
def beta_n(V):  return 0.125*np.exp(-(V+65.0)/80.0)

V_range = np.linspace(-100, 50, 400)
m_inf = alpha_m(V_range)/(alpha_m(V_range)+beta_m(V_range))
h_inf = alpha_h(V_range)/(alpha_h(V_range)+beta_h(V_range))
n_inf = alpha_n(V_range)/(alpha_n(V_range)+beta_n(V_range))
tau_m = 1/(alpha_m(V_range)+beta_m(V_range))
tau_h = 1/(alpha_h(V_range)+beta_h(V_range))
tau_n = 1/(alpha_n(V_range)+beta_n(V_range))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(V_range, m_inf, color='steelblue', lw=2.5, label='$m_\\infty(V)$')
axes[0].plot(V_range, h_inf, color='crimson',   lw=2.5, label='$h_\\infty(V)$')
axes[0].plot(V_range, n_inf, color='darkorange',lw=2.5, label='$n_\\infty(V)$')
axes[0].set_xlabel('V (mV)'); axes[0].set_ylabel('Steady-state value')
axes[0].set_title('Activation/inactivation curves')
axes[0].legend(fontsize=9)

axes[1].plot(V_range, tau_m, color='steelblue', lw=2.5, label='$\\tau_m(V)$')
axes[1].plot(V_range, tau_h, color='crimson',   lw=2.5, label='$\\tau_h(V)$')
axes[1].plot(V_range, tau_n, color='darkorange',lw=2.5, label='$\\tau_n(V)$')
axes[1].set_xlabel('V (mV)'); axes[1].set_ylabel('Time constant (ms)')
axes[1].set_title('Voltage-dependent time constants')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

Cm  = 1.0
gNa, ENa = 120.0, 50.0
gK,  EK  = 36.0, -77.0
gL,  EL  = 0.3,  -54.387

def hh_rhs(t, y, I_ext):
    V, m, h, n = y
    INa = gNa*m**3*h*(V-ENa)
    IK  = gK*n**4*(V-EK)
    IL  = gL*(V-EL)
    dV = (I_ext(t) - INa - IK - IL)/Cm
    dm = alpha_m(V)*(1-m) - beta_m(V)*m
    dh = alpha_h(V)*(1-h) - beta_h(V)*h
    dn = alpha_n(V)*(1-n) - beta_n(V)*n
    return [dV, dm, dh, dn]

# resting initial condition: hold V fixed at V0 and let gates reach x_inf(V0)
V0 = -65.0
m0 = alpha_m(V0)/(alpha_m(V0)+beta_m(V0))
h0 = alpha_h(V0)/(alpha_h(V0)+beta_h(V0))
n0 = alpha_n(V0)/(alpha_n(V0)+beta_n(V0))
y0 = [V0, m0, h0, n0]
print(f"Resting state: V0={V0} mV, m0={m0:.4f}, h0={h0:.4f}, n0={n0:.4f}")

def pulse(t, t_on=5, t_off=60, amp=10.0):
    return amp if t_on <= t <= t_off else 0.0

t_span = (0, 100)
t_eval = np.linspace(*t_span, 4000)
sol = solve_ivp(hh_rhs, t_span, y0, args=(lambda t: pulse(t),),
                 t_eval=t_eval, max_step=0.02)

fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

axes[0].plot(sol.t, sol.y[0], color='steelblue', lw=1.8)
axes[0].axvspan(5, 60, color='orange', alpha=0.15,
                 label='$I_{ext}=10\\,\\mu A/cm^2$')
axes[0].set_ylabel('V (mV)')
axes[0].set_title('Membrane potential')
axes[0].legend(fontsize=8, loc='upper right')

axes[1].plot(sol.t, sol.y[1], color='steelblue',  lw=1.5, label='$m(t)$')
axes[1].plot(sol.t, sol.y[2], color='crimson',    lw=1.5, label='$h(t)$')
axes[1].plot(sol.t, sol.y[3], color='darkorange', lw=1.5, label='$n(t)$')
axes[1].set_xlabel('t (ms)'); axes[1].set_ylabel('Gating variable')
axes[1].set_title('Gating variables')
axes[1].legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 4.5))
amps = [2, 3, 5, 10]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(amps)))
for amp, color in zip(amps, colors):
    sol = solve_ivp(hh_rhs, (0, 60), y0,
                     args=(lambda t, amp=amp: pulse(t, 5, 60, amp),),
                     t_eval=np.linspace(0, 60, 3000), max_step=0.02)
    ax.plot(sol.t, sol.y[0], color=color, lw=1.8,
            label=f'$I={amp}\\,\\mu A/cm^2$')

ax.set_xlabel('t (ms)'); ax.set_ylabel('V (mV)')
ax.set_title('Response to current pulses of increasing amplitude')
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.show()

I_vals = np.linspace(0, 40, 25)
rates = []
for I_amp in I_vals:
    sol = solve_ivp(hh_rhs, (0, 300), y0,
                     args=(lambda t, amp=I_amp: pulse(t, 5, 300, amp),),
                     t_eval=np.linspace(0, 300, 15000), max_step=0.02)
    peaks, _ = find_peaks(sol.y[0], height=0)
    peak_t = sol.t[peaks]
    peak_t = peak_t[peak_t > 20]           # discard transient
    rate = 1000.0/np.mean(np.diff(peak_t)) if len(peak_t) > 1 else 0.0
    rates.append(rate)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(I_vals, rates, 'o-', color='crimson', lw=1.8, markersize=4)
ax.set_xlabel('$I_{ext}$ ($\\mu A/cm^2$)'); ax.set_ylabel('Firing rate (Hz)')
ax.set_title('f–I curve')
plt.tight_layout()
plt.show()

a_fhn, b_fhn, eps_fhn = 0.7, 0.8, 0.08

def fhn(t, y, I):
    v, w = y
    dv = v - v**3/3 - w + I
    dw = eps_fhn*(v + a_fhn - b_fhn*w)
    return [dv, dw]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
v_arr = np.linspace(-2.5, 2.5, 300)
w_null = (v_arr + a_fhn)/b_fhn

cases = [(0.0, 'steelblue', '$I=0$ (excitable)'),
         (0.5, 'crimson',   '$I=0.5$ (oscillatory)')]

for I, color, lbl in cases:
    v_null = v_arr - v_arr**3/3 + I
    sol = solve_ivp(fhn, (0, 300), [-1.0, -0.5], args=(I,), max_step=0.02,
                     t_eval=np.linspace(200, 300, 3000))
    axes[0].plot(v_arr, v_null, color=color, ls='--', lw=1.2, alpha=0.6)
    axes[0].plot(sol.y[0], sol.y[1], color=color, lw=2, label=lbl)

axes[0].plot(v_arr, w_null, color='k', ls=':', lw=1.3, label='$w$-nullcline')
axes[0].set_xlim(-2.5, 2.5); axes[0].set_ylim(-1, 2.5)
axes[0].set_xlabel('v'); axes[0].set_ylabel('w')
axes[0].set_title('Phase plane: nullclines and trajectories')
axes[0].legend(fontsize=8)

for I, color, lbl in cases:
    sol = solve_ivp(fhn, (0, 200), [-1.0, -0.5], args=(I,), max_step=0.02,
                     t_eval=np.linspace(0, 200, 4000))
    axes[1].plot(sol.t, sol.y[0], color=color, lw=1.8, label=lbl)

axes[1].set_xlabel('t'); axes[1].set_ylabel('v(t)')
axes[1].set_title('Time series')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()