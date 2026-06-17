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
#| label: fig-rc-filter
#| fig-cap: "RC low-pass filter ($R=1\\,\\text{k}\\Omega$, $C=1\\,\\mu\\text{F}$, $\\tau=1\\,\\text{ms}$, $f_c=159\\,\\text{Hz}$). Left: step response — the capacitor charges toward $E_0=5\\,\\text{V}$ with time constant $\\tau$. Right: Bode magnitude plot — the filter passes low frequencies and attenuates high frequencies, rolling off at $-20\\,\\text{dB/decade}$ above $f_c$."

R_val  = 1e3     # 1 kΩ
C_val  = 1e-6    # 1 μF
tau    = R_val * C_val   # 1 ms
fc     = 1/(2*np.pi*tau) # 159 Hz
E0     = 5.0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Step response ────────────────────────────────────────────
t_step = np.linspace(0, 6*tau, 400)
V_step = E0 * (1 - np.exp(-t_step/tau))
axes[0].plot(t_step*1e3, V_step, color='steelblue', lw=2.5)
axes[0].axhline(E0, color='k', ls='--', lw=1, label=f'$E_0={E0}$ V')
axes[0].axvline(tau*1e3, color='crimson', ls=':', lw=1.5,
                label=f'$\\tau={tau*1e3:.0f}$ ms ($V_C=0.632E_0$)')
axes[0].plot(tau*1e3, E0*(1-1/np.e), 'ro', markersize=7, zorder=5)
axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('$V_C(t)$ (V)')
axes[0].set_title('Step response ($V_{in}=5$ V, $V_C(0)=0$)')
axes[0].legend(fontsize=8.5); axes[0].set_ylim(0, 5.5)

# ── Bode magnitude plot ──────────────────────────────────────
f_arr = np.logspace(1, 5, 500)  # 10 Hz to 100 kHz
omega_arr = 2*np.pi*f_arr
H_mag  = 1.0/np.sqrt(1 + (tau*omega_arr)**2)
H_dB   = 20*np.log10(H_mag)
# -20 dB/decade asymptote
H_asymp = np.where(f_arr < fc, np.zeros_like(f_arr),
                   20*np.log10(fc/f_arr))

axes[1].semilogx(f_arr, H_dB, color='steelblue', lw=2.5, label='$|H(f)|$ (dB)')
axes[1].semilogx(f_arr, H_asymp, color='crimson', lw=1.5, ls='--',
                 label='$-20$ dB/decade asymptote')
axes[1].axvline(fc, color='darkorange', ls=':', lw=1.5,
                label=f'Cut-off $f_c={fc:.0f}$ Hz ($-3$ dB)')
axes[1].axhline(-3, color='darkorange', ls=':', lw=1)
axes[1].set_xlabel('Frequency (Hz)'); axes[1].set_ylabel('$|H|$ (dB)')
axes[1].set_title('Bode magnitude plot (log frequency)')
axes[1].legend(fontsize=8); axes[1].set_ylim(-60, 5)
axes[1].grid(True, which='both', alpha=0.3)

plt.suptitle(f'RC Low-Pass Filter ($R={R_val/1e3:.0f}\\,\\mathrm{{k}}\\Omega$, '
             f'$C={C_val*1e6:.0f}\\,\\mu\\mathrm{{F}}$, '
             f'$\\tau={tau*1e3:.0f}\\,\\mathrm{{ms}}$)', fontsize=11)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-rlc-free
#| fig-cap: "RLC free response ($L=0.1\\,\\text{H}$, $C=100\\,\\mu\\text{F}$, $V_0=10\\,\\text{V}$). Left: capacitor voltage $V_C=Q/C$ for three damping regimes. The underdamped circuit oscillates like a decaying spring. Right: phase portrait $(Q, I)$ — the underdamped case spirals inward; the overdamped case decays without orbiting."

L_val = 0.1     # H
C_val = 1e-4    # 100 μF
V0    = 10.0    # initial voltage
Q0    = C_val * V0

omega0 = 1/np.sqrt(L_val*C_val)  # 316.2 rad/s
R_crit = 2*np.sqrt(L_val/C_val)  # critical resistance

R_cases = [
    (R_crit*0.15, 'steelblue',  'Underdamped $R=0.15R_c$'),
    (R_crit,      'darkorange', 'Critically damped $R=R_c$'),
    (R_crit*3.0,  'crimson',    'Overdamped $R=3R_c$'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
t_plot = np.linspace(0, 0.06, 1000)

for R_val, color, lbl in R_cases:
    alpha = R_val/(2*L_val)
    def rlc_free(t, y, a=alpha):
        return [y[1], -2*a*y[1] - omega0**2*y[0]]
    sol = solve_ivp(rlc_free, (0, 0.06), [Q0, 0.0], t_eval=t_plot, max_step=5e-5)
    Vc = sol.y[0] / C_val  # voltage across capacitor
    axes[0].plot(sol.t*1e3, Vc, color=color, lw=2, label=lbl)
    axes[1].plot(sol.y[0]*1e3, sol.y[1], color=color, lw=2, label=lbl)

axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('$V_C(t)$ (V)')
axes[0].set_title(f'RLC free response ($\\omega_0={omega0:.0f}$ rad/s, $R_c={R_crit:.1f}\\,\\Omega$)')
axes[0].legend(fontsize=8)

axes[1].plot(Q0*1e3, 0, 'ko', markersize=7, label='IC', zorder=5)
axes[1].plot(0, 0, 'k*', markersize=10, label='Equilibrium', zorder=5)
axes[1].set_xlabel('$Q$ (mC)'); axes[1].set_ylabel('$I = Q\'$ (A)')
axes[1].set_title('Phase portrait $(Q, I)$')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-rlc-impedance
#| fig-cap: "Series RLC driven response ($L=10\\,\\text{mH}$, $C=10\\,\\mu\\text{F}$, $\\omega_0\\approx 3162\\,\\text{rad/s}$, $f_0\\approx 503\\,\\text{Hz}$). Left: impedance $|Z(\\omega)|$ vs. frequency for four values of $R$. The minimum impedance at resonance equals $R$. Right: normalized amplitude response $|H(\\omega)| = R/|Z(\\omega)|$ — the curves are labeled by their $Q$ factor. Higher $Q$ gives a sharper, taller peak."

L_v  = 10e-3   # 10 mH
C_v  = 10e-6   # 10 μF
omega0_v = 1/np.sqrt(L_v*C_v)

R_vals = [5.0, 10.0, 25.0, 100.0]
Q_vals_computed = [omega0_v*L_v/R for R in R_vals]
colors_r = plt.cm.viridis(np.linspace(0.1, 0.85, len(R_vals)))

f_arr   = np.logspace(1.5, 4.5, 1000)
omega_r = 2*np.pi*f_arr

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for R, Qc, color in zip(R_vals, Q_vals_computed, colors_r):
    Z  = np.sqrt(R**2 + (omega_r*L_v - 1/(omega_r*C_v))**2)
    H  = R/Z   # normalized: current amplitude / max current
    axes[0].semilogx(f_arr, Z, color=color, lw=2, label=f'$R={R}\\,\\Omega$, $Q={Qc:.1f}$')
    axes[1].semilogx(f_arr, H, color=color, lw=2, label=f'$Q={Qc:.1f}$')

f0_v = omega0_v/(2*np.pi)
for ax in axes:
    ax.axvline(f0_v, color='k', ls='--', lw=1.2, label=f'$f_0={f0_v:.0f}$ Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.grid(True, which='both', alpha=0.2)

axes[0].set_ylabel('$|Z(\\omega)|$ ($\\Omega$)')
axes[0].set_title('Impedance magnitude')
axes[0].legend(fontsize=7.5)

axes[1].axhline(1/np.sqrt(2), color='gray', ls=':', lw=1.2, label='$-3$ dB ($1/\\sqrt{2}$)')
axes[1].set_ylabel('Normalized amplitude $|H|$')
axes[1].set_title('Amplitude response (normalized to resonant peak)')
axes[1].legend(fontsize=7.5)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-am-radio
#| fig-cap: "AM radio tuning circuit ($L=100\\,\\mu\\text{H}$, $R=6.28\\,\\Omega$, $Q=100$, $f_0=1\\,\\text{MHz}$). Left: amplitude response centered at $f_0=1\\,\\text{MHz}$ with bandwidth $\\Delta f = 10\\,\\text{kHz}$ — the circuit passes the desired station (shaded blue) and rejects neighboring stations (shaded red). Right: how varying $C$ shifts the resonant frequency to tune across the AM band (540–1700 kHz)."

L_am  = 100e-6   # 100 μH
R_am  = 6.28     # Ω  => Q = omega0*L/R = 100 at 1 MHz
C_am  = 253.3e-12  # 253.3 pF => f0 = 1 MHz

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Selectivity at f0=1 MHz ──────────────────────────────────
f_arr  = np.linspace(900e3, 1100e3, 2000)
omega_ = 2*np.pi*f_arr
Z_am   = np.sqrt(R_am**2 + (omega_*L_am - 1/(omega_*C_am))**2)
H_am   = R_am/Z_am

f0_am  = 1/(2*np.pi*np.sqrt(L_am*C_am))
Q_am   = f0_am*L_am*2*np.pi/R_am
BW_am  = f0_am/Q_am

axes[0].plot(f_arr/1e3, H_am, color='steelblue', lw=2.5, label=f'$Q={Q_am:.0f}$, BW$={BW_am/1e3:.0f}$ kHz')
axes[0].axvline(f0_am/1e3, color='k', ls='--', lw=1, alpha=0.7)
# Desired station passband
mask_pass = np.abs(f_arr - f0_am) < BW_am/2
axes[0].fill_between(f_arr[mask_pass]/1e3, 0, H_am[mask_pass],
                     alpha=0.25, color='steelblue', label='Desired station')
# Adjacent stations
for f_adj in [990e3, 1010e3]:
    mask_adj = (np.abs(f_arr - f_adj) < 5e3)
    axes[0].fill_between(f_arr[mask_adj]/1e3, 0, H_am[mask_adj],
                         alpha=0.3, color='crimson')
axes[0].annotate('Rejected\nstation', xy=(990, 0.08), fontsize=8, ha='center', color='crimson')
axes[0].annotate('Rejected\nstation', xy=(1010, 0.08), fontsize=8, ha='center', color='crimson')
axes[0].annotate(f'$f_0={f0_am/1e3:.0f}$ kHz', xy=(f0_am/1e3+2, 0.75), fontsize=8)
axes[0].set_xlabel('Frequency (kHz)'); axes[0].set_ylabel('Normalized amplitude')
axes[0].set_title('Selectivity at $f_0=1$ MHz')
axes[0].legend(fontsize=8.5); axes[0].set_ylim(0, 1.1)

# ── Tuning by varying C ──────────────────────────────────────
f_tune = np.linspace(540e3, 1700e3, 200)  # AM band
C_tune = 1/(L_am*(2*np.pi*f_tune)**2)    # capacitance needed for each f0
f_sweep = np.linspace(400e3, 1900e3, 2000)
for f_target, alpha_val in zip([600e3, 1000e3, 1500e3], [0.5, 1.0, 0.6]):
    C_t = 1/(L_am*(2*np.pi*f_target)**2)
    omega_s = 2*np.pi*f_sweep
    Z_t = np.sqrt(R_am**2 + (omega_s*L_am - 1/(omega_s*C_t))**2)
    H_t = R_am/Z_t
    lbl = f'$C={C_t*1e12:.0f}$ pF, $f_0={f_target/1e3:.0f}$ kHz'
    axes[1].plot(f_sweep/1e3, H_t, lw=2, label=lbl, alpha=alpha_val if alpha_val<1 else 1.0)

axes[1].axvspan(540, 1700, alpha=0.06, color='steelblue', label='AM band (540–1700 kHz)')
axes[1].set_xlabel('Frequency (kHz)'); axes[1].set_ylabel('Normalized amplitude')
axes[1].set_title('Tuning: varying $C$ shifts $f_0 = 1/2\\pi\\sqrt{LC}$')
axes[1].legend(fontsize=7.5); axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-dc-motor
#| fig-cap: "DC motor step response to $V=12$ V applied at $t=0$ (all initial conditions zero). Left: angular velocity $\\omega(t)$ and armature current $I(t)$ — the current spikes immediately (electrical time constant $\\tau_e=5\\,\\text{ms}$) while speed builds more slowly (mechanical time constant $\\tau_m=100\\,\\text{ms}$). Right: phase portrait $(\\omega, I)$ showing the trajectory from rest to steady state. Both quantities approach their analytically computed steady-state values (dashed lines)."

# Motor parameters
La  = 0.01   # H   (armature inductance)
Ra  = 2.0    # Ω   (armature resistance)
J   = 0.01   # kg·m² (moment of inertia)
b   = 0.1    # N·m·s (viscous friction)
Kb  = 0.5    # V·s/rad (back-EMF constant)
Kt  = 0.5    # N·m/A  (torque constant)
V_dc = 12.0  # V

tau_e = La/Ra        # electrical time constant
tau_m = J/b          # mechanical time constant
print(f"tau_e = {tau_e*1e3:.1f} ms,  tau_m = {tau_m*1e3:.1f} ms")

# Steady state
omega_ss = Kt*V_dc / (b*Ra + Kb*Kt)
I_ss     = b*omega_ss / Kt
print(f"omega_ss = {omega_ss:.3f} rad/s = {omega_ss*60/(2*np.pi):.1f} RPM")
print(f"I_ss     = {I_ss:.3f} A")

def motor_ode(t, y):
    I, omega = y
    dI     = (V_dc - Ra*I - Kb*omega) / La
    domega = (Kt*I  - b*omega)        / J
    return [dI, domega]

t_plot = np.linspace(0, 0.5, 2000)
sol = solve_ivp(motor_ode, (0, 0.5), [0.0, 0.0], t_eval=t_plot, max_step=1e-4)
I_t, omega_t = sol.y

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Time domain ──────────────────────────────────────────────
ax_tw = axes[0].twinx()
axes[0].plot(t_plot*1e3, omega_t, color='steelblue', lw=2.5, label='$\\omega(t)$ (rad/s)')
ax_tw.plot(t_plot*1e3,  I_t,     color='crimson',   lw=2, ls='--', label='$I(t)$ (A)')
axes[0].axhline(omega_ss, color='steelblue', ls=':', lw=1.2,
                label=f'$\\omega_{{ss}}={omega_ss:.2f}$ rad/s')
ax_tw.axhline(I_ss, color='crimson', ls=':', lw=1.2,
              label=f'$I_{{ss}}={I_ss:.2f}$ A')

axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Angular velocity $\\omega$ (rad/s)', color='steelblue')
ax_tw.set_ylabel('Current $I$ (A)', color='crimson')
axes[0].set_title('DC Motor step response ($V=12$ V)')

lines1, labels1 = axes[0].get_legend_handles_labels()
lines2, labels2 = ax_tw.get_legend_handles_labels()
axes[0].legend(lines1+lines2, labels1+labels2, fontsize=8, loc='center right')

# ── Phase portrait (omega, I) ────────────────────────────────
axes[1].plot(omega_t, I_t, color='steelblue', lw=2.5)
axes[1].plot(0, 0, 'ko', markersize=8, label='Start $(0,0)$', zorder=5)
axes[1].plot(omega_ss, I_ss, 'g*', markersize=12,
             label=f'Steady state $({omega_ss:.1f},{I_ss:.1f})$', zorder=5)
axes[1].set_xlabel('Angular velocity $\\omega$ (rad/s)')
axes[1].set_ylabel('Current $I$ (A)')
axes[1].set_title('Phase portrait $(\\omega, I)$')
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-motor-control
#| fig-cap: "DC motor speed control by varying applied voltage. Left: step responses for four voltage levels — the steady-state speed scales linearly with $V$ (the motor is a linear system). Right: effect of friction coefficient $b$ on transient response with $V=12$ V — higher friction gives faster settling but lower steady-state speed."

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
t_plot = np.linspace(0, 0.5, 1000)

# Varying voltage
for V_val, color, lbl in zip([3, 6, 9, 12],
                              plt.cm.Blues(np.linspace(0.4, 0.9, 4)),
                              ['$V=3$ V','$V=6$ V','$V=9$ V','$V=12$ V']):
    def motor_V(t, y, V=V_val):
        I, omega = y
        return [(V - Ra*I - Kb*omega)/La, (Kt*I - b*omega)/J]
    sol = solve_ivp(motor_V, (0,0.5), [0,0], t_eval=t_plot, max_step=1e-4)
    axes[0].plot(sol.t*1e3, sol.y[1], color=color, lw=2, label=lbl)
    omega_ss_v = Kt*V_val/(b*Ra + Kb*Kt)
    axes[0].axhline(omega_ss_v, color=color, ls=':', lw=1, alpha=0.7)

axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('$\\omega(t)$ (rad/s)')
axes[0].set_title('Speed control: varying $V$')
axes[0].legend(fontsize=8.5)

# Varying friction
V_fixed = 12.0
for b_val, color, lbl in zip([0.05, 0.1, 0.2, 0.5],
                              plt.cm.Reds(np.linspace(0.4, 0.9, 4)),
                              ['$b=0.05$','$b=0.1$','$b=0.2$','$b=0.5$']):
    def motor_b(t, y, bv=b_val):
        I, omega = y
        return [(V_fixed - Ra*I - Kb*omega)/La, (Kt*I - bv*omega)/J]
    sol = solve_ivp(motor_b, (0,0.5), [0,0], t_eval=t_plot, max_step=1e-4)
    omega_ss_b = Kt*V_fixed/(b_val*Ra + Kb*Kt)
    axes[1].plot(sol.t*1e3, sol.y[1], color=color, lw=2,
                 label=f'{lbl}, $\\omega_{{ss}}={omega_ss_b:.1f}$')

axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('$\\omega(t)$ (rad/s)')
axes[1].set_title('Effect of friction $b$ on transient ($V=12$ V)')
axes[1].legend(fontsize=7.5)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
