#| code-fold: true
#| code-summary: "Show the code"

import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import solve, lu, qr, svd, norm, lstsq
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-vectors
#| fig-cap: "Geometric view of vectors in $\\mathbb{R}^2$: the sum $\\mathbf{u}+\\mathbf{v}$ (parallelogram law), and the projection of $\\mathbf{u}$ onto $\\mathbf{v}$."

u = np.array([3.0, 1.0])
v = np.array([1.0, 3.0])
proj = (np.dot(u,v)/np.dot(v,v)) * v

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect('equal')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

origin = np.array([0,0])
for vec, color, lbl in [(u,'steelblue','$\\mathbf{u}=(3,1)$'),
                        (v,'crimson','$\\mathbf{v}=(1,3)$'),
                        (u+v,'seagreen','$\\mathbf{u}+\\mathbf{v}=(4,4)$')]:
    ax.annotate('', xy=vec, xytext=origin,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(*(vec*0.55), lbl, color=color, fontsize=11)

# Parallelogram
ax.plot([u[0],u[0]+v[0]], [u[1],u[1]+v[1]], '--', color='steelblue', lw=1, alpha=0.5)
ax.plot([v[0],u[0]+v[0]], [v[1],u[1]+v[1]], '--', color='crimson', lw=1, alpha=0.5)

# Projection
ax.annotate('', xy=proj, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))
ax.plot([u[0],proj[0]], [u[1],proj[1]], ':', color='darkorange', lw=1.5)
ax.text(*(proj*0.5+np.array([0.1,-0.4])), r'$\mathrm{proj}_\mathbf{v}\mathbf{u}$',
        color='darkorange', fontsize=10)

ax.set_xlim(-0.5, 5); ax.set_ylim(-0.5, 5)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Vector operations in $\\mathbb{R}^2$')
plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"

u = np.array([1.0, 2.0, 3.0]); v = np.array([4.0, -1.0, 2.0])
print("Vector operations in R^3:")
print(f"  u = {u},  v = {v}")
print(f"  u + v = {u+v},  3u = {3*u}")
print(f"  u·v = {np.dot(u,v):.4f},  ||u|| = {norm(u):.4f},  ||v|| = {norm(v):.4f}")
angle_deg = np.degrees(np.arccos(np.dot(u,v)/(norm(u)*norm(v))))
print(f"  angle(u,v) = {angle_deg:.2f}°")
print(f"  u×v = {np.cross(u,v)}  (cross product in R^3)")
proj_u_onto_v = (np.dot(u,v)/np.dot(v,v))*v
print(f"  proj_v(u) = {proj_u_onto_v}")


#| code-fold: true
#| code-summary: "Show the code"

A = np.array([[2,1,-1],[1,3,2],[3,-1,1]], dtype=float)
print(f"A =\n{A}\n")
print(f"det(A)    = {np.linalg.det(A):.4f}")
print(f"rank(A)   = {np.linalg.matrix_rank(A)}")
print(f"trace(A)  = {np.trace(A):.4f}")
print(f"A^(-1)    =\n{np.round(np.linalg.inv(A), 4)}")
print(f"A*A^(-1) ≈ I:\n{np.round(A @ np.linalg.inv(A), 6)}")

# Verify det as volume scaling
v1 = A[:,0]; v2 = A[:,1]; v3 = A[:,2]
print(f"\nColumns are vectors v1={v1}, v2={v2}, v3={v3}")
print(f"Parallelepiped volume = |det(A)| = {abs(np.linalg.det(A)):.4f}")


#| code-fold: true
#| code-summary: "Show the code"

print("=== Example: 3×3 system ===")
# 2x1 + x2 - x3 = 8
# x1 + 3x2 + 2x3 = 14
# 3x1 - x2 + x3 = 2
A_aug = sym.Matrix([[2,1,-1,8],[1,3,2,14],[3,-1,1,2]])
rref, pivots = A_aug.rref()
print("Augmented matrix [A|b]:")
display(Math(sym.latex(A_aug)))
print("RREF:")
display(Math(sym.latex(rref)))
print(f"Pivot columns: {pivots}")
print(f"Solution: x1=2, x2=4, x3=0")

print("\n=== Example: underdetermined 2×4 system ===")
B = sym.Matrix([[1,2,0,1,3],[0,0,1,-1,2]])
rref_B, piv_B = B.rref()
print("RREF of [A|b]:")
display(Math(sym.latex(rref_B)))
print(f"Pivot columns: {piv_B}")
print("Free variables: x2 and x4 (columns 1,3 are non-pivot)")
print("Solution: x1=3-2*x2-x4, x3=2+x4, with x2,x4 free")


#| code-fold: true
#| code-summary: "Show the code"

A_num = np.array([[2,1,-1],[1,3,2],[3,-1,1]], dtype=float)
b_num = np.array([8.0, 14.0, 2.0])

x_sol = solve(A_num, b_num)
print(f"scipy.linalg.solve: x = {np.round(x_sol, 6)}")
print(f"Residual ||Ax-b|| = {norm(A_num @ x_sol - b_num):.2e}")
print(f"Condition number = {np.linalg.cond(A_num):.4f}")


#| code-fold: true
#| code-summary: "Show the code"

print("=== Example 1: 2×2 matrix ===")
A1 = np.array([[4,1],[2,3]], dtype=float)
print(f"A = {A1}")
vals, vecs = np.linalg.eig(A1)
for i, (lam, v) in enumerate(zip(vals, vecs.T)):
    print(f"  λ{i+1}={lam:.2f},  v{i+1}={v},  Av={A1@v},  λ*v={lam*v}  error={norm(A1@v-lam*v):.2e}")

print("\n=== Exact computation with SymPy ===")
A1_sym = sym.Matrix([[4,1],[2,3]])
lam = sym.Symbol('lambda')
char_eq = (A1_sym - lam*sym.eye(2)).det()
print(f"Characteristic equation: {sym.expand(char_eq)} = 0")
roots = sym.solve(char_eq, lam)
print(f"Eigenvalues: {roots}")
for root in roots:
    V = (A1_sym - root*sym.eye(2)).nullspace()
    print(f"  λ={root}: eigenvector = {V[0].T}")

print("\n=== Diagonalization: A = PDP^{-1} ===")
A1_sym2 = sym.Matrix([[4,1],[2,3]])
P, D = A1_sym2.diagonalize()
display(Math(r"P = " + sym.latex(P) + r",\quad D = " + sym.latex(D)))
print(f"Verify PDP^(-1) == A: {sym.simplify(P*D*P.inv()) == A1_sym2}")


#| code-fold: true
#| code-summary: "Show the code"

S = np.array([[4,2,0],[2,3,1],[0,1,5]], dtype=float)
vals_S, vecs_S = np.linalg.eigh(S)  # eigh guarantees real evals for symmetric
print(f"Symmetric matrix S =\n{S}\n")
print(f"Eigenvalues (all real): {np.round(vals_S, 4)}")
print(f"Orthogonality of eigenvectors (Q^T Q should be I):")
print(np.round(vecs_S.T @ vecs_S, 8))
Lambda = np.diag(vals_S)
print(f"Reconstruction S = Q*Lambda*Q^T error: {norm(vecs_S@Lambda@vecs_S.T - S):.2e}")


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-eigengeom
#| fig-cap: "Geometric effect of matrix $A=\\begin{pmatrix}4&1\\\\2&3\\end{pmatrix}$ on the unit circle (blue). The matrix stretches the circle into an ellipse (orange). The eigenvectors (red arrows) point in the directions that are merely scaled, not rotated. The scaling factors are the eigenvalues $\\lambda_1=5$, $\\lambda_2=2$."

A_vis = np.array([[4,1],[2,3]], dtype=float)
vals_v, vecs_v = np.linalg.eig(A_vis)
theta = np.linspace(0, 2*np.pi, 300)
circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A_vis @ circle

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect('equal')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.plot(*circle, color='steelblue', lw=2, label='Unit circle')
ax.plot(*ellipse, color='darkorange', lw=2, label='$A$ applied to circle')
for i, (lam, v) in enumerate(zip(vals_v, vecs_v.T)):
    ax.annotate('', xy=lam*v, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='crimson', lw=2.5))
    ax.annotate('', xy=-lam*v, xytext=[0,0],
                arrowprops=dict(arrowstyle='->', color='crimson', lw=2.5))
    ax.text(*(lam*v*1.1), f'$\\lambda_{i+1}={lam:.0f}$', color='crimson', fontsize=10)
ax.set_xlim(-7,7); ax.set_ylim(-7,7)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Eigenvalues/vectors as scaling directions')
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"

A_lu = np.array([[2,1,-1],[4,5,-3],[2,3,1]], dtype=float)
P_lu, L_lu, U_lu = lu(A_lu)
print(f"A =\n{A_lu}\n")
print(f"L =\n{np.round(L_lu,4)}\n")
print(f"U =\n{np.round(U_lu,4)}\n")
print(f"PLU = A: {np.allclose(P_lu@L_lu@U_lu, A_lu)}")
print(f"Solving multiple right-hand sides with LU: fast for large systems")


#| code-fold: true
#| code-summary: "Show the code"

A_qr = np.array([[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
Q_qr, R_qr = qr(A_qr, mode='economic')
print(f"A (4×3):\n{A_qr}\n")
print(f"Q (4×3, orthonormal columns):\n{np.round(Q_qr,4)}\n")
print(f"R (3×3, upper triangular):\n{np.round(R_qr,4)}\n")
print(f"Q^T Q = I: {np.allclose(Q_qr.T@Q_qr, np.eye(3))}")
print(f"QR = A: {np.allclose(Q_qr@R_qr, A_qr)}")


#| code-fold: true
#| code-summary: "Show the code"

A_svd = np.array([[3,2,2],[2,3,-2]], dtype=float)
U_s, sigma, VT_s = svd(A_svd)
print(f"A =\n{A_svd}\n")
print(f"Singular values σ = {sigma}")
print(f"rank(A) = {np.sum(sigma > 1e-10)}")
print(f"Condition number κ(A) = {sigma[0]/sigma[-1]:.4f}")
Sigma_full = np.zeros_like(A_svd)          # shape (2,3), matches A
np.fill_diagonal(Sigma_full, sigma)        # place singular values on diagonal
print(f"Reconstruction error ||U*Σ*V^T - A|| = {norm(U_s@Sigma_full@VT_s - A_svd):.2e}")


#| code-fold: true
#| code-summary: "Show the code"

B_sym = sym.Matrix([[1,2,3],[4,5,6],[7,8,9]])
print(f"Matrix A (rank-deficient):\n{B_sym}\n")
print(f"rank = {B_sym.rank()}")
print(f"Null space basis: {B_sym.nullspace()}")
print(f"Column space basis: {B_sym.columnspace()}")
print(f"Left null space: {B_sym.T.nullspace()}")
print(f"\nRank-Nullity: rank({B_sym.rank()}) + nullity({3-B_sym.rank()}) = {3}")


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-circuit
#| fig-cap: "Resistor network: three nodes (node 3 grounded), four resistors. Nodal analysis yields the $2\\times 2$ system $Y\\mathbf{V}=\\mathbf{I}$ whose solution gives all node voltages directly."

# Network: 4 nodes (node 4 = ground = 0 V)
# Resistors: R12=2Ω, R13=4Ω, R23=3Ω, R24=1Ω, R34=5Ω
# Source: I1=5 A injected at node 1, I2=2 A injected at node 2, I3=0
G12, G13, G23, G24, G34 = 1/2, 1/4, 1/3, 1.0, 1/5

Y = np.array([
    [G12+G13,   -G12,    -G13],
    [-G12,   G12+G23+G24, -G23],
    [-G13,   -G23,    G13+G23+G34]
])
I_inj = np.array([5.0, 2.0, 0.0])  # injected currents
V_sol = solve(Y, I_inj)
print("Nodal admittance matrix Y:")
print(np.round(Y,4))
print(f"\nInjected currents I = {I_inj} A")
print(f"Node voltages: V1={V_sol[0]:.4f} V, V2={V_sol[1]:.4f} V, V3={V_sol[2]:.4f} V")
print(f"Residual ||YV - I|| = {norm(Y@V_sol - I_inj):.2e}")

# Verify by KCL at node 1: sum of outgoing currents = 5 A
I_12 = G12*(V_sol[0]-V_sol[1])
I_13 = G13*(V_sol[0]-V_sol[2])
I_14 = G13*V_sol[0]  # no direct to ground in this simplified model
print(f"\nKCL check at node 1: outgoing currents sum = {Y[0]@V_sol:.4f} A (should be 5.0)")


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-truss
#| fig-cap: "Simple 3-bar planar truss. Each bar contributes a stiffness component in the direction of its axis. The global stiffness matrix is assembled by summing contributions, and $K\\mathbf{u}=\\mathbf{f}$ gives the nodal displacements."

# Three bars meeting at a free node; all other nodes pinned.
# Bar 1: horizontal (angle 0°), AE=1, L=1
# Bar 2: vertical (angle 90°), AE=1, L=1
# Bar 3: diagonal (angle 45°), AE=1, L=sqrt(2)
AE_val = 100e3  # N  (typical for thin steel bar)

def bar_stiffness(AE, L, angle_deg):
    """Local stiffness contribution to 2-DOF free node"""
    th = np.radians(angle_deg)
    cx, cy = np.cos(th), np.sin(th)
    return (AE/L)*np.array([[cx**2, cx*cy],[cx*cy, cy**2]])

K_global = np.zeros((2,2))
bars_info = [(AE_val, 1.0, 0),   # horizontal
             (AE_val, 1.0, 90),  # vertical
             (AE_val, np.sqrt(2), 45)]  # diagonal

for AE_b, L_b, ang in bars_info:
    K_global += bar_stiffness(AE_b, L_b, ang)

F_applied = np.array([10e3, -20e3])  # 10 kN right, 20 kN down
u_disp = solve(K_global, F_applied)

print(f"Global stiffness matrix K [N/m]:\n{np.round(K_global,1)}\n")
print(f"Applied force F = {F_applied/1e3} kN")
print(f"Displacements u = [{u_disp[0]*1e6:.2f}, {u_disp[1]*1e6:.2f}] µm")
print(f"Residual ||Ku-F|| = {norm(K_global@u_disp - F_applied):.2e}")

# Plot truss
fig, ax = plt.subplots(figsize=(6,5))
node_free = np.array([1.0, 1.0])  # free node at center
nodes = {'A':[0,1], 'B':[1,0], 'C':[2,1], 'Free':[1,1]}
bar_conns = [('A','Free'), ('B','Free'), ('C','Free')]
colors_b = ['steelblue','crimson','seagreen']
for (n1,n2), col in zip(bar_conns, colors_b):
    p1, p2 = np.array(nodes[n1]), np.array(nodes[n2])
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], '-', color=col, lw=3)
for name, pos in nodes.items():
    ax.plot(*pos, 'ko', markersize=10, zorder=5)
    ax.text(pos[0]+0.05, pos[1]+0.08, name, fontsize=10)
ax.annotate('', xy=node_free+np.array([0.3,0]),  xytext=node_free,
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2.5))
ax.annotate('', xy=node_free+np.array([0,-0.3]), xytext=node_free,
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2.5))
ax.text(node_free[0]+0.32, node_free[1]+0.02, '$F_x$', color='darkorange')
ax.text(node_free[0]+0.05, node_free[1]-0.35, '$F_y$', color='darkorange')
ax.set_aspect('equal'); ax.set_xlim(-0.3, 2.5); ax.set_ylim(0.3, 1.6)
ax.set_title('3-bar pin-jointed truss: $K\\mathbf{u}=\\mathbf{f}$')
ax.axis('off')
plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-lsq
#| fig-cap: "Least-squares quadratic fit to noisy temperature vs. time data. The design matrix $A$ has columns $[1, t, t^2]$. The normal equations $(A^TA)\\mathbf{c}=A^T\\mathbf{b}$ are solved for the coefficients $\\mathbf{c}=(c_0,c_1,c_2)^T$."

np.random.seed(42)
t_data = np.linspace(0, 5, 25)
T_true = 20 + 8*t_data - 1.2*t_data**2  # true: parabolic heating then cooling
T_meas = T_true + np.random.normal(0, 1.5, len(t_data))

# Design matrix for quadratic model
A_ls = np.column_stack([np.ones_like(t_data), t_data, t_data**2])
# Solve via normal equations
c_ls = solve(A_ls.T @ A_ls, A_ls.T @ T_meas)
# Or equivalently: c_ls, _, _, _ = lstsq(A_ls, T_meas)

t_fine = np.linspace(0, 5, 200)
T_true_fine = 20 + 8*t_fine - 1.2*t_fine**2   # true model on fine grid
A_fine = np.column_stack([np.ones_like(t_fine), t_fine, t_fine**2])
T_fit = A_fine @ c_ls

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(t_data, T_meas, color='steelblue', s=50, zorder=5, label='Measurements')
ax.plot(t_fine, T_true_fine, 'k--', lw=1.5, label='True model')
ax.plot(t_fine, T_fit,  color='crimson', lw=2.5,
        label=f'LS fit: $c_0={c_ls[0]:.2f}$, $c_1={c_ls[1]:.2f}$, $c_2={c_ls[2]:.2f}$')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Temperature (°C)')
ax.set_title('Least-squares quadratic fit: overdetermined system $A\\mathbf{c}\\approx\\mathbf{b}$')
ax.legend(fontsize=9)

print(f"True  coefficients: c0=20, c1=8, c2=-1.2")
print(f"Fitted coefficients: c0={c_ls[0]:.4f}, c1={c_ls[1]:.4f}, c2={c_ls[2]:.4f}")
print(f"RMS residual: {norm(A_ls@c_ls-T_meas)/np.sqrt(len(t_data)):.4f} °C")
plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-markov
#| fig-cap: "Markov chain: four states (equipment conditions: New, Good, Fair, Failed) with maintenance-driven transitions. Left: the $4\\times 4$ transition matrix visualized as a heatmap. Right: convergence of the state distribution to the stationary vector $\\boldsymbol{\\pi}$ regardless of starting state."

# Equipment degradation model (4 states: New, Good, Fair, Failed)
P_mc = np.array([
    [0.85, 0.10, 0.05, 0.00],   # New stays new 85%, degrades
    [0.10, 0.75, 0.10, 0.05],   # Good
    [0.05, 0.15, 0.70, 0.10],   # Fair
    [0.00, 0.00, 0.15, 0.85],   # Failed (repair brings back)
])
# Note: this is row-stochastic; P[i,j] = prob of going FROM i TO j
# Stationary: solve pi^T P = pi^T <=> P^T pi = pi
# Using: (P^T - I)pi = 0, sum(pi)=1

A_st = P_mc.T - np.eye(4)
A_st[-1,:] = 1.0
b_st = np.zeros(4); b_st[-1] = 1.0
pi_st = solve(A_st, b_st)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Heatmap
im = axes[0].imshow(P_mc, cmap='Blues', vmin=0, vmax=1)
states = ['New','Good','Fair','Failed']
axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
axes[0].set_xticklabels(states, fontsize=9); axes[0].set_yticklabels(states, fontsize=9)
axes[0].set_xlabel('To state'); axes[0].set_ylabel('From state')
axes[0].set_title('Transition matrix $P$')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f'{P_mc[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if P_mc[i,j]>0.5 else 'black')
plt.colorbar(im, ax=axes[0], fraction=0.046)

# Convergence from different initial states
n_steps = 30
colors_mc = ['steelblue','crimson','seagreen','darkorange']
for init_state, color in zip(range(4), colors_mc):
    x0 = np.zeros(4); x0[init_state] = 1.0
    dist = x0.copy()
    history = [dist.copy()]
    for _ in range(n_steps):
        dist = P_mc.T @ dist
        history.append(dist.copy())
    history = np.array(history)
    axes[1].plot(history[:,0], color=color, lw=1.8, label=f'Start: {states[init_state]}')
axes[1].axhline(pi_st[0], color='k', ls='--', lw=1.5, label=f'$\\pi_{{\\text{{New}}}}={pi_st[0]:.3f}$')
axes[1].set_xlabel('Step'); axes[1].set_ylabel('P(New state)')
axes[1].set_title(f'Convergence to $\\pi_{{\\text{{New}}}}={pi_st[0]:.3f}$')
axes[1].legend(fontsize=8)
print(f"Stationary distribution π = {dict(zip(states, np.round(pi_st,4)))}")
print(f"Verify P^T π = π: {np.allclose(P_mc.T @ pi_st, pi_st)}")

plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-pca
#| fig-cap: "PCA on 2D correlated sensor data. Left: raw data cloud with two principal component directions (red arrows), scaled by their singular values. PC1 points along the main axis of variation; PC2 is orthogonal. Right: data projected onto the first two PCs — the scatter is maximized along PC1."

np.random.seed(7)
n_obs = 150
# Two correlated sensors
x1 = np.random.randn(n_obs)
x2 = 0.7*x1 + 0.5*np.random.randn(n_obs)
X_raw = np.column_stack([x1, x2])
X_c   = X_raw - X_raw.mean(axis=0)  # center

U_p, s_p, VT_p = svd(X_c, full_matrices=False)
explained = s_p**2 / np.sum(s_p**2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Raw data + PCs
axes[0].scatter(*X_c.T, alpha=0.4, color='steelblue', s=25, label='Data')
scale = s_p / np.sqrt(n_obs-1)
for i, color in enumerate(['crimson','seagreen']):
    pc = VT_p[i,:] * scale[i]
    axes[0].annotate('', xy=pc*2, xytext=-pc*2,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    axes[0].text(*(pc*2.2), f'PC{i+1}\n({explained[i]*100:.1f}%)',
                color=color, fontsize=9, ha='center')
axes[0].set_aspect('equal')
axes[0].set_xlabel('Sensor 1'); axes[0].set_ylabel('Sensor 2')
axes[0].set_title('PCA: principal components of 2D data')

# Projected data
scores = X_c @ VT_p.T
axes[1].scatter(scores[:,0], scores[:,1], alpha=0.4, color='darkorange', s=25)
axes[1].axhline(0, color='k', lw=0.5); axes[1].axvline(0, color='k', lw=0.5)
axes[1].set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
axes[1].set_title('Data in PC coordinates')

print(f"Singular values: σ1={s_p[0]:.4f}, σ2={s_p[1]:.4f}")
print(f"Variance explained: PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%")
print(f"PC1 direction: {np.round(VT_p[0],4)}")

plt.tight_layout(); plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-svd-approx
#| fig-cap: "Low-rank SVD approximation of a $30\\times 40$ test matrix (simulating a grayscale image). As the rank $k$ increases, the reconstruction quality improves. The singular value spectrum (right) shows how rapidly the information concentrates in the first few singular values."

np.random.seed(3)
m_img, n_img = 30, 40
# Create a low-rank-ish matrix (as if a simplified image)
true_rank = 8
A_img = (np.random.randn(m_img, true_rank) @ np.random.randn(true_rank, n_img)
         + 0.3*np.random.randn(m_img, n_img))
U_i, s_i, VT_i = svd(A_img)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
ranks = [1, 2, 4, true_rank]
for ax, k in zip(axes[0,:], ranks):
    A_k = U_i[:,:k] @ np.diag(s_i[:k]) @ VT_i[:k,:]
    rel_err = norm(A_img-A_k,'fro')/norm(A_img,'fro')
    var_frac = np.sum(s_i[:k]**2)/np.sum(s_i**2)
    ax.imshow(A_k, cmap='viridis', aspect='auto')
    ax.set_title(f'rank-{k}\n{var_frac*100:.0f}% var, err={rel_err:.2f}', fontsize=9)
    ax.axis('off')

# Original
axes[1,0].imshow(A_img, cmap='viridis', aspect='auto')
axes[1,0].set_title('Original', fontsize=9); axes[1,0].axis('off')

# Singular value spectrum
ax_spec = axes[1,1]
ax_spec.semilogy(range(1, len(s_i)+1), s_i, 'o-', color='steelblue', markersize=4)
ax_spec.axvline(true_rank, color='crimson', ls='--', lw=1.5, label=f'True rank={true_rank}')
ax_spec.set_xlabel('Index $i$'); ax_spec.set_ylabel('$\\sigma_i$')
ax_spec.set_title('Singular value spectrum'); ax_spec.legend(fontsize=8)

# Cumulative variance
ax_cv = axes[1,2]
cum_var = np.cumsum(s_i**2)/np.sum(s_i**2)
ax_cv.plot(range(1,len(s_i)+1), cum_var*100, 'o-', color='seagreen', markersize=4)
ax_cv.axhline(90, color='gray', ls=':', lw=1.2, label='90% threshold')
ax_cv.set_xlabel('$k$'); ax_cv.set_ylabel('% variance explained')
ax_cv.set_title('Cumulative variance'); ax_cv.legend(fontsize=8)

axes[1,3].axis('off')
plt.suptitle('Low-rank SVD approximation (Eckart–Young theorem)', fontsize=11)
plt.tight_layout(); plt.show()


#| code-fold: false

import numpy as np
from scipy.linalg import solve, lu, qr, svd, norm, lstsq

A = np.array([[2.,1.,-1.],[1.,3.,2.],[3.,-1.,1.]])
b = np.array([8., 14., 2.])

# Solving
x = solve(A, b)                           # exact solution
x_ls, _, _, _ = lstsq(A[:2,:], b[:2])    # least squares

# Factorizations
P, L, U     = lu(A)
Q, R        = qr(A)
U_s, s, VT  = svd(A)

# Eigenvalues
vals, vecs  = np.linalg.eig(A)           # general
vals_sym    = np.linalg.eigh(A)[0]       # symmetric (real, sorted)

# Matrix properties
print(f"det={np.linalg.det(A):.2f}, rank={np.linalg.matrix_rank(A)}")
print(f"cond={np.linalg.cond(A):.4f}, trace={np.trace(A):.2f}")
print(f"norm (Frobenius)={norm(A,'fro'):.4f}, norm (2)={norm(A,2):.4f}")


#| code-fold: false

import sympy as sym

A_s = sym.Matrix([[4,1],[2,3]])

print("Determinant:", A_s.det())
print("Inverse:\n", A_s.inv())
print("Eigenvalues:", A_s.eigenvals())
print("Eigenvectors:")
for val, mult, vecs in A_s.eigenvects():
    print(f"  λ={val}, multiplicity={mult}, v={vecs}")

rref_A, pivots = A_s.rref()
print("RREF:", rref_A)
P_d, D_d = A_s.diagonalize()
print(f"Diagonalization: P={P_d}, D={D_d}")
null_A = sym.Matrix([[1,2,3],[4,5,6],[7,8,9]]).nullspace()
print(f"Null space: {null_A}")


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m,'__version__',None)))
