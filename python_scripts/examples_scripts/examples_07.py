#| label: setup
#| code-fold: true
#| code-summary: "Show the code"

# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| label: ex1-numpy

A = sym.Matrix([[2, -1], [3, 4]])
B = sym.Matrix([[0, 5], [-2, 1]])
x = sym.Matrix([1, -3])

display(Math(r'A + B = ' + sym.latex(A + B)))
display(Math(r'B - 2A = ' + sym.latex(B - 2*A)))
display(Math(r'AB = ' + sym.latex(A * B)))
display(Math(r'BA = ' + sym.latex(B * A)))
display(Math(r'A^2 = ' + sym.latex(A**2)))
display(Math(r'B\mathbf{x} = ' + sym.latex(B * x)))
display(Math(r'\det A = ' + sym.latex(A.det())))
display(Math(r'A^{-1} = ' + sym.latex(A.inv())))


#| label: ex2-sympy

b = sym.Matrix([3, -1])

# Method (a): inverse matrix
x_inv = A.inv() * b
display(Math(r'\mathbf{x} = A^{-1}\mathbf{b} = ' + sym.latex(x_inv)))

# Method (b): Cramer's rule — replace columns manually
A1 = A.copy(); A1[:, 0] = b
A2 = A.copy(); A2[:, 1] = b
x1_cramer = A1.det() / A.det()
x2_cramer = A2.det() / A.det()
display(Math(r"x_1 = \frac{\det A_1}{\det A} = \frac{"
            + sym.latex(A1.det()) + r"}{"
            + sym.latex(A.det()) + r"} = "
            + sym.latex(x1_cramer)))
display(Math(r"x_2 = \frac{\det A_2}{\det A} = \frac{"
            + sym.latex(A2.det()) + r"}{"
            + sym.latex(A.det()) + r"} = "
            + sym.latex(x2_cramer)))


#| label: fig-ex2
#| fig-cap: "Geometric interpretation of $A\\mathbf{x} = \\mathbf{b}$. Each equation defines a line; the unique solution $(1,-1)$ is their intersection (red dot)."
#| code-fold: true
#| code-summary: "Show the code"

x1_vals = np.linspace(-2, 4, 300)

# Line 1: 2x1 - x2 = 3  =>  x2 = 2x1 - 3
line1 = 2*x1_vals - 3

# Line 2: 3x1 + 4x2 = -1  =>  x2 = (-1 - 3x1)/4
line2 = (-1 - 3*x1_vals) / 4

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x1_vals, line1, color='steelblue', lw=2,
        label=r'$2x_1 - x_2 = 3$')
ax.plot(x1_vals, line2, color='darkorange', lw=2,
        label=r'$3x_1 + 4x_2 = -1$')
ax.plot(1, -1, 'o', color='tomato', ms=9, zorder=5,
        label=r'Solution $(1,-1)$')
ax.axhline(0, color='gray', lw=0.7)
ax.axvline(0, color='gray', lw=0.7)
ax.set_xlim(-2, 4)
ax.set_ylim(-5, 4)
ax.set_xlabel(r'$x_1$', fontsize=13)
ax.set_ylabel(r'$x_2$', fontsize=13)
ax.set_title(r'$A\mathbf{x} = \mathbf{b}$: intersection of two lines', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


#| label: ex3-sympy

A3 = sym.Matrix([[1, 2], [3, 2]])
lam = sym.Symbol('lambda')

char_poly = A3.charpoly(lam)
display(Math(r'\text{Characteristic polynomial: }\quad'
            + sym.latex(sym.Eq(char_poly.as_expr(), 0))))

eigvals = sym.solve(char_poly.as_expr(), lam)
display(Math(r'\text{Eigenvalues: }\quad \lambda = ' + sym.latex(eigvals)))

for ev in eigvals:
    evecs = (A3 - ev*sym.eye(2)).nullspace()
    display(Math(r'\lambda = ' + sym.latex(ev)
                 + r',\quad \mathbf{v} = ' + sym.latex(evecs[0])))


#| label: fig-ex3
#| fig-cap: "Eigenvectors of $A$ plotted from the origin (scaled for visibility). The vector $\\mathbf{v}_1 = (2,3)^T$ corresponds to $\\lambda_1 = 4$ and $\\mathbf{v}_2 = (-1,1)^T$ to $\\lambda_2 = -1$. Arrows show how $A$ maps each eigenvector to a scalar multiple of itself."
#| code-fold: true
#| code-summary: "Show the code"

A3_np = np.array([[1, 2], [3, 2]], dtype=float)
v1 = np.array([2, 3], dtype=float)
v2 = np.array([-1, 1], dtype=float)
v1n = v1 / np.linalg.norm(v1)
v2n = v2 / np.linalg.norm(v2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='gray', lw=0.7)
ax.axvline(0, color='gray', lw=0.7)

# Eigenvectors
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
          color='steelblue', width=0.012, label=r'$\mathbf{v}_1=(2,3)^T,\;\lambda_1=4$')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
          color='tomato', width=0.012, label=r'$\mathbf{v}_2=(-1,1)^T,\;\lambda_2=-1$')

# A*v1 and A*v2 (should be lambda*v)
Av1 = A3_np @ v1   # = 4*v1
Av2 = A3_np @ v2   # = -1*v2
ax.quiver(0, 0, Av1[0], Av1[1], angles='xy', scale_units='xy', scale=1,
          color='steelblue', alpha=0.35, width=0.012, linestyle='dashed')
ax.quiver(0, 0, Av2[0], Av2[1], angles='xy', scale_units='xy', scale=1,
          color='tomato', alpha=0.35, width=0.012, linestyle='dashed')

# Annotations
ax.annotate(r'$\mathbf{v}_1$', xy=v1, xytext=(2.2, 2.7), fontsize=12,
            color='steelblue')
ax.annotate(r'$A\mathbf{v}_1 = 4\mathbf{v}_1$', xy=Av1,
            xytext=(8.2, 11.5), fontsize=10, color='steelblue', alpha=0.7)
ax.annotate(r'$\mathbf{v}_2$', xy=v2, xytext=(-1.4, 1.1), fontsize=12,
            color='tomato')
ax.annotate(r'$A\mathbf{v}_2 = -\mathbf{v}_2$', xy=Av2,
            xytext=(0.8, -1.4), fontsize=10, color='tomato', alpha=0.7)

ax.set_xlim(-3, 12)
ax.set_ylim(-3, 13)
ax.set_aspect('equal')
ax.set_xlabel(r'$v_1$', fontsize=13)
ax.set_ylabel(r'$v_2$', fontsize=13)
ax.set_title(r'Eigenvectors of $A$  (entries: row 1 = [1, 2], row 2 = [3, 2])',
             fontsize=12)
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()


#| label: ex4-sympy

A4 = sym.Matrix([[-3, 1], [0, -3]])

char_poly4 = A4.charpoly(lam)
display(Math(r'\text{Characteristic polynomial: }\quad'
            + sym.latex(sym.Eq(char_poly4.as_expr(), 0))))

display(Math(r'\text{Eigenvalues: }' + sym.latex(A4.eigenvals())))

for ev, mult, evecs in A4.eigenvects():
    display(Math(r'\lambda = ' + sym.latex(ev)
                 + r'\;(\text{multiplicity } ' + str(mult) + r'),\quad'
                 + r'\mathbf{v} = ' + sym.latex(evecs[0])))


#| label: ex5-sympy

A5 = sym.Matrix([[1, -5], [2, -3]])

char_poly5 = A5.charpoly(lam)
display(Math(r'\text{Characteristic polynomial: }\quad'
            + sym.latex(sym.Eq(char_poly5.as_expr(), 0))))

roots5 = sym.solve(char_poly5.as_expr(), lam)
display(Math(r'\text{Eigenvalues: }\quad \lambda = ' + sym.latex(roots5)))

for ev, mult, evecs in A5.eigenvects():
    display(Math(r'\lambda = ' + sym.latex(ev)
                 + r',\quad \mathbf{v} = ' + sym.latex(evecs[0])))


#| label: ex6-sympy

beta = sym.Symbol('beta', real=True)

A6 = sym.Matrix([[-1, beta], [-1, -1]])
char_poly6 = A6.charpoly(lam)
display(Math(r'\text{Characteristic polynomial: }\quad'
            + sym.latex(sym.Eq(char_poly6.as_expr(), 0))))

roots6 = sym.solve(char_poly6.as_expr(), lam)
display(Math(r'\lambda(\beta) = ' + sym.latex(roots6)))


#| label: fig-ex6
#| fig-cap: "Real part (solid) and imaginary part (dashed) of the eigenvalues $\\lambda(\\beta) = -1 \\pm \\sqrt{-\\beta}$ as functions of $\\beta$. For $\\beta < 0$ the eigenvalues are real; for $\\beta > 0$ they are complex conjugates with real part $-1$. The transition occurs at $\\beta = 0$."
#| code-fold: true
#| code-summary: "Show the code"

beta_vals = np.linspace(-4, 4, 800)

# Eigenvalue formula: -1 ± sqrt(-beta)
# For beta < 0: sqrt(-beta) is real
# For beta > 0: sqrt(-beta) = i*sqrt(beta) -- complex

real_part_plus  = np.where(beta_vals <= 0,
                           -1 + np.sqrt(np.maximum(-beta_vals, 0)),
                           -1 * np.ones_like(beta_vals))
real_part_minus = np.where(beta_vals <= 0,
                           -1 - np.sqrt(np.maximum(-beta_vals, 0)),
                           -1 * np.ones_like(beta_vals))
imag_part = np.where(beta_vals >= 0,
                     np.sqrt(np.maximum(beta_vals, 0)),
                     np.zeros_like(beta_vals))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: real parts
ax = axes[0]
ax.plot(beta_vals, real_part_plus,  color='steelblue', lw=2,
        label=r'$\operatorname{Re}(\lambda_1)$')
ax.plot(beta_vals, real_part_minus, color='tomato', lw=2,
        label=r'$\operatorname{Re}(\lambda_2)$')
ax.axvline(0, color='gray', lw=0.8, ls='--')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.axvline(-1, color='black', lw=0.8, ls=':')
ax.set_xlabel(r'$\beta$', fontsize=13)
ax.set_ylabel(r'$\operatorname{Re}(\lambda)$', fontsize=13)
ax.set_title('Real parts of eigenvalues', fontsize=11)
ax.legend(fontsize=9)
ax.annotate(r'$\beta=-1$', xy=(-1, 0), xytext=(-3, 0.7), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'))

# Right: imaginary parts
ax = axes[1]
ax.plot(beta_vals,  imag_part, color='steelblue', lw=2,
        label=r'$\operatorname{Im}(\lambda_1) = +\sqrt{\beta}$')
ax.plot(beta_vals, -imag_part, color='tomato', lw=2,
        label=r'$\operatorname{Im}(\lambda_2) = -\sqrt{\beta}$')
ax.axvline(0, color='gray', lw=0.8, ls='--')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel(r'$\beta$', fontsize=13)
ax.set_ylabel(r'$\operatorname{Im}(\lambda)$', fontsize=13)
ax.set_title('Imaginary parts of eigenvalues', fontsize=11)
ax.legend(fontsize=9)

plt.suptitle(r'Eigenvalues of $A(\beta)$ vs.\ $\beta$', fontsize=12)
plt.tight_layout()
plt.show()


#| label: session-info

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}'
                for m in globals().values()
                if getattr(m, '__version__', None)))
