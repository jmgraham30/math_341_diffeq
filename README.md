# MATH 341 Differential Equations Course Materials

[Course Website](https://math341diffeq.netlify.app/)

This project uses Python. Instructions for setting up the Python environment are provided below. 
The course materials include [Quarto](https://quarto.org/) notebooks that demonstrate concepts in differential equations, 
so having the correct environment is essential for running the code without issues.

## Python Environment Setup Using Conda

This project uses a conda environment called `diffeq_py` managed via [conda-forge](https://conda-forge.org/). Follow the steps below to reproduce the environment on your machine.

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) or [Miniforge](https://github.com/conda-forge/miniforge) (recommended) installed on your system.

### Option 1 — Create from the environment file (recommended)

A conda environment file is provided at `diffeq_py_environment.yml`. Before using it, you should **update the file** to pin the exact package versions used in this project:

```yaml
name: diffeq_py
channels:
  - conda-forge
dependencies:
  - python=3.14.4
  - numpy=2.4.3
  - scipy=1.17.1
  - sympy=1.14.0
  - matplotlib=3.10.8
  - jupyterlab
  - ipykernel
```

Then create and activate the environment:

```bash
conda env create -f diffeq_py_environment.yml
conda activate diffeq_py
```

### Option 2 — Create manually

```bash
conda create -n diffeq_py -c conda-forge \
  python=3.14.4 \
  numpy=2.4.3 \
  scipy=1.17.1 \
  sympy=1.14.0 \
  matplotlib=3.10.8 \
  jupyterlab \
  ipykernel
conda activate diffeq_py
```

### Register the Jupyter kernel

After activating the environment, register it as a Jupyter kernel so it appears in JupyterLab:

```bash
python -m ipykernel install --user --name diffeq_py --display-name "Python (diffeq_py)"
```

### Launch JupyterLab

```bash
jupyter lab
```

Select the **"Python (diffeq_py)"** kernel when opening notebooks.


## Alternative Setup with uv

If you use [uv](https://docs.astral.sh/uv/) for Python and package management, you can reproduce the environment as follows.

### Prerequisites

- `uv` installed. If not, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

**1. Clone the repository and navigate into it:**

```bash
git clone https://github.com/jmgraham30/math_341_diffeq.git
cd math_341_diffeq
```

**2. Create a virtual environment with the correct Python version:**

```bash
uv venv --python 3.14.4
```

> `uv` will automatically download Python 3.14.4 if it is not already available on your system.

**3. Activate the virtual environment:**

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**4. Install the required packages:**

```bash
uv pip install \
  numpy==2.4.3 \
  scipy==1.17.1 \
  sympy==1.14.0 \
  matplotlib==3.10.8 \
  jupyterlab \
  ipykernel
```

**5. Register the Jupyter kernel:**

```bash
python -m ipykernel install --user --name diffeq_py --display-name "Python (diffeq_py)"
```

**6. Launch JupyterLab:**

```bash
jupyter lab
```

Select the **"Python (diffeq_py)"** kernel when opening notebooks.

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg