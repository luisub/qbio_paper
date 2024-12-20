# UQ Bio Paper

**Author:** Luis Aguilera

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## 📜 Description

<div style="display: flex; align-items: center;">
    <img src="docs/uqbio_2024_logo.png" alt="UQ Bio Logo" style="width: 200px; margin-right: 20px;">
    <div>
        This repository contains the code required to reproduce the figures and analyses presented in the UQ Bio paper.
    </div>
</div>

---

## 🛠️ Instructions

### 📘 Notebooks

1. **Source Code**  
   Access core code, model definitions, plots, and imports: [imports.py](https://github.com/luisub/qbio_paper/blob/main/src/imports.py)

2. **Simulating Gene Expression (Spatio-Temporal Dynamics)**  
   Use the notebook: [simulated_data.ipynb](https://github.com/luisub/qbio_paper/blob/main/notebooks/simulated_data.ipynb)

3. **Simulating Constitutive Models (Stochastic & Deterministic)**  
   Use the notebook: [constitutive.ipynb](https://github.com/luisub/qbio_paper/blob/main/notebooks/constitutive.ipynb)

4. **Parameter Estimation (Metropolis-Hastings Algorithm)**  
   Execute the Bash script: [runner.sh](https://github.com/luisub/qbio_paper/blob/main/notebooks/runner.sh)

5. **Metropolis-Hastings Algorithm Implementation**  
   Access the Python implementation: [parameter_estimation_constitutive.py](https://github.com/luisub/qbio_paper/blob/main/notebooks/parameter_estimation_constitutive.py)

---

## 🧪 Simulating Model Perturbations

### Constitutive Model

A constitutive model is defined as follows:

| Index | Reaction                                | Description                       | Reaction Rate               |
|-------|------------------------------------------|-----------------------------------|-----------------------------|
| $r_1$ | $ \xrightarrow[]{k_{r}} R_{n}$           | Active gene produces mRNA          | $k_{r}$                     |
| $r_2$ | $R_{n} \xrightarrow[]{k_{t}} R_{c}$      | mRNA transport to cytoplasm        | $k_{t} \cdot [R_{n}]$       |
| $r_3$ | $R_{c} \xrightarrow[]{k_{p}} R_{c} + P$  | Cytoplasmic mRNA produces protein  | $k_{p} \cdot [R_{c}]$       |
| $r_4$ | $R_{n} \xrightarrow[]{\gamma_{r}} \phi$  | Nuclear mRNA decay                 | $\gamma_{r} \cdot [R_{n}]$  |
| $r_5$ | $R_{c} \xrightarrow[]{\gamma_{r}} \phi$  | Cytoplasmic mRNA decay             | $\gamma_{r} \cdot [R_{c}]$  |
| $r_6$ | $P \xrightarrow[]{\gamma_p} \phi$         | Protein decay                      | $\gamma_p \cdot [P]$        |
* Notice that the model assumes the same decay rate for nuclear and cytoplasmic RNA.

### Spatio-Temporal Model (Gene State)

The spatio-temporal model includes additional reactions for gene state toggling:

| Index | Reaction                                    | Description          | Reaction Rate                    |
|-------|----------------------------------------------|----------------------|----------------------------------|
| $r_7$ | $G_{off} \xrightarrow[]{k_{on}} G_{on}$      | Gene activation      | $k_{on} \cdot [G_{off}]$         |
| $r_8$ | $G_{on} \xrightarrow[]{k_{off}} G_{off}$     | Gene deactivation    | $k_{off} \cdot [G_{on}]$         |

In the spatio-temporal model, $r_1$ depends on the gene state:

| Index | Reaction                                    | Description                 | Reaction Rate                |
|-------|----------------------------------------------|-----------------------------|------------------------------|
| $r_1*$ | $G_{on} \xrightarrow[]{k_{r}} R_{n} + G_{on}$ | Active gene produces mRNA   | $k_{r} \cdot [G_{on}]$       |

---

## 🔬 Inhibiting Parameters for Perturbations

To simulate perturbations (such as drug application), you can inhibit certain parameters by passing them into the `inhibited_parameters` dictionary. Set `apply_drug = True`, define `drug_application_time`, and adjust the `inhibition_constant` to scale the targeted parameter.

**Example: Inhibiting RNA Production**

```python
# Simulating inhibition in the RNA production
apply_drug = True
drug_application_time = 120
inhibition_constant = 0.1 
inhibited_parameters = {'kr': kr * inhibition_constant}
```

### Additional Example inhibiting protein production.

You can inhibit other parameters by adding them to the `inhibited_parameters` dictionary. For example:

```python
inhibited_parameters = {
    'k_p': k_p * inhibition_constant
}
```

This allows you to simulate different scenarios where the drug affects different steps during gene expression. Ensure the `apply_drug` flag is set to `True` and specify the `drug_application_time` to define when the inhibition takes effect during the simulation.

### Complete Parameter Example for the Spatio Temporal Model using in Appendix 2.

Here is the complete example of the parameter setup used during the optimization process:

```python
# ---------------------------------------------
# Model Parameters
# ---------------------------------------------
k_on = 0.5                 # Gene activation rate (per time unit)
k_off = 0.1                # Gene deactivation rate (per time unit)
k_r = 3.0                  # RNA production rate (per active gene, per time unit)
k_p = 0.9                  # Protein production rate (per mRNA, per time unit)
gamma_r = 0.08             # RNA degradation rate (nuclear and cytoplasmic, per time unit)
gamma_p = 0.45             # Protein degradation rate (per time unit)
transport_rate = 1.0       # Rate of RNA transport from nucleus to cytoplasm (per time unit)
diffusion_rate = 10.0      # Diffusion rate for RNA and proteins (arbitrary units)

# ---------------------------------------------
# Simulation Parameters
# ---------------------------------------------
total_simulation_time = 200          # Total duration of the simulation (time units)
number_of_simulated_cells = 200      # Number of cells to simulate
cytosol_diameter = 110               # Cytoplasm diameter (arbitrary spatial units)
nucleus_diameter = 60                # Nucleus diameter (arbitrary spatial units)
model_type = '2D'                    # Model geometry: '2D' or '3D'

# ---------------------------------------------
# Drug Inhibition Parameters
# ---------------------------------------------
apply_drug = True                     # Flag to enable or disable drug application
drug_application_time = 120           # Time to apply drug during simulation (time units)
inhibition_constant = 0.1             # Scaling factor to reduce the inhibited parameter
inhibited_parameters = {              # Parameters to inhibit and their scaled values
    'transport_rate': transport_rate * inhibition_constant
}
position_TS = 'random'                # Position of the transcription site (e.g., 'random' or specific coordinates)

```

By following this approach, you can simulate diverse perturbations and study their effects on the system.

---

## 💻 Code Installation

### Installation on a Local Computer

To set up the repository and its dependencies, follow these steps:

#### Prerequisites

We recommend installing [Anaconda](https://www.anaconda.com) to manage your Python environment.

#### Steps

1. **Clone the Repository**
   ```sh
   git clone https://github.com/luisub/qbio_paper.git
   ```

2. **Create and Activate a Virtual Environment**
   ```sh
   conda create -n qbio_paper_env python=3.10 -y
   conda activate qbio_paper_env
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Deactivate the Environment** (when done)
   ```sh
   conda deactivate
   ```

5. **Remove the Environment** (if no longer needed)
   ```sh
   conda env remove -n qbio_paper_env
   ```

---

## 📜 License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](https://opensource.org/licenses/BSD-3-Clause) file for more details.

---