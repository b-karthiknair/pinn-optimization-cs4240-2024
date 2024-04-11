# Reproduction of *Solving real‑world optimization tasks using physics‑informed neural computing*


The repository focus on code reproduction of the paper [Solving real‑world optimization tasks using physics‑informed neural computing](https://www.nature.com/articles/s41598-023-49977-3.pdf). For the task of The physics‑informed neural network (PINN) is implemented on the [Google JAX](https://github.com/google/jax) framework, from scratch (inspired by [this](https://github.com/tud-phi/ics-pa-sv?tab=readme-ov-file) repository). 

> *Link to blog post: [link](https://medium.com/@lohani.manan/9108df3b70f8)*


|Name|ID|Contact|Contribution|
|-|-|-|-|
|Karthik Biju Nair|5967333|k.bijunair@student.tudelft.nl|code replication, new data|
|Akansha Mukherjee|5973767|a.mukherjee-11@student.tudelft.nl|code reproduction,  code replication|
|Manan Lohani|5915821|m.j.lohani@student.tudelft.nl|hyperparams check, code reproduction|
|Bakul Jangley|6055826|bjangley@student.tudelft.nl|ablation study, code reproduction|

## Overview

Physics-informed neural networks (PINNs) are a class of machine learning models that incorporate physical laws or principles into their architecture, enabling them to effectively learn from limited data and perform tasks such as optimization and control in physical systems. In this work, we apply PINNs to three distinct optimization tasks, each with unique characteristics:

1. **Pendulum Swing-Up**: The task is to swing up a pendulum to reach a specified goal state with constrained external torques, in a given time period.

2. **Shortest-Time Path**: Given two points in space, the objective is to determine the shortest-time path connecting them while adhering to physical constraints.

3. **Minimal-Thrust Swingby Trajectory**: In this task, we aim to find a trajectory for a spacecraft that involves swingby maneuvers around celestial bodies while minimizing the required thrust.

## Repository Structure
- `data` - Contains code from the original repository
- `group_23` - Contains code re-implementation, hyperparams check, new data check, 

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/b-karthiknair/pinn-optimization-cs4240-2024.git
    cd pinn-optimization-cs4240-2024
    ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```