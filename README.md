# Joint Optimization of Access Control and Resource Allocation for LEO Satellite IoT

*A DRL-Based Framework for Efficient Random Access and Resource Management in LEO Satellite Systems*

<p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue.svg"> <img src="https://img.shields.io/badge/DeepRL-DDPG%2FDQN-orange"> <img src="https://img.shields.io/badge/Status-Research%20Project-brightgreen"> <img src="https://img.shields.io/badge/Domain-LEO%20Satellite%20IoT-lightgrey"> </p>

## 📌 Overview

This repository provides an integrated simulation and optimization framework for **joint access-control and resource-allocation** in **LEO satellite IoT systems**.
The system aims to solve two key problems under massive device access:

1. achieving efficient random access during the access-control phase

2. enabling high-efficiency dynamic allocation of time–frequency resources

To address these challenges, the project incorporates:

* Deep Reinforcement Learning (DDPG / DQN)

* Classical optimization (DP, Bandit)

* Satellite IoT system modelling

* Joint long-term utility optimization guided by Lyapunov theory

The goal is to achieve **lower access delay, higher resource utilization, and better large-scale terminal adaptation**.

## 📂 Repository Structure
```
Joint-Optimization-of-Access-Control-and-Resource-Allocation-for-LEO-Satellite
│
├── SatelliteIoT_env.py           # LEO satellite IoT environment simulator
├── ControlCentre.py              # System controller & process manager
│
├── Pre_training.py               # Pre-training scripts for DRL agents
│
├── PositionOptimization_Bandit.py  # Bandit-based optimization baseline
├── PositionOptimization_DQN.py     # DQN-based optimization algorithm
├── PositionOptimization_DP.py      # Dynamic Programming (DP) baseline
├── PositionOptimization_DDPG.py    # DDPG-based DRL agent (core method)
│
└── README.md
```

## 🚀 Features
### 🔹 1. Joint Optimization Framework

Simultaneously optimizes ACB-based access control and time–frequency resource allocation, achieving long-term dynamic optimization.

### 🔹 2. DRL-Enhanced Satellite Decision Making

* Lightweight DRL agent suitable for on-board deployment

* Learns adaptive policies under dynamic traffic and channel variations

* Supports DDPG / DQN architectures

### 🔹 3. General Satellite IoT Simulation Environment

Includes modeling of:

* Terminal distribution & random access attempts

* Satellite coverage window

### 🔹 4. Multiple Comparison Baselines
| Method | Category             | Role                        |
| ------ | -------------------- | --------------------------- |
| DP     | Classic optimization | Strong analytical baseline  |
| Bandit | Online learning      | Lightweight alternative     |
| DQN    | Deep RL              | Discrete control            |
| DDPG   | Deep RL              | **Main proposed algorithm** |

## 🛠️ Usage
### ▶️ 1. Run DDPG (main method)
```
python PositionOptimization_DDPG.py
```

### ▶️ 2. Run DQN baseline
```
python PositionOptimization_DQN.py
```

### ▶️ 3. Run Bandit baseline
```
python PositionOptimization_Bandit.py
```

### ▶️ 4. Run Dynamic Programming (DP)
```
python PositionOptimization_DP.py
```

### ▶️ 5. Pre-training (optional)
```
python Pre_training.py
```

### ▶️ 6. Main control center simulation
```
python ControlCentre.py
```

## 📄 License

This project will be released under an open-source license after related confidential components are cleared.
