# RL
# 🧠 Taxi Reinforcement Learning Agent

This repository contains a simple implementation of a **Q-learning agent** trained to solve the **Taxi-v3 environment** from [Gymnasium](https://gymnasium.farama.org/). It was developed as part of the *Reinforcement Learning* course project at the Faculty of Computer Engineering, Fall 1403.

---

## 🚕 Project Overview

The Taxi environment is a discrete OpenAI Gym task where a taxi must learn to pick up and drop off passengers at the correct locations in the shortest number of steps. The agent uses **Q-learning**, a model-free reinforcement learning algorithm, to learn an optimal policy.

---

## 📂 Files

- `main.py` — Implementation of the `QlearningAgent` class using Q-table-based learning.
- `RLProject.pdf` — Project description and course instructions (in Persian).

---

## 🔧 Features

- **Environment**: `Taxi-v3` from Gymnasium
- **Algorithm**: Q-learning
- **Action selection**: Combination of exploration (under-explored actions) and exploitation (greedy policy)
- **Custom logic**: Frequency-based exploration strategy
- **Test mode**: Runs trained policy with visual rendering

---

## 🛠️ Dependencies 

- Python 3.7+
- `gymnasium`
- `numpy`

Install dependencies and Running:

```bash
pip install gymnasium numpy
python main.py

