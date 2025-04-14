# 🧩 Simulated Annealing for 3-SAT

This repository contains an implementation of the **Simulated Annealing (SA)** algorithm applied to the **3-SAT** problem. The goal is to minimize the number of unsatisfied clauses by exploring the space of Boolean variable assignments using a probabilistic search strategy.

We tested multiple configurations of the `SA_max` parameter (1, 5, and 10) across CNF instances with 20, 100, and 250 variables, evaluating how different settings influence convergence and final solution quality.

---

## ⚙️ Installation

Make sure you have Python 3 installed, then run:

```bash
pip install matplotlib seaborn pandas joblib
