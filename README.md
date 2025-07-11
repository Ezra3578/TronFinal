# TronFinal
Final Delivery of the System Sciences Foundations Project



Tron Multi-Agent Reinforcement Learning (PPO simulating MAPPO)

This project implements a multi-agent reinforcement learning (MARL) system for the game Tron, using Proximal Policy Optimization (PPO). A MAPPO-style setup is simulated by assigning one shared policy per team, enabling cooperation within teams and competition across teams. 

---

(-)Setup Instructions (Linux / WSL)

Clone the repository, create a Conda environment with Python 3.10.12, activate it, and install all required dependencies:

```bash
git clone https://github.com/Ezra3578/TronFinal.git && cd TronFinal
conda create -n tron-env python=3.10.12 -y && conda activate tron-env
pip install "ray[rllib]"==2.9.3
pip install pettingzoo==1.23.1
pip install gymnasium==0.29.1
pip install pygame==2.5.2
pip install matplotlib==3.8.4
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install torch
```

Warning: Make sure to use `pip` from within the Conda environment.


(-)How to Run

Test trained agents:

```bash
python testing.py
```
---

(-) License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
You are free to use, modify, and distribute this code, provided that any derivative work is also licensed under GPL-3.0 and clearly credits the original authors.

For full details, see the LICENSE file or visit https://www.gnu.org/licenses/gpl-3.0.html.
