# A2C Implementation from Scratch using PyTorch

<p align="center">
  <img src="https://logos-world.net/wp-content/uploads/2024/08/OpenAI-Logo.png" alt="OpenAI Logo" width="130"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="PyTorch Logo" width="80"/>
</p>

---

## 🧠 Introduction

This repository contains a custom, from-scratch implementation of the **Advantage Actor-Critic (A2C)** algorithm using **PyTorch**.

The structure and logic follow principles similar to **OpenAI’s foundational implementations**, but the entire code was written manually as a way to **deeply understand and internalize Reinforcement Learning (RL) concepts**. This process was a **huge confidence booster** and a **steep learning point** for me in the field of RL.

---

## 📜 Research Papers Referenced

- **Reinforcement Learning: An Introduction** by Sutton & Barto  
  📖 [http://incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html)

- **High-Dimensional Continuous Control Using Generalized Advantage Estimation**  
  📄 Schulman et al. (2015) – [https://arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `Actor_Critic.py` | Contains the class definition of the A2C algorithm with GAE and entropy regularization. |
| `Lunar_Lander_A2C.ipynb` | Jupyter Notebook to train the A2C model on OpenAI Gym’s LunarLander-v2. Includes visualizations and final rendered agent results. |
| `play_lunar.py` | Lets you play the LunarLander environment manually using keyboard controls. Great for experiencing how difficult the task is. |

---

## ⚙️ Features of This Implementation

- Actor-Critic networks defined using PyTorch's `nn.Sequential`
- Generalized Advantage Estimation (GAE) for more stable training
- Entropy bonus to encourage exploration
- Supports multiple environments for parallel learning (`n_envs`)
- Lightweight, educational, and customizable

---

## 🚀 What's Next

📌 **Proximal Policy Optimization (PPO)** —  
Coming up next is a from-scratch PPO implementation using PyTorch, building upon the foundations laid here.

---

## 🤝 Acknowledgments

Big thanks to **OpenAI** for its open-source contributions and foundational research, which inspired this learning journey.  
Also to the PyTorch community for making deep learning so accessible and powerful.

---

## 🧠 Author

**Achintya Sharma**  
🔗 [LinkedIn](https://www.linkedin.com/in/achintya47)

---
