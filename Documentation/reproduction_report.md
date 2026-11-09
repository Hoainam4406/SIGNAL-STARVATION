# Stable-Baselines3 Benchmark Reproduction Report

- **Date:** 2026-11-09 21:43:28
- **Framework:** Stable-Baselines3 v2.3.3 + Gymnasium
- **Compute Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA: True)
- **Threshold Criterion:** Deviation $\le 1.5\%$ compared to official SB3 documentation targets.
- **Formula:** 
  $$\text{Deviation} = \frac{|\text{Actual} - \text{Target}|}{\text{Target}} \times 100\%$$

---

## 1. Executive Summary

| Algorithm | Benchmark | Hardware | Timesteps | Target (SB3 Docs) | Actual Mean Reward | Std Across Seeds | Deviation (%) | DoD Status ($\le 1.5\%$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A2C** | `CartPole-v1` | `cuda` | 500,000 | **500.0** | **499.53** | $\pm 1.05$ | **0.09%** | ✅ **PASS** |
| **DQN** | `CartPole-v1` | `cuda` | 50,000 | **500.0** | **499.34** | $\pm 1.48$ | **0.13%** | ✅ **PASS** |
| **PPO** | `CartPole-v1` | `cuda` | 100,000 | **500.0** | **496.74** | $\pm 4.66$ | **0.65%** | ✅ **PASS** |

---

## 2. Detailed Per-Seed Results

| Algorithm | Benchmark | Seed | Device | Timesteps | Mean Reward (20 eps) | Std Reward | Deviation (%) | Wall Time (s) | FPS | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A2C | `CartPole-v1` | 10 | `cuda` | 500,000 | 500.00 | $\pm 0.00$ | 0.00% | 1493.36s | 334.8 | PASS ✅ |
| A2C | `CartPole-v1` | 20 | `cuda` | 500,000 | 497.65 | $\pm 10.24$ | 0.47% | 1488.26s | 336.0 | PASS ✅ |
| A2C | `CartPole-v1` | 30 | `cuda` | 500,000 | 500.00 | $\pm 0.00$ | 0.00% | 1479.87s | 337.9 | PASS ✅ |
| A2C | `CartPole-v1` | 40 | `cuda` | 500,000 | 500.00 | $\pm 0.00$ | 0.00% | 1466.43s | 341.0 | PASS ✅ |
| A2C | `CartPole-v1` | 50 | `cuda` | 500,000 | 500.00 | $\pm 0.00$ | 0.00% | 1508.18s | 331.5 | PASS ✅ |
| DQN | `CartPole-v1` | 10 | `cuda` | 50,000 | 496.70 | $\pm 8.72$ | 0.66% | 828.29s | 60.4 | PASS ✅ |
| DQN | `CartPole-v1` | 20 | `cuda` | 50,000 | 500.00 | $\pm 0.00$ | 0.00% | 838.78s | 59.6 | PASS ✅ |
| DQN | `CartPole-v1` | 30 | `cuda` | 50,000 | 500.00 | $\pm 0.00$ | 0.00% | 879.78s | 56.8 | PASS ✅ |
| DQN | `CartPole-v1` | 40 | `cuda` | 50,000 | 500.00 | $\pm 0.00$ | 0.00% | 782.85s | 63.9 | PASS ✅ |
| DQN | `CartPole-v1` | 50 | `cuda` | 50,000 | 500.00 | $\pm 0.00$ | 0.00% | 775.52s | 64.5 | PASS ✅ |
| PPO | `CartPole-v1` | 10 | `cuda` | 100,000 | 500.00 | $\pm 0.00$ | 0.00% | 2535.34s | 39.4 | PASS ✅ |
| PPO | `CartPole-v1` | 20 | `cuda` | 100,000 | 489.15 | $\pm 19.92$ | 2.17% | 2508.13s | 39.9 | FAIL ❌ |
| PPO | `CartPole-v1` | 30 | `cuda` | 100,000 | 495.35 | $\pm 20.27$ | 0.93% | 2501.93s | 40.0 | PASS ✅ |
| PPO | `CartPole-v1` | 40 | `cuda` | 100,000 | 499.20 | $\pm 3.49$ | 0.16% | 2521.8s | 39.7 | PASS ✅ |
| PPO | `CartPole-v1` | 50 | `cuda` | 100,000 | 500.00 | $\pm 0.00$ | 0.00% | 2500.27s | 40.0 | PASS ✅ |

---

## 3. Reference Documentation & Methodology

1. **Official SB3 Documentation & Benchmarks:**
   - [Stable-Baselines3 Official Documentation](https://stable-baselines3.readthedocs.io/en/master/)
   - [RL Baselines3 Zoo Benchmark Results & Pretrained Models](https://github.com/DLR-RM/rl-baselines3-zoo)
   - [SB3 Model Zoo CartPole-v1 Benchmark](https://huggingface.co/sb3)
   - Paper: Raffin et al., *Stable-Baselines3: Reliable Reinforcement Learning Implementations*, JMLR 2021.

2. **Hyperparameter Configurations:**
   - **PPO (`configs/ppo_default.yaml`):** Schulman et al. (2017) clipped objective with GAE ($\lambda=0.95$), learning rate $3.0 \times 10^{-4}$.
   - **DQN (`configs/dqn_default.yaml`):** Mnih et al. (2015) with double Q-updates, target update interval 10, batch size 64, learning rate $2.3 \times 10^{-3}$, network architecture `[256, 256]`.
   - **A2C (`configs/a2c_default.yaml`):** Mnih et al. (2016) synchronous advantage actor-critic baseline with 8 parallel environments, learning rate $7.0 \times 10^{-4}$.

3. **Evaluation Protocol:**
   - Standard Gymnasium `CartPole-v1` with max episodic horizon = 500 steps.
   - Evaluated using deterministic policy across 20 evaluation episodes per seed.
   - 5 independent random seeds (`[10, 20, 30, 40, 50]`) evaluated per algorithm.
   - Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA acceleration enabled).

---

## 4. Definition of Done (DoD) Verification

- [x] **PPO:** Deviation $\le 1.5\%$ on `CartPole-v1` (100K steps, 5 seeds).
- [x] **DQN:** Deviation $\le 1.5\%$ on `CartPole-v1` (50K steps, 5 seeds).
- [x] **A2C:** Deviation $\le 1.5\%$ on `CartPole-v1` (500K steps, 5 seeds).
- [x] Output stored at `results/reproduction/benchmark_results.csv`.
- [x] Comprehensive documentation generated at `Documentation/reproduction_report.md`.
