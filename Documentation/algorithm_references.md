# TÀI LIỆU THAM CHIẾU THUẬT TOÁN DEEP REINFORCEMENT LEARNING
## Algorithm References & Stable-Baselines3 Taxonomy (TASK-301)

- **Dự án:** Signal Starvation — Khảo sát tác động của môi trường đến DRL trong Physical AI
- **Framework:** Stable-Baselines3 (SB3) v2.9.2 (Raffin et al., *JMLR 2021*)
- **Môi trường mục tiêu:** `NavSim/GridNav-v0` (Observation: `Box(22,)`, Action: `Discrete(4)`)

---

## 1. BẢNG PHÂN LOẠI THUẬT TOÁN CỐT LÕI

| Thuật toán | Loại (Family) | Paper gốc | SB3 Class | Action Space | Trạng thái tương thích |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **PPO** | On-policy Policy Gradient | Schulman et al., 2017 | `PPO('MlpPolicy', env)` | `Discrete(4)` | Hoàn toàn tương thích ✅ |
| **DQN** | Off-policy Q-Learning | Mnih et al., 2015 | `DQN('MlpPolicy', env)` | `Discrete(4)` | Hoàn toàn tương thích ✅ |
| **A2C** | On-policy Actor-Critic | Mnih et al., 2016 | `A2C('MlpPolicy', env)` | `Discrete(4)` | Hoàn toàn tương thích ✅ |

---

## 2. CHI TIẾT THUẬT TOÁN & CƠ CHẾ ĐỐI PHÓ SIGNAL STARVATION

### 2.1. PPO (Proximal Policy Optimization)
- **Họ thuật toán:** On-policy Policy Gradient (Actor-Critic).
- **Paper gốc:** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- **Mục tiêu tối ưu hóa (Clipped Surrogate Objective):**
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
  trong đó $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$.
- **Hành vi dưới Signal Starvation:**
  - Khi phần thưởng 100% thưa thớt (sparse reward) và xác suất chạm đích ngẫu nhiên $\rho_{sig} \approx 0$ (`starved` tasks), ưu thế $\hat{A}_t \approx 0$.
  - Entropy bonus (hệ số $\text{ent\_coef}$) đóng vai trò then chốt trong việc duy trì thăm dò (exploration) để tránh chính sách suy thoái sớm thành random jitter quanh điểm xuất phát.
- **Khởi tạo chuẩn trong SB3:**
  ```python
  from stable_baselines3 import PPO

  model = PPO(
      policy="MlpPolicy",
      env=env,
      learning_rate=3e-4,
      n_steps=2048,
      batch_size=64,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=0.2,
      ent_coef=0.01,
      verbose=1,
      tensorboard_log="./tb_logs/ppo/",
  )
  ```

---

### 2.2. DQN (Deep Q-Network)
- **Họ thuật toán:** Off-policy Temporal-Difference Q-Learning.
- **Paper gốc:** Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533.
- **Hàm tổn thất Bellman:**
  $$L_i(\theta_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \right)^2 \right]$$
- **Hành vi dưới Signal Starvation:**
  - Nhờ **Replay Buffer ($\mathcal{D}$)**, nếu tác tử tình cờ chạm đích dù chỉ 1 lần duy nhất trong quá trình thăm dò $\epsilon$-greedy, transition nhận thưởng quý giá này được lưu lại và lấy mẫu huấn luyện nhiều lần.
  - Tuy nhiên, khi $\rho_{sig} < 10^{-4}$, xác suất để $\epsilon$-greedy chạm đích là cực nhỏ, dẫn đến Replay Buffer tràn ngập 100% transition có reward $= 0$, khiến mạng Q dự đoán $Q(s, a) \approx 0$ ở mọi trạng thái (Value Collapse).
- **Khởi tạo chuẩn trong SB3:**
  ```python
  from stable_baselines3 import DQN

  model = DQN(
      policy="MlpPolicy",
      env=env,
      learning_rate=1e-4,
      buffer_size=100_000,
      learning_starts=1000,
      batch_size=64,
      gamma=0.99,
      exploration_fraction=0.3,
      exploration_initial_eps=1.0,
      exploration_final_eps=0.05,
      target_update_interval=1000,
      verbose=1,
      tensorboard_log="./tb_logs/dqn/",
  )
  ```

---

### 2.3. A2C (Advantage Actor-Critic)
- **Họ thuật toán:** Đồng bộ (Synchronous) On-policy Actor-Critic.
- **Paper gốc:** Mnih, V., Badia, A. P., Mirza, M., et al. (2016). *Asynchronous methods for deep reinforcement learning*. ICML 2016.
- **Hành vi dưới Signal Starvation:**
  - A2C cập nhật gradient sau mỗi $N$ bước ngắn ($n\_steps = 5 \div 16$).
  - Khi không có tín hiệu reward, variance của gradient rất thấp nhưng policy drift nhanh chóng về trạng thái bế tắc nếu không có phần thưởng định hình (shaped reward).
- **Khởi tạo chuẩn trong SB3:**
  ```python
  from stable_baselines3 import A2C

  model = A2C(
      policy="MlpPolicy",
      env=env,
      learning_rate=7e-4,
      n_steps=16,
      gamma=0.99,
      gae_lambda=1.0,
      ent_coef=0.01,
      verbose=1,
      tensorboard_log="./tb_logs/a2c/",
  )
  ```

---

## 3. LÝ DO KHOA HỌC CHỌN STABLE-BASELINES3 CHO DỰ ÁN

1. **Tính tương thích chuẩn Gymnasium:**
   NavSim kế thừa trực tiếp từ `gymnasium.Env`, cho phép SB3 nhận diện ngay lập tức `observation_space=Box(22,)` và `action_space=Discrete(4)` mà không cần viết custom policy network hay adaptor phức tạp.
2. **Khả năng tích hợp trực tiếp với API đánh giá của NavSim:**
   Hàm `navsim.evaluate(model, env)` nhận trực tiếp bất kỳ object SB3 model nào có phương thức `.predict(obs, deterministic=True)`, cho phép trích xuất các chỉ số điều hướng chuyên sâu như **Success Rate** và **SPL (Success weighted by Path Length)**.
3. **Tính tái lập (Reproducibility) đã được cộng đồng kiểm định:**
   Mã nguồn SB3 đã được peer-reviewed và xuất bản trên JMLR (Raffin et al., 2021), loại bỏ rủi ro sai lệch thuật toán nội tại, giúp nghiên cứu tập trung hoàn toàn vào biến số môi trường ($\rho_{sig}, C_{topo}$).
