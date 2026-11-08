# ENVIRONMENT_AUDIT_PROTOCOL.md: Quy trình Kiểm định & Audit Môi trường Giả lập Ban đầu

> **LƯU Ý DÀNH CHO AGENT:** Đây là quy trình bắt buộc phải hoàn thành **trước khi viết bất kỳ mã nguồn chính thức nào khác**. Agent phải đọc hiểu tệp này, kiểm thử tài nguyên môi trường có sẵn, thu thập số liệu demo thực tế và lập báo cáo kết quả gửi Người dùng (User) xác nhận.

---

## 1. Mục tiêu của Audit Protocol (Protocol Objectives)
1. **Xác minh tính khả thi kỹ thuật:** Kiểm tra tính sẵn sàng của các tệp mô phỏng, dependencies, và API tương thích của môi trường (Môi trường đã tải sẵn ở thư mục "F/navsim" và "F:\navsim-data", chỉ cần đọc hiểu cách sử dụng môi trường xây dựng sẵn này mà thôi ).
2. **Trích xuất thông số môi trường ban đầu (Baseline Profiling):** Đo lường và định lượng các đặc trưng môi trường ($C_{topo}$, $\omega_{POMDP}$, $\sigma_{env}$, $\rho_{sig}$) từ tài nguyên có sẵn.
3. **Chạy thử nghiệm khói (Sanity / Smoke Test):** Thực thi vòng lặp tương tác ngẫu nhiên (Random Agent Step Loop) trên môi trường với dữ liệu/bước nhỏ để ghi nhận bằng chứng hoạt động thực tế.
4. **Báo cáo & Demo:** Kết xuất dữ liệu thực nghiệm sơ bộ (demo logs, render frame/video, bảng thông số) để trình Người dùng phê duyệt trước khi đưa task vào Sprint Backlog.

---

## 2. Quy trình Thực thi 4 Bước dành cho Agent (Agent Execution Steps)

### Bước 1: Khám phá & Kiểm tra Tài nguyên (Resource Discovery & Inspection)
Agent tiến hành kiểm tra cấu trúc tài nguyên môi trường trong dự án hoặc kho lưu trữ:
- **Kiểm tra định dạng file:** Kiểm tra tệp bản đồ, cấu trúc đồ thị ( có thể các định dạng khác nhau tùy vào dự án ).
- **Kiểm tra API chuẩn:** Kiểm tra xem môi trường đã tuân thủ chuẩn `gymnasium.Env` (các phương thức `.reset()`, `.step()`, `observation_space`, `action_space`) hay chưa.
- **Kiểm tra Phụ thuộc (Dependencies):** Kiểm tra các thư viện tương thích (`torch`, `gymnasium`, `numpy`, `pybullet`, `habitat-sim`, v.v.).

### Bước 2: Thực thi Sanity Check & Thu thập Dữ liệu Demo
Agent chạy một script Python nhỏ nghiệm thu tính năng cơ bản của môi trường (tối đa $N = 100$ đến $1000$ steps):

```python
# Mẫu logic Sanity Check Agent cần thực hiện
import time
import gymnasium as gym

def audit_environment(env_id_or_config):
    # 1. Khởi tạo & Đo thời gian reset
    start_time = time.time()
    env = gym.make(env_id_or_config)
    obs, info = env.reset(seed=42)
    reset_duration = time.time() - start_time
    
    # 2. Thu thập không gian quan sát & hành động
    obs_space = env.observation_space
    act_space = env.action_space
    
    # 3. Chạy Random Action Loop (Smoke Test)
    step_rewards = []
    step_times = []
    
    done = False
    truncated = False
    
    for _ in range(100):
        t0 = time.time()
        action = act_space.sample() # Random action
        next_obs, reward, done, truncated, info = env.step(action)
        step_times.append(time.time() - t0)
        step_rewards.append(reward)
        
        if done or truncated:
            obs, info = env.reset()

    env.close()
    
    # 4. Trích xuất Metrics Demo
    return {
        "reset_time_sec": reset_duration,
        "obs_space": str(obs_space),
        "action_space": str(act_space),
        "mean_step_fps": 1.0 / (sum(step_times) / len(step_times)),
        "reward_sparsity_ratio": step_rewards.count(0) / len(step_rewards),
        "min_reward": float(min(step_rewards)),
        "max_reward": float(max(step_rewards))
    }
```

### Bước 3: Định lượng Thuộc tính Môi trường (Environmental Profiling)
Agent trích xuất các chỉ số đặc trưng môi trường theo khung nghiên cứu đã định dạng tại `RESEARCH_PROTOCOL.md`:
- **Topological Complexity ($C_{topo}$):** Tính toán mật độ vật cản hoặc entropy đồ thị không gian.
- **Observability Factor ($\omega_{POMDP}$):** Xác định dạng không gian quan sát (Full state hay Partial image/sensor).
- **Reward Sparsity ($\rho_{sig}$):** Tỷ lệ bước chạy nhận được phần thưởng khác 0 trên tổng số bước.

### Bước 4: Lập Báo cáo Audit & Trình bày Bằng chứng Demo
Agent tạo tệp báo cáo audit `audit_reports/ENV_AUDIT_REPORT.md` với mẫu chuẩn bên dưới và gửi Người dùng phê duyệt.

---

## 3. Khung Báo cáo Mẫu (Audit Report Template Output)

```markdown
# AUDIT REPORT: Pre-flight Simulation Environment Check

- **Ngày thực hiện:** [YYYY-MM-DD]
- **Nhánh Git (Branch):** `feature/env-audit-baseline`
- **Môi trường Kiểm thử:** [Tên môi trường / Đường dẫn cấu hình]

### 1. Thông số Kỹ thuật & Không gian Trạng thái (Technical Specs)
- **Observation Space:** `Box(low=-inf, high=inf, shape=(84, 84, 3), dtype=uint8)`
- **Action Space:** `Discrete(5)`
- **Thời gian Reset trung bình:** `0.042s`
- **Tốc độ mô phỏng trung bình (Throughput):** `450 FPS`

### 2. Định lượng Thuộc tính Môi trường Ban đầu (Environmental Profile)
| Chỉ số (Property Metric) | Giá trị Định lượng | Ghi chú & Đánh giá |
| :--- | :--- | :--- |
| **Topological Complexity ($C_{topo}$)** | `0.68` | Mức độ phức tạp trung bình - cao |
| **Partial Observability ($\omega_{POMDP}$)** | `Image-based (84x84)` | Che khuất góc nhìn robot |
| **Reward Sparsity ($\rho_{sig}$)** | `94.2% zero rewards` | Tín hiệu phần thưởng rất thưa thớt (Signal Starved) |
| **Stochasticity ($\sigma_{env}$)** | `0.05` | Nhiễu động lực học nhỏ |

### 3. Minh chứng Thực nghiệm Demo (Evidence & Demo Logs)
- **Log chạy thử nghiệm 100 steps (Snippet):**
  ```text
  [Step 000] Obs Shape: (84,84,3) | Action: 2 | Reward: 0.000 | Done: False
  [Step 001] Obs Shape: (84,84,3) | Action: 0 | Reward: 0.000 | Done: False
  ...
  [Step 045] Obs Shape: (84,84,3) | Action: 1 | Reward: 1.000 | Done: True
  ```
- **File đính kèm:** `audit_reports/logs/random_step_test.csv`

### 4. Kết luận & Đề xuất Task tiếp theo cho Backlog
- [x] Môi trường hoạt động ổn định, đủ điều kiện kết nối với các thuật toán DRL.
- **Đề xuất Task:** Tích hợp class thuật toán DRL đầu tiên (PPO/DQN) để tiến hành Sanity Check trên môi trường này.
```

---

## 4. Bảng Kiểm Tra Hoàn Thành Audit (Definition of Done - Audit Phase)
Một phiên Audit môi trường tài nguyên được coi là hoàn tất (**Passed**) khi Agent đáp ứng đầy đủ các tiêu chí:
1. [ ] Không gặp lỗi Crash/Exception khi khởi tạo và chạy `reset()` / `step()`.
2. [ ] Đã trích xuất và bảng hóa được ít nhất 3 thuộc tính môi trường ban đầu.
3. [ ] Đã xuất file log CSV/Text minh chứng chạy thử nghiệm ngẫu nhiên thành công.
4. [ ] Đã commit báo cáo audit lên branch `feature/env-audit-*` trên GitHub.
5. [ ] Người dùng (User) xác nhận báo cáo và duyệt kế hoạch bước tiếp theo.