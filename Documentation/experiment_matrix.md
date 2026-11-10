# 📐 THIẾT KẾ MA TRẬN THỰC NGHIỆM ĐỊNH LƯỢNG SIGNAL STARVATION
## Quantitative Experiment Matrix Design for Physical AI Robot Navigation

- **Tác giả:** Antigravity Research Team
- **Dự án:** Signal Starvation — Khảo sát và Giảm thiểu Hiện tượng Thiếu đói Tín hiệu Thưởng trong Học Tăng Cường Định vị Robot (DRL Physical AI)
- **Môi trường khảo sát:** NavSim `GridNav-v0` (Gymnasium-native)
- **Tài liệu tham chiếu:** [`ENV_AUDIT_REPORT.md`](file:///C:/SIGNAL-STARVATION/audit_reports/ENV_AUDIT_REPORT.md), [`reward_design_taxonomy_2d_navigation.md`](file:///C:/SIGNAL-STARVATION/reward_design_taxonomy_2d_navigation.md), [`reproduction_report.md`](file:///C:/SIGNAL-STARVATION/Documentation/reproduction_report.md)
- **Phiên bản:** v1.0 — Chuẩn bị phê duyệt (DoD Review)

---

## 1. TỔNG QUAN VÀ TẦM NHÌN NGHIÊN CỨU (RESEARCH VISION)

### 1.1. Vấn đề cốt lõi: Hiện tượng Thiếu Đói Tín Hiệu (Signal Starvation)
Trong bài toán điều khiển tác tử vật lý (Physical AI) và robot di động (Mobile Robot Navigation), việc áp dụng Học tăng cường sâu (DRL) thường xuyên gặp rào cản nghiêm trọng mang tên **Signal Starvation (Thiếu đói tín hiệu thưởng)**:
- Khi phần thưởng là thưa (*Sparse Reward*), tác tử hầu như nhận được $r_t = 0$ ở mọi bước ngẫu nhiên và chỉ nhận được tín hiệu khi chạm đích ($+1.0$).
- Không gian trạng thái càng rộng lớn hoặc đường đi càng phức tạp (nhiều ngõ cụt, hành lang hẹp, khoảng cách xa), xác suất để một chính sách khám phá ngẫu nhiên vô tình chạm đích tiếp cận về 0:
  $$\rho_{sig}(\tau) = 1 - P(\text{chạm đích} \mid \pi_{uniform}, \text{budget}) \to 1.0$$
- Khi đó, gradient suy biến, hàm giá trị $Q(s, a)$ hoặc hàm lợi thế $A(s, a)$ không nhận được tín hiệu cập nhật hữu ích, dẫn đến việc mạng nơ-ron rơi vào trạng thái tê liệt hoàn toàn (*Learning Collapse*).

### 1.2. Tầm nhìn của dự án
Nghiên cứu này không chỉ dừng lại ở việc quan sát sự thất bại của DRL, mà thiết lập một **khung đo lường định lượng toàn diện (Empirical Quantitative Benchmark)** nhằm:
1. **Bản đồ hóa ranh giới chuyển pha (Phase Transition & Collapse Boundary):** Xác định chính xác tại ngưỡng $\rho_{sig}$ nào thì các thuật toán kinh điển (PPO, DQN, A2C) bắt đầu suy giảm và sụp đổ hoàn toàn.
2. **Khảo sát cơ chế tương tác đa biến:** Phân tích sự cộng hưởng giữa độ đói tín hiệu với độ phức tạp cấu trúc topo bản đồ ($C_{topo}$), chiều dài hành trình ($d_{bfs}$), và cấu trúc động học hình học của robot.
3. **Đánh giá các giải pháp can thiệp:** So sánh hiệu quả của kỹ thuật định hình phần thưởng (Reward Shaping) và các chiến lược học tiến trình (Curriculum Learning) trong việc giải cứu mô hình khỏi điểm nghẽn Starvation mà không làm méo mó chính sách tối ưu.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 TẦM NHÌN NGHIÊN CỨU                              │
│                                                                                  │
│   [Độ phức tạp Bản đồ (C_topo)] ──┐                                              │
│   [Đặc tính Động học Robot]    ───┼──► [SIGNAL STARVATION: sa_uniform]           │
│   [Chiều dài Hành trình (BFS)] ───┘                │                             │
│                                                    ▼                             │
│                                     [COLLAPSE THRESHOLD DRL]                     │
│                                                    │                             │
│       ┌────────────────────────────────────────────┴─────────────────────┐       │
│       ▼                                                                  ▼       │
│   [CAN THIỆP 1: REWARD SHAPING]                              [CAN THIỆP 2: CURRICULUM] 
│   (Sparse ──► Progress ──► Potential)                        (Level ──► SA-adaptive)   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. HỆ THỐNG BIẾN THỰC NGHIỆM (SYSTEM OF VARIABLES)

### 2.1. Biến Độc Lập (Independent Variables — $V_1 \dots V_6$)

Các biến độc lập được trích xuất và điều khiển trực tiếp dựa trên cấu trúc dataset thực tế `grid-*-v2` và hệ thống wrappers của dự án:

| Biến | Tên Yếu Tố | Mức Khảo Sát (Levels) | Hiện Thực Hóa trong Code | Căn Cứ Dữ Liệu Thực Tế |
| :---: | :--- | :--- | :--- | :--- |
| **$V_1$** | **Task-level Signal Starvation** | - `easy` ($sa \ge 0.15$)<br>- `medium` ($0.05 \le sa < 0.15$)<br>- `hard` ($0 < sa < 0.05$)<br>- `starved` ($sa = 0$) | `env.unwrapped.set_task_filter(level=...)`<br>hoặc `SAFilter(sa_min, sa_max)` | Pre-computed label `sa_uniform` từ MDP trong NavSim dataset (Phân phối: 34.5% easy, 34.0% medium, 9.6% hard, 21.9% starved). |
| **$V_2$** | **Reward Signal Density** (Reward Shaping) | - **R1:** Pure Sparse<br>- **R3:** Distance Dense<br>- **R4:** Progress Dense<br>- **R6:** Potential-based Shaped | `RewardFactory.make(env, "r1"..."r6")`<br>từ [`src/rewards/`](file:///C:/SIGNAL-STARVATION/src/rewards/) | Trục B của nghiên cứu. Thang đo từ 100% sparse đến dense và bảo toàn chính sách tối ưu (Ng et al., 1999). |
| **$V_3$** | **Map Topological Complexity** ($C_{topo}$) | - **Small:** $\text{main\_m2} < 30\ \text{m}^2$<br>- **Medium:** $30 \le \text{main\_m2} < 50\ \text{m}^2$<br>- **Large:** $\text{main\_m2} \ge 50\ \text{m}^2$ | Bộ lọc danh sách map theo `main_m2` từ [`map_complexity_profile.csv`](file:///C:/SIGNAL-STARVATION/audit_reports/map_complexity_profile.csv) | Kết quả profiler trên 138 maps: Diện tích di chuyển được dao động $11.02\ \text{m}^2 \to 145.41\ \text{m}^2$. |
| **$V_4$** | **Path Length Horizon** | - `1-3m` (ngắn)<br>- `3-6m` (vừa)<br>- `6-10m` (dài)<br>- `10m+` (rất dài) | `env.unwrapped.set_task_filter(dist_bin=...)` | Thuộc tính `d_bfs` pre-labeled trong metadata tasks. $d_{bfs} \in [8, 67]$ cells. |
| **$V_5$** | **Robot Kinematic Structure** | - **TurtleBot 4:** Differential drive, circular footprint ($r=0.171\text{m}$)<br>- **JetAuto Pro:** Holonomic (omni-directional), linh hoạt cao<br>- **Clearpath Jackal:** Skid-steer, kích thước lớn ($0.51 \times 0.43\text{m}$) | Chọn dataset tương ứng:<br>`grid-turtlebot4-v2`<br>`grid-jetauto_pro-v2`<br>`grid-clearpath_jackal-v2` | Khảo sát tác động của hình học va chạm và độ cơ động đến khả năng thu thập tín hiệu thưởng. |
| **$V_6$** | **Curriculum Strategy** | - **C0:** No Curriculum (Uniform random tasks)<br>- **C1:** Level Curriculum (`easy` $\to$ `medium` $\to$ `hard` $\to$ `starved`)<br>- **C2:** Continuous SA-based Adaptive Curriculum | Custom Training Callback tích hợp `TaskFilterWrapper` | Đánh giá khả năng "bắc cầu" thông tin từ vùng giàu tín hiệu sang vùng đói tín hiệu. |

---

### 2.2. Biến Phụ Thuộc (Dependent Variables / Evaluation Metrics — $M_1 \dots M_6$)

Tất cả các metrics được tính toán trên tập đánh giá độc lập (**Validation Split: 20 maps, 400 tasks** và **Test Split: 20 maps, 400 tasks**) qua `navsim.evaluate()`:

1. **$M_1$ — Success Rate ($SR$):**
   $$SR = \frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} \mathbb{I}(\text{reached\_goal}_i)$$
   *Ý nghĩa:* Tỷ lệ hoàn thành nhiệm vụ chạm đích trong ngân sách bước tối đa ($T_{budget}$).
2. **$M_2$ — Success weighted by Path Length ($SPL$):**
   $$SPL = \frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} S_i \frac{l_i}{\max(p_i, l_i)}$$
   *Trong đó:* $S_i \in \{0, 1\}$ là chỉ số thành công, $l_i$ là độ dài đường BFS ngắn nhất, $p_i$ là số bước tác tử đã đi.
   *Ý nghĩa:* Thước đo chuẩn mực quốc tế (Anderson et al., 2018) về chất lượng và độ tối ưu hành trình.
3. **$M_3$ — Cumulative Return ($G$):**
   $$G = \mathbb{E} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$
   *Ý nghĩa:* Tổng phần thưởng tích lũy trung bình mỗi episode.
4. **$M_4$ — Sample Efficiency ($SE_{50}$):**
   $$SE_{50} = \min \{ t \mid SR(t) \ge 0.50 \}$$
   *Ý nghĩa:* Số lượng timesteps tương tác môi trường cần thiết để mô hình đạt mức thành công $50\%$.
5. **$M_5$ — Generalization Gap ($\Delta_{gen}$):**
   $$\Delta_{gen} = SR_{val} - SR_{test}$$
   *Ý nghĩa:* Độ sụt giảm hiệu năng giữa môi trường validation đã biết và test hoàn toàn mới (đánh giá overfitting hình học).
6. **$M_6$ — Alignment Correlation ($r_{sa}$):**
   $$r_{sa} = \text{Corr}(P(\text{success} \mid \tau_i), sa\_uniform_i)$$
   *Ý nghĩa:* Hệ số tương quan Pearson/Spearman giữa xác suất thành công thực tế của tác tử với nhãn `sa_uniform` của dataset. Đo lường mức độ đồng bộ giữa lý thuyết MDP và thực nghiệm DRL.

---

### 2.3. Các Điều Kiện Chuẩn Hóa & Kiểm Soát (Controlled Variables)

- **Thuật toán DRL:**
  - **PPO:** On-policy baseline (Schulman et al., 2017) với clipped surrogate objective.
  - **DQN:** Off-policy Q-learning baseline (Mnih et al., 2015) với replay buffer và target network.
  - **A2C:** Fast on-policy synchronous actor-critic baseline (Mnih et al., 2016).
- **Phần cứng thực thi:** Tăng tốc CUDA trên **NVIDIA GeForce RTX 3050 Laptop GPU** (đã xác thực hoạt động 100% trong `reproduction_report.md`).
- **Độ tin cậy thống kê:** Mỗi cấu hình thực nghiệm chạy trên tối thiểu **$N = 5$ random seeds** (`[10, 20, 30, 40, 50]`).
- **Ngân sách huấn luyện chuẩn:** $100,000$ timesteps cho PPO/DQN per run (đủ để hội tụ các bài toán định vị trên lưới).

---

## 3. CHI TIẾT MA TRẬN THỰC NGHIỆM CHÍNH (CORE EXPERIMENTAL SUITE)

Hệ thống thực nghiệm được chia thành 4 ma trận thành phần có mối quan hệ nhân quả chặt chẽ:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LỘ TRÌNH 4 MA TRẬN THỰC NGHIỆM                         │
│                                                                                 │
│   [EXP 1: V1 x V2]  ──► Xác định ngưỡng sụp đổ & Chọn hàm thưởng tối ưu         │
│          │                                                                      │
│          ▼                                                                      │
│   [EXP 2: V3 x V1]  ──► Đo độ nhạy Topology không gian trên từng cấp độ starved │
│          │                                                                      │
│          ▼                                                                      │
│   [EXP 3: V4 x V5]  ──► Khảo sát giới hạn chiều dài đường đi theo động học     │
│          │                                                                      │
│          ▼                                                                      │
│   [EXP 4: V6 (Curriculum)] ──► Can thiệp phục hồi cho vùng starved              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1. THỰC NGHIỆM 1 (CORE): $V_1 \times V_2$ — Sức Chống Chịu Khi Đói Tín Hiệu Dưới Các Dạng Hàm Thưởng
- **Mục tiêu:** Khảo sát mức độ suy giảm hiệu năng khi thiếu hụt tín hiệu thưởng, tìm kiếm **ngưỡng gãy đổ (Phase Transition Threshold)** giữa các cấp độ task khi có và không có kỹ thuật định hình phần thưởng (Reward Shaping).
- **Ý nghĩa đối với tầm nhìn dự án:** Đây là bài toán trung tâm của đề tài. Thí nghiệm này chứng minh bằng số liệu thực nghiệm rằng: Khi task rơi vào vùng `starved` ($sa\_uniform = 0$), thuật toán DRL bắt buộc phải có reward shaping (đặc biệt là Potential-based R6) mới có thể giải phóng gradient để học, đồng thời xác định ranh giới mà tại đó reward shaping không còn hiệu quả nếu không có curriculum.

#### Thiết kế Ma trận $V_1 \times V_2$:

| $V_1$ \ $V_2$ | **R1: Pure Sparse** | **R3: Distance Dense** | **R4: Progress Dense** | **R6: Potential Shaped** |
| :--- | :---: | :---: | :---: | :---: |
| **T1: Easy** ($sa \ge 0.15$) | Run 1.1 | Run 1.2 | Run 1.3 | Run 1.4 |
| **T2: Medium** ($0.05 \le sa < 0.15$) | Run 1.5 | Run 1.6 | Run 1.7 | Run 1.8 |
| **T3: Hard** ($0 < sa < 0.05$) | Run 1.9 | Run 1.10 | Run 1.11 | Run 1.12 |
| **T4: Starved** ($sa = 0$) | Run 1.13 | Run 1.14 | Run 1.15 | Run 1.16 |

- **Điều kiện cố định:** Robot = `turtlebot4`, Map split = `train`/`val`.
- **Số lượng runs:** $4\ (V_1) \times 4\ (V_2) \times 2\ (\text{Algos: PPO, DQN}) \times 5\ (\text{Seeds}) =$ **160 runs**.
- **Giả thuyết khoa học ($H_1$):**
  - Dưới R1 (Sparse), Success Rate sẽ giảm đột ngột từ $>80\%$ ở Easy xuống gần $0\%$ ở Hard và Starved.
  - Dưới R4 và R6, Success Rate tại Hard và Starved được duy trì $>50\%$.
  - R6 (Potential-based) mang lại SPL cao hơn R3 nhờ tính chất bảo toàn tối ưu chính sách.

---

### 3.2. THỰC NGHIỆM 2: $V_3 \times V_1$ — Tác Động Của Cấu Trúc Topology Bản Đồ
- **Mục tiêu:** Đo lường sự suy giảm hiệu năng khi không gian di chuyển thay đổi quy mô và độ phức tạp hình học ($C_{topo}$) ở từng mức độ đói tín hiệu.
- **Ý nghĩa đối với tầm nhìn dự án:** Làm rõ câu hỏi: *Liệu một bản đồ lớn hay một bản đồ nhiều vách ngăn gây ra sự đói tín hiệu nặng nề hơn?* Thí nghiệm phân tích sự tương tác giữa diện tích bản đồ (`main_m2`) và task difficulty để xác định xem sự thất bại của DRL là do "không gian tìm kiếm quá rộng" hay do "cấu trúc hình học ngăn cản".

#### Thiết kế Ma trận $V_3 \times V_1$:

| $V_3$ (Map Scale) \ $V_1$ (Task Level) | **Easy** | **Medium** | **Hard** | **Starved** |
| :--- | :---: | :---: | :---: | :---: |
| **Small** ($\text{main\_m2} < 30\ \text{m}^2$) | Run 2.1 | Run 2.2 | Run 2.3 | Run 2.4 |
| **Medium** ($30 \le \text{main\_m2} < 50\ \text{m}^2$) | Run 2.5 | Run 2.6 | Run 2.7 | Run 2.8 |
| **Large** ($\text{main\_m2} \ge 50\ \text{m}^2$) | Run 2.9 | Run 2.10 | Run 2.11 | Run 2.12 |

- **Điều kiện cố định:** Robot = `turtlebot4`, Reward = Best reward từ EXP 1 (R4 hoặc R6).
- **Số lượng runs:** $3\ (V_3) \times 4\ (V_1) \times 2\ (\text{Algos}) \times 5\ (\text{Seeds}) =$ **120 runs**.
- **Giả thuyết khoa học ($H_2$):**
  - Trên bản đồ Small, tác tử có thể giải quyết được cả một phần task Hard ngay cả khi tín hiệu thưa nhờ đường kính bản đồ nhỏ.
  - Trên bản đồ Large, hiệu năng trên nhóm Starved sụt giảm mạnh nhất do không gian trạng thái bùng nổ tổ hợp ($|S| \propto \text{main\_m2}$).

---

### 3.3. THỰC NGHIỆM 3: $V_4 \times V_5$ — Khả Năng Khái Quát Hóa Theo Chiều Dài Hành Trình và Động Học Robot
- **Mục tiêu:** Đánh giá tính tổng quát hóa theo chiều dài đường đi ($d_{bfs}$) trên các cấu trúc động học và kích thước thân xe khác nhau.
- **Ý nghĩa đối với tầm nhìn dự án:** Chứng minh tính ứng dụng thực tiễn của nghiên cứu trong Robotics. Các dòng robot có kích thước thân xe khác nhau (TurtleBot4 nhỏ tròn vs Jackal to dài) sẽ gặp trở ngại khác nhau khi đi qua cửa hẹp; động học toàn hướng (JetAuto omni) sẽ ít bị bế tắc hơn động học vi sai truyền thống.

#### Thiết kế Ma trận $V_4 \times V_5$:

| $V_4$ (Path Horizon) \ $V_5$ (Robot Model) | **TurtleBot 4** (Differential, Small) | **JetAuto Pro** (Omnidirectional) | **Clearpath Jackal** (Skid-steer, Large) |
| :--- | :---: | :---: | :---: |
| **Short** (`1-3m`) | Run 3.1 | Run 3.2 | Run 3.3 |
| **Medium** (`3-6m`) | Run 3.4 | Run 3.5 | Run 3.6 |
| **Long** (`6-10m`) | Run 3.7 | Run 3.8 | Run 3.9 |
| **Ultra-long** (`10m+`) | Run 3.10 | Run 3.11 | Run 3.12 |

- **Điều kiện cố định:** Reward = R6 (Potential-based), Dataset selection tương ứng cho mỗi robot.
- **Số lượng runs:** $4\ (V_4) \times 3\ (V_5) \times 2\ (\text{Algos}) \times 5\ (\text{Seeds}) =$ **120 runs**.
- **Giả thuyết khoa học ($H_3$):**
  - Clearpath Jackal có tỷ lệ va chạm cao hơn đáng kể và SPL suy giảm nhanh nhất khi hành trình dài ra do kích thước cồng kềnh.
  - JetAuto Pro duy trì sample efficiency cao nhất ở cự ly dài ($10\text{m}+$) nhờ khả năng di chuyển linh hoạt không cần xoay thân xe trước khi tịnh tiến.

---

### 3.4. THỰC NGHIỆM 4: $V_6$ — Đánh Giá Các Chiến Lược Curriculum Learning
- **Mục tiêu:** Khảo sát xem chiến lược học tăng dần độ khó (Curriculum Learning) có thể giúp tác tử vượt qua "cái chết ngạt" trong các task Starved mà không cần lạm dụng reward shaping hay không.
- **Ý nghĩa đối với tầm nhìn dự án:** Đưa ra giải pháp thuật toán hoàn chỉnh. Nếu Reward Shaping có rủi ro tạo ra các hành vi cục bộ (sub-optimal loops), thì Curriculum Learning trên thước đo `sa_uniform` cung cấp một phương pháp luận tự nhiên: bắt đầu từ các task có mật độ tín hiệu cao ($sa \ge 0.15$) và chuyển tiếp dần sang các task đói tín hiệu hoàn toàn ($sa = 0$).

#### Thiết kế Ma trận $V_6$:

| Chiến lược Curriculum | Mô tả cơ chế chuyển tầng | Sparse Reward (R1) | Shaped Reward (R6) |
| :--- | :--- | :---: | :---: |
| **C0: No Curriculum** | Lấy mẫu ngẫu nhiên đồng đều trên toàn bộ dataset | Run 4.1 | Run 4.2 |
| **C1: Level Curriculum** | Bắt đầu ở `easy`, tăng lên `medium` $\to$ `hard` $\to$ `starved` khi $SR \ge 70\%$ | Run 4.3 | Run 4.4 |
| **C2: Adaptive SA Curriculum**| Lọc ngưỡng $sa_{min}$ giảm liên tục từ $0.30 \to 0.0$ theo tiến độ training | Run 4.5 | Run 4.6 |

- **Điều kiện cố định:** Robot = `turtlebot4`, Đánh giá trên tập con gồm **100% Starved Tasks**.
- **Số lượng runs:** $3\ (V_6) \times 2\ (\text{Reward}) \times 2\ (\text{Algos}) \times 5\ (\text{Seeds}) =$ **60 runs**.
- **Giả thuyết khoa học ($H_4$):**
  - Tác tử không có curriculum (C0) dưới phần thưởng R1 hoàn toàn thất bại ($SR = 0\%$) trên tập Starved.
  - Chiến lược Adaptive SA Curriculum (C2) giúp tác tử đạt $SR > 40\%$ ngay cả khi chỉ huấn luyện với phần thưởng thuần R1 (Sparse).

---

## 4. BẢNG TỔNG HỢP TOÀN BỘ MA TRẬN VÀ KẾ HOẠCH TÀI NGUYÊN

| Nhóm Thực Nghiệm | Cặp Biến Khảo Sát | Số Cấu Hình | Thuật Toán | Số Seeds | Tổng Số Runs | Ước Tính Thời Gian (GPU RTX 3050) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP 1: Core Starvation** | $V_1 \times V_2$ (Task Level $\times$ Reward) | 16 | PPO, DQN | 5 | **160 runs** | ~2.5 giờ |
| **EXP 2: Topology Scale** | $V_3 \times V_1$ (Map Scale $\times$ Task Level) | 12 | PPO, DQN | 5 | **120 runs** | ~2.0 giờ |
| **EXP 3: Horizon & Robot**| $V_4 \times V_5$ (Path Length $\times$ Robot) | 12 | PPO, DQN | 5 | **120 runs** | ~2.0 giờ |
| **EXP 4: Curriculum** | $V_6 \times V_2$ (Curriculum $\times$ Reward) | 6 | PPO, DQN | 5 | **60 runs** | ~1.0 giờ |
| **TỔNG CỘNG TOÀN DỰ ÁN** | — | **46 configs** | **2 algos** | **5** | **460 runs** | **~7.5 giờ** |

> [!TIP]
> **Tối ưu hóa thời gian thực nghiệm:**
> Nhờ môi trường NavSim đạt **~36,000 FPS** và mô hình SB3 được tăng tốc bằng GPU NVIDIA RTX 3050 cùng cơ chế song song hóa đa tiến trình (5 parallel workers trên CPU 16 cores), toàn bộ ma trận 460 runs có thể hoàn thành trong vòng **chưa đến 1 ngày làm việc** (khoảng 7–8 giờ chạy liên tục).

---

## 5. PHƯƠNG PHÁP PHÂN TÍCH THỐNG KÊ (STATISTICAL PROTOCOL)

Để đảm bảo tính khoa học chặt chẽ và khả năng xuất bản bài báo khoa học chất lượng cao, các kết quả thu được sẽ được xử lý qua quy trình thống kê chuẩn:

1. **Ước lượng Khoảng Tin Cậy (Confidence Intervals):**
   - Với mọi độ đo ($M_1 \dots M_6$), báo cáo kết quả dưới dạng:
     $$\bar{X} \pm 1.96 \times \frac{s}{\sqrt{N}} \quad (95\%\ \text{Confidence Interval, Bootstrap } B=10,000)$$
2. **Kiểm Định Ý Nghĩa Thống Kê (Significance Testing):**
   - **Two-way ANOVA:** Đánh giá mức độ ảnh hưởng chính và tác động tương tác giữa các cặp biến ($V_1 \times V_2$, $V_3 \times V_1$, $V_4 \times V_5$).
   - **Tukey HSD Post-hoc Test:** So sánh bắt cặp giữa các dạng hàm thưởng (R1 vs R4 vs R6) để khẳng định sự vượt trội có ý nghĩa thống kê ($p < 0.01$).
   - **Mann-Whitney U Test:** Áp dụng cho các phân phối không chuẩn (SPL và Success Rate ở vùng starved).
3. **Phân Tích Tương Quan (Alignment Verification):**
   - Vẽ biểu đồ Scatter Plot và tính hệ số tương quan tuyến tính $R^2$ giữa `sa_uniform` và thực nghiệm empirical success để kiểm chứng giá trị đại diện của thước đo lý thuyết từ NavSim.

---

## 6. TIÊU CHÍ HOÀN THÀNH (DEFINITION OF DONE - DoD)

- [x] Thiết kế đầy đủ 6 biến độc lập ($V_1 \dots V_6$) và 6 biến phụ thuộc ($M_1 \dots M_6$).
- [x] Làm rõ ma trận 4 thực nghiệm tương tác ($V_1 \times V_2$, $V_3 \times V_1$, $V_4 \times V_5$, $V_6$).
- [x] Miêu tả sâu sắc ý nghĩa của từng thực nghiệm gắn liền với tầm nhìn nghiên cứu Signal Starvation.
- [x] Lập bảng hoạch định tài nguyên phần cứng và ước tính thời gian thực thi cụ thể trên GPU RTX 3050.
- [x] Đã khởi tạo tệp tài liệu chính thức tại [`Documentation/experiment_matrix.md`](file:///C:/SIGNAL-STARVATION/Documentation/experiment_matrix.md).
- [ ] **Người dùng review và chốt phê duyệt thiết kế ma trận.**
