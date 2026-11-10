# 📐 MA TRẬN THỰC NGHIỆM ĐỊNH LƯỢNG: ẢNH HƯỞNG CỦA CÁC ĐẶC TRƯNG MÔI TRƯỜNG ĐẾN HIỆU SUẤT CỐT LÕI CỦA DRL AGENTS TRONG PHYSICAL AI
## Quantitative Empirical Matrix: Impact of Environmental Properties on Core DRL Agent Performance in Physical AI Navigation

- **Tác giả:** Antigravity Research Team
- **Đề tài chính:** Khảo sát và định lượng ảnh hưởng của các đặc trưng môi trường (*environmental properties*) đến hiệu suất cốt lõi của các DRL agents (được sử dụng làm bộ não cho hệ thống Physical AI).
- **Cơ chế phân tích khoa học:** Tiếp cận qua lăng kính hiện tượng **Signal Starvation (Thiếu đói tín hiệu do cấu trúc môi trường)**.
- **Môi trường mô phỏng:** NavSim `GridNav-v0` (Gymnasium-native)
- **Tài liệu nền tảng:** [`ENV_AUDIT_REPORT.md`](file:///C:/SIGNAL-STARVATION/audit_reports/ENV_AUDIT_REPORT.md), [`reward_design_taxonomy_2d_navigation.md`](file:///C:/SIGNAL-STARVATION/reward_design_taxonomy_2d_navigation.md), [`reproduction_report.md`](file:///C:/SIGNAL-STARVATION/Documentation/reproduction_report.md)
- **Phiên bản:** v2.0 — Căn chỉnh chuẩn hóa theo mục tiêu cốt lõi của đề tài

---

## 1. MỤC TIÊU CỐT LÕI VÀ TẦM NHÌN NGHIÊN CỨU

### 1.1. Mục Tiêu Chính của Đề Tài
Mục tiêu cốt lõi của nghiên cứu là **khảo sát và định lượng một cách có hệ thống ảnh hưởng của các đặc trưng môi trường (*environmental properties*) đến hiệu suất của các mô hình Học tăng cường sâu (DRL agents)** đóng vai trò là cơ cấu ra quyết định và điều hướng (bộ não) cho các hệ thống Physical AI (Mobile Robots).

Trong các hệ thống Physical AI thực tế, môi trường vật lý không bao giờ đồng nhất mà luôn biến thiên đa dạng về:
1. **Quy mô và hình học không gian** (diện tích phòng, hành lang, ngõ cụt, vật cản phân mảnh).
2. **Quy mô khoảng cách nhiệm vụ** (đường đi ngắn vài mét đến dài xuyên phòng).
3. **Đặc tính phản hồi tín hiệu từ bài toán** (phần thưởng tự nhiên cực thưa hay có tín hiệu dẫn đường).
4. **Ràng buộc vật lý - động học của thân xe** (bán kính va chạm, cơ chế chuyển động vi sai hay toàn hướng).

### 1.2. Lăng Kính Khoa Học: Hiện Tượng Signal Starvation Do Cấu Trúc Môi Trường
Để giải thích về mặt cơ chế tại sao và khi nào các đặc trưng môi trường khiến DRL agents suy giảm hiệu năng hoặc tê liệt hoàn toàn, nghiên cứu sử dụng hiện tượng **Signal Starvation (Thiếu đói tín hiệu do cấu trúc môi trường)** làm luận điểm lý giải trung tâm:
- Khi một môi trường có cấu trúc topo phức tạp hoặc hành trình quá dài dưới hàm thưởng thưa, xác suất tác tử tình cờ chạm đích bằng chính sách khám phá ngẫu nhiên tiến dần về 0 ($\rho_{sig} \to 1.0$).
- Sự thiếu đói tín hiệu này là **hệ quả trực tiếp sinh ra từ sự tương tác giữa cấu trúc hình học của môi trường với bài toán**, làm triệt tiêu gradient cập nhật của mạng nơ-ron và gây sụp đổ học tập (*Learning Collapse*).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MÔ HÌNH QUAN HỆ NHÂN QUẢ                                 │
│                                                                                        │
│   ĐẶC TRƯNG MÔI TRƯỜNG (Environmental Properties)                                      │
│   ┌────────────────────────────────────────────────────────────────────────────┐       │
│   │ • Topology không gian & Diện tích di chuyển (C_topo, main_m2)              │       │
│   │ • Chiều dài hành trình Start-Goal (d_bfs)                                  │       │
│   │ • Ràng buộc hình học thân xe & Động học Robot (Kinematics & Footprint)     │       │
│   │ • Mật độ tín hiệu thưởng tự nhiên của bài toán (Reward Sparsity)           │       │
│   └─────────────────────────────────────┬──────────────────────────────────────┘       │
│                                         │ (Tác động sinh ra cơ chế)                    │
│                                         ▼                                              │
│               [CƠ CHẾ LÝ GIẢI KHOA HỌC: SIGNAL STARVATION (sa_uniform)]                │
│                                         │                                              │
│                                         ▼ (Quyết định)                                 │
│   3 TRỤ CỘT HIỆU SUẤT CỦA DRL AGENTS (Core Performance Metrics)                        │
│   ┌────────────────────────────────────────────────────────────────────────────┐       │
│   │ 1. Độ chính xác & Tỷ lệ thành công (Success Rate & Cumulative Reward)      │       │
│   │ 2. Khả năng Tổng quát hóa (Generalization Gap trên môi trường chưa thấy)   │       │
│   │ 3. Hiệu suất mẫu & Tốc độ huấn luyện (Sample Efficiency & Training Time)   │       │
│   └────────────────────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3. Tiêu Chí Hoàn Thành Mục Tiêu (Definition of Done)
1. **Báo cáo Thực nghiệm:** Hoàn thiện bản báo cáo khoa học chứng minh tính hợp lý của phương pháp luận, với dữ liệu từ số lượng bài testing đủ lớn (tối thiểu **$N = 5$ random seeds độc lập** trên mỗi biến cấu hình).
2. **Kết quả Định lượng Chi tiết:** Phân tích rõ ràng mối quan hệ nhân quả và mức độ ảnh hưởng của từng yếu tố môi trường đối với **3 nhóm metrics chuẩn**:
   - *Độ chính xác / Tỷ lệ thành công (Success Rate & Cumulative Reward).*
   - *Khả năng tổng quát hóa ($Generalization Gap$ trên môi trường chưa thấy).*
   - *Thời gian huấn luyện / Hiệu suất mẫu ($Sample Efficiency / Training Time$).*
3. **Lý giải Khoa học & Giả thuyết:** Cung cấp phân tích sâu sắc gắn liền với hiện tượng *Signal Starvation* (Thiếu đói tín hiệu do cấu trúc môi trường), tổng kết các giả thuyết đã được kiểm chứng và đề xuất các giả thuyết mới cần mở rộng kèm điều kiện thực nghiệm chi tiết.

---

## 2. HỆ THỐNG BIẾN THỰC NGHIỆM ĐỊNH LƯỢNG

### 2.1. Nhóm Biến Độc Lập: Các Đặc Trưng Môi Trường ($V_1 \dots V_5$) & Can Thiệp Thuật Toán ($V_6$)

Các biến độc lập được ánh xạ trực tiếp từ cấu trúc dataset thực tế `grid-*-v2` và hệ thống wrappers môi trường của dự án:

| Ký hiệu | Đặc Trưng Môi Trường / Yếu Tố | Mức Khảo Sát (Levels) | Hiện Thực Hóa Kỹ Thuật | Cơ Sở Dữ Liệu Thực Tế |
| :---: | :--- | :--- | :--- | :--- |
| **$V_1$** | **Mức độ Đói tín hiệu của Tác vụ** (*Task Starvation Characteristic*) | - `easy` ($sa \ge 0.15$)<br>- `medium` ($0.05 \le sa < 0.15$)<br>- `hard` ($0 < sa < 0.05$)<br>- `starved` ($sa = 0$) | `set_task_filter(level=...)`<br>hoặc `SAFilter(sa_min, sa_max)` | Chỉ số định lượng `sa_uniform` pre-computed từ MDP trong dataset NavSim (34.5% easy, 34.0% medium, 9.6% hard, 21.9% starved). |
| **$V_2$** | **Mật độ Phản hồi Tín hiệu Thưởng** (*Environmental Reward Density*) | - **R1:** Pure Sparse (Mặc định)<br>- **R3:** Distance Dense<br>- **R4:** Progress Dense<br>- **R6:** Potential-based Shaped | `RewardFactory.make(env, "r1"..."r6")`<br>từ [`src/rewards/`](file:///C:/SIGNAL-STARVATION/src/rewards/) | Biến thiên từ môi trường thưởng cực thưa tự nhiên (R1) đến dày đặc (R3, R4) và bảo toàn chính sách tối ưu (R6 - Ng et al., 1999). |
| **$V_3$** | **Quy mô & Độ phức tạp Topology Bản đồ** (*Topological Complexity & Area*) | - **Small:** $\text{main\_m2} < 30\ \text{m}^2$<br>- **Medium:** $30 \le \text{main\_m2} < 50\ \text{m}^2$<br>- **Large:** $\text{main\_m2} \ge 50\ \text{m}^2$ | Bộ lọc danh sách map theo diện tích di chuyển `main_m2` | Kết quả profiler trên 138 maps thực tế: Diện tích dao động $11.02\ \text{m}^2 \to 145.41\ \text{m}^2$, mật độ chướng ngại vật $1 - \text{main\_frac} \approx 9.1\%$. |
| **$V_4$** | **Chiều dài Hành trình Start-Goal** (*Spatial Path Length Horizon*) | - `1-3m` (ngắn)<br>- `3-6m` (vừa)<br>- `6-10m` (dài)<br>- `10m+` (rất dài) | `set_task_filter(dist_bin=...)` | Thuộc tính phân nhóm cự ly BFS `dist_bin` pre-labeled trong metadata tasks ($d_{bfs} \in [8, 67]$ cells). |
| **$V_5$** | **Ràng buộc Hình học Thân xe & Động học** (*Kinematic Footprint Constraint*) | - **TurtleBot 4:** Differential, hình tròn nhỏ ($r=0.171\text{m}$)<br>- **JetAuto Pro:** Holonomic (omni-directional), cơ động cao<br>- **Clearpath Jackal:** Skid-steer, hình chữ nhật lớn ($0.51 \times 0.43\text{m}$) | Chọn dataset môi trường:<br>`grid-turtlebot4-v2`<br>`grid-jetauto_pro-v2`<br>`grid-clearpath_jackal-v2` | Khảo sát ảnh hưởng của va chạm cạnh hẹp và ràng buộc quay thân xe đối với khả năng di chuyển trong không gian vật lý. |
| **$V_6$** | **Chiến lược Can thiệp Tiến trình Môi trường** (*Environmental Curriculum Strategy*) | - **C0:** Không có curriculum (Uniform random)<br>- **C1:** Level Curriculum (`easy` $\to$ `medium` $\to$ `hard` $\to$ `starved`)<br>- **C2:** Adaptive SA Curriculum ($sa_{min}$ trượt giảm dần) | Custom Training Callback tích hợp `TaskFilterWrapper` | Đánh giá năng lực của chiến lược thay đổi phân phối môi trường huấn luyện theo độ khó tăng dần để vượt qua điểm nghẽn học tập. |

---

### 2.2. Nhóm Biến Phụ Thuộc: 3 Trụ Cột Hiệu Suất Cốt Lõi của DRL Agents

Mọi đánh giá định lượng được thực hiện trên tập kiểm thử độc lập (**Validation Split: 20 maps, 400 tasks** và **Test Split: 20 maps, 400 tasks**) thông qua giao thức chuẩn `navsim.evaluate()`:

#### Trụ Cột 1: Độ Chính Xác & Tỷ Lệ Hoàn Thành Nhiệm Vụ (Accuracy & Task Completion)
- **$M_1$ — Success Rate ($SR$):**
  $$SR = \frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} \mathbb{I}(\text{reached\_goal}_i) \in [0.0, 1.0]$$
  *Ý nghĩa:* Tỷ lệ nhiệm vụ robot điều hướng thành công đến mục tiêu trong ngân sách bước tối đa cho phép.
- **$M_2$ — Cumulative Reward ($G$):**
  $$G = \frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} \sum_{t=0}^T \gamma^t r_{i, t}$$
  *Ý nghĩa:* Tổng điểm thưởng trung bình tích lũy được trong mỗi episode, phản ánh chất lượng tối ưu hóa mục tiêu.
- **$M_{1b}$ — Success weighted by Path Length ($SPL$):**
  $$SPL = \frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} S_i \frac{l_i}{\max(p_i, l_i)} \in [0.0, 1.0]$$
  *Ý nghĩa:* Thước đo chuẩn hóa quốc tế (Anderson et al., 2018) kết hợp giữa việc đến đích ($S_i=1$) và độ dài đường đi ngắn nhất ($l_i / p_i$).

#### Trụ Cột 2: Khả Năng Tổng Quát Hóa (Generalization Capability)
- **$M_3$ — Generalization Gap ($\Delta_{gen}$):**
  $$\Delta_{gen} = SR_{val} - SR_{test}$$
  *Ý nghĩa:* Độ sụt giảm hiệu năng khi agent đối mặt với 20 bản đồ kiểm thử hoàn toàn mới (Test split) so với 20 bản đồ đã gặp trong quá trình validation. Đại diện cho mức độ nhạy cảm của agent với sự thay đổi bố cục môi trường (tránh hiện tượng overfitting cấu trúc hình học quen thuộc).
- **$M_{3b}$ — Generalization SPL Gap ($\Delta_{SPL}$):**
  $$\Delta_{SPL} = SPL_{val} - SPL_{test}$$
  *Ý nghĩa:* Đánh giá sự suy giảm về độ mượt mà và tối ưu của đường đi trên địa hình chưa từng biết.

#### Trụ Cột 3: Hiệu Suất Mẫu & Tốc Độ Huấn Luyện (Sample Efficiency & Training Time)
- **$M_4$ — Sample Efficiency ($SE_{50}$):**
  $$SE_{50} = \min \{ t \mid SR(t) \ge 0.50 \}$$
  *Ý nghĩa:* Số lượng bước tương tác môi trường (timesteps) tối thiểu để mô hình đạt mức thành công $50\%$. Nếu sau toàn bộ ngân sách vẫn không đạt $50\%$, $SE_{50} = \infty$ (đại diện cho trạng thái học thất bại).
- **$M_5$ — Training Throughput & Wall Time ($FPS$ & $T_{wall}$):**
  $$FPS = \frac{\text{Total Timesteps}}{T_{wall}}\ (\text{steps/second})$$
  *Ý nghĩa:* Tốc độ thực thi và thời gian tính toán thực tế của pipeline trên phần cứng GPU NVIDIA GeForce RTX 3050 Laptop GPU.

#### Metric Bổ Trợ: Hệ Số Đồng Bộ Lý Thuyết - Thực Nghiệm (Mechanistic Alignment)
- **$M_6$ — Correlation: Empirical Success vs. $sa\_uniform$ ($r_{sa}$):**
  $$r_{sa} = \text{Corr}(P(\text{success} \mid \tau_i), sa\_uniform_i)$$
  *Ý nghĩa:* Đo lường mức độ tương quan (Pearson & Spearman) giữa xác suất thành công thực tế của mô hình với nhãn độ khó lý thuyết MDP từ dataset NavSim, chứng minh tính hợp lý khoa học của proxy môi trường.

---

### 2.3. Điều Kiện Kiểm Soát & Chuẩn Hóa (Controlled Parameters)

- **Thuật toán DRL đại diện:**
  - **PPO (On-policy):** Đại diện cho họ Policy Gradient ổn định cao, tối ưu bằng clipped objective.
  - **DQN (Off-policy):** Đại diện cho họ Value-based với Experience Replay và Target Network.
  - **A2C (Synchronous On-policy):** Dùng đối chứng tốc độ và cập nhật đồng bộ.
- **Phần cứng thực nghiệm:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1, PyTorch 2.5.1+cu121) + 16-core CPU.
- **Độ tin cậy thống kê:** Tối thiểu **$N = 5$ random seeds độc lập** (`[10, 20, 30, 40, 50]`) cho mỗi cấu hình thực nghiệm.
- **Ngân sách huấn luyện chuẩn:** $100,000$ steps (PPO, DQN) per run, đảm bảo đủ thời gian quan sát hiện tượng hội tụ hoặc sụp đổ.

---

## 3. THIẾT KẾ CHI TIẾT 4 MA TRẬN THỰC NGHIỆM

Hệ thống thực nghiệm được thiết kế thành 4 ma trận tương tác nhằm bóc tách quan hệ nhân quả giữa từng nhóm đặc trưng môi trường với 3 trụ cột hiệu suất của DRL agents:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              CẤU TRÚC 4 MA TRẬN THỰC NGHIỆM                            │
│                                                                                        │
│   [MA TRẬN 1: V1 x V2] ──► Mức độ đói tín hiệu x Mật độ thưởng môi trường              │
│            │               (Khảo sát Tỷ lệ thành công & Hiệu suất mẫu - Ngưỡng gãy đổ) │
│            ▼                                                                           │
│   [MA TRẬN 2: V3 x V1] ──► Quy mô & Topology bản đồ x Mức độ đói tín hiệu              │
│            │               (Khảo sát Khả năng tổng quát hóa & Độ nhạy diện tích)       │
│            ▼                                                                           │
│   [MA TRẬN 3: V4 x V5] ──► Chiều dài hành trình x Ràng buộc động học Robot             │
│            │               (Khảo sát Suy giảm hiệu năng theo cự ly & Kích thước xe)    │
│            ▼                                                                           │
│   [MA TRẬN 4: V6]      ──► Chiến lược Curriculum thích nghi môi trường                 │
│                            (Cơ chế phục hồi hiệu suất trong môi trường cực đoan)       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1. MA TRẬN 1: $V_1 \times V_2$ — Tác Động Phối Hợp Giữa Mức Độ Đói Tín Hiệu Tác Vụ và Mật Độ Thưởng Môi Trường
- **Câu hỏi nghiên cứu:** *Đặc trưng đói tín hiệu của tác vụ ($V_1$) ảnh hưởng như thế nào đến Tỷ lệ thành công ($M_1$) và Hiệu suất mẫu ($M_4$) của DRL agents dưới các cấu hình mật độ thưởng môi trường ($V_2$) khác nhau? Đâu là ngưỡng gãy đổ (Phase Transition Threshold)?*
- **Ý nghĩa đối với đề tài:** Xác định giới hạn chịu đựng của DRL agents trước tính chất thưởng thưa tự nhiên của môi trường vật lý. Kiểm chứng xem kỹ thuật định hình phần thưởng (Reward Shaping) có thực sự giải quyết được bài toán thiếu đói tín hiệu mà không làm lệch chính sách tối ưu hay không.

#### Bảng Cấu Hình Ma Trận 1 ($V_1 \times V_2$):

| $V_1$ (Task Starvation) \ $V_2$ (Reward Density) | **R1: Pure Sparse** (Tự nhiên) | **R3: Distance Dense** (Dày) | **R4: Progress Dense** (Tiến độ) | **R6: Potential Shaped** (Bảo toàn) |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** ($sa \ge 0.15$) | Config 1.1 | Config 1.2 | Config 1.3 | Config 1.4 |
| **Medium** ($0.05 \le sa < 0.15$) | Config 1.5 | Config 1.6 | Config 1.7 | Config 1.8 |
| **Hard** ($0 < sa < 0.05$) | Config 1.9 | Config 1.10 | Config 1.11 | Config 1.12 |
| **Starved** ($sa = 0$) | Config 1.13 | Config 1.14 | Config 1.15 | Config 1.16 |

- **Thông số cố định:** Robot = `turtlebot4`, Map pool = `grid-turtlebot4-v2` train/val split.
- **Quy mô thực nghiệm:** 16 ô cấu hình $\times$ 2 thuật toán (PPO, DQN) $\times$ 5 seeds = **160 runs**.
- **Giả thuyết khoa học cần kiểm chứng ($H_1$):**
  1. Dưới môi trường thưởng thưa tự nhiên (R1), tồn tại một điểm gãy đổ sắc nét (*sharp phase transition*): Hiệu suất sụp đổ từ $SR > 80\%$ ở Easy xuống $SR \approx 0\%$ ở Hard và Starved do hiện tượng Signal Starvation triệt tiêu tín hiệu học.
  2. Dưới hàm thưởng Potential-based (R6), agent duy trì $SR > 60\%$ trên cả tập Starved mà vẫn đạt $SPL$ tiệm cận tối ưu nhờ bảo toàn thứ tự ưu tiên chính sách của Ng et al. (1999).
  3. DQN có hiệu suất mẫu ($SE_{50}$) tốt hơn PPO trong các bài toán Medium nhờ cơ chế tái sử dụng mẫu (Experience Replay), nhưng dễ phân kỳ hơn PPO khi rơi vào môi trường Starved kết hợp hàm thưởng R3 do lỗi ước lượng Q-value tích lũy.

---

### 3.2. MA TRẬN 2: $V_3 \times V_1$ — Tác Động Của Quy Mô & Cấu Trúc Topology Bản Đồ Đến Khả Năng Tổng Quát Hóa
- **Câu hỏi nghiên cứu:** *Quy mô diện tích di chuyển (`main_m2`) và độ phức tạp topology bản đồ ($V_3$) làm suy giảm Tỷ lệ thành công ($M_1$) và mở rộng Generalization Gap ($M_3$) như thế nào ở từng cấp độ đói tín hiệu ($V_1$)?*
- **Ý nghĩa đối với đề tài:** Phân định rõ ràng ảnh hưởng giữa hai yếu tố môi trường: Sự thất bại của agent là do **không gian tìm kiếm trạng thái bùng nổ** ($|S| \propto \text{main\_m2}$) hay do **cấu trúc hình học vật cản ngăn cách** tạo ra bẫy tín hiệu (ngõ cụt, phòng cô lập).

#### Bảng Cấu Hình Ma Trận 2 ($V_3 \times V_1$):

| $V_3$ (Map Topology Scale) \ $V_1$ (Task Level) | **Easy Tasks** | **Medium Tasks** | **Hard Tasks** | **Starved Tasks** |
| :--- | :---: | :---: | :---: | :---: |
| **Small Maps** ($\text{main\_m2} < 30\ \text{m}^2$) | Config 2.1 | Config 2.2 | Config 2.3 | Config 2.4 |
| **Medium Maps** ($30 \le \text{main\_m2} < 50\ \text{m}^2$) | Config 2.5 | Config 2.6 | Config 2.7 | Config 2.8 |
| **Large Maps** ($\text{main\_m2} \ge 50\ \text{m}^2$) | Config 2.9 | Config 2.10 | Config 2.11 | Config 2.12 |

- **Thông số cố định:** Robot = `turtlebot4`, Reward = R6 (Potential-based để đảm bảo agent nhận được tín hiệu học nền).
- **Quy mô thực nghiệm:** 12 ô cấu hình $\times$ 2 thuật toán $\times$ 5 seeds = **120 runs**.
- **Giả thuyết khoa học cần kiểm chứng ($H_2$):**
  1. Trên bản đồ Small, tác tử đạt Generalization Gap thấp ($\Delta_{gen} < 5\%$) ngay cả trên task Hard vì đường kính đồ thị nhỏ hạn chế không gian lạc lối của agent.
  2. Trên bản đồ Large, Generalization Gap nới rộng mạnh mẽ ($\Delta_{gen} > 25\%$) đối với task Hard và Starved, phản ánh hiện tượng agent bị "học vẹt" (overfitting) các lộ trình cụ thể trong tập train và thất bại khi cấu trúc phòng ốc ở tập test thay đổi.

---

### 3.3. MA TRẬN 3: $V_4 \times V_5$ — Ảnh Hưởng Của Chiều Dài Hành Trình và Ràng Buộc Động Học Robot
- **Câu hỏi nghiên cứu:** *Chiều dài hành trình Start-Goal ($V_4$) làm suy giảm độ chính xác và tốc độ hội tụ ra sao khi tương tác với các cấu trúc động học và kích thước thân xe khác nhau ($V_5$)?*
- **Ý nghĩa đối với đề tài:** Gắn kết lý thuyết DRL với bài toán Physical AI thực tế. Trong môi trường thực, robot không phải là chất điểm vô hình mà có kích thước va chạm (footprint) và cơ chế di chuyển cụ thể (vi sai, toàn hướng, skid-steer). Thí nghiệm này định lượng mức độ suy thoái hiệu năng khi chiều dài nhiệm vụ tăng dần đối với từng kiểu thân xe.

#### Bảng Cấu Hình Ma Trận 3 ($V_4 \times V_5$):

| $V_4$ (Path Horizon) \ $V_5$ (Robot Model) | **TurtleBot 4** (Nhỏ tròn, vi sai) | **JetAuto Pro** (Vừa, toàn hướng omni) | **Clearpath Jackal** (Lớn chữ nhật, skid-steer) |
| :--- | :---: | :---: | :---: |
| **Short** (`1-3m`) | Config 3.1 | Config 3.2 | Config 3.3 |
| **Medium** (`3-6m`) | Config 3.4 | Config 3.5 | Config 3.6 |
| **Long** (`6-10m`) | Config 3.7 | Config 3.8 | Config 3.9 |
| **Ultra-long** (`10m+`) | Config 3.10 | Config 3.11 | Config 3.12 |

- **Thông số cố định:** Reward = R6, Dataset tương ứng cho mỗi robot (`grid-turtlebot4-v2`, `grid-jetauto_pro-v2`, `grid-clearpath_jackal-v2`).
- **Quy mô thực nghiệm:** 12 ô cấu hình $\times$ 2 thuật toán $\times$ 5 seeds = **120 runs**.
- **Giả thuyết khoa học cần kiểm chứng ($H_3$):**
  1. Với hành trình ngắn (`1-3m`), cả 3 robot đều đạt $SR > 90\%$, sự khác biệt về kích thước thân xe chưa bộc lộ rõ rệt.
  2. Khi hành trình kéo dài sang `10m+`, Clearpath Jackal suy giảm $SR$ và $SPL$ nhanh nhất do kích thước thân xe lớn dễ va chạm khi đi qua các cửa hẹp, làm gia tăng hiện tượng đói tín hiệu do va chạm kết thúc sớm episode.
  3. JetAuto Pro đạt Sample Efficiency ($SE_{50}$) vượt trội trên các hành trình dài nhờ động học toàn hướng (omni) cho phép tịnh tiến và né vật cản mà không cần thực hiện chuỗi xoay thân xe phức tạp.

---

### 3.4. MA TRẬN 4: $V_6$ — Đánh Giá Năng Lực Can Thiệp Của Chiến Lược Environmental Curriculum
- **Câu hỏi nghiên cứu:** *Liệu việc thay đổi phân phối môi trường theo chiến lược học tăng dần (Curriculum Learning — $V_6$) có thể giúp agent vượt qua hiện tượng Signal Starvation trong các môi trường cực đoan (Starved Tasks) mà không cần can thiệp méo mó hàm thưởng hay không?*
- **Ý nghĩa đối với đề tài:** Đóng vai trò là giải pháp phương pháp luận hoàn thiện đề tài. Chứng minh rằng thay vì cố gắng sửa hàm thưởng (vốn có thể tạo ra các hành vi cục bộ sai lệch), việc tổ chức lại lộ trình tiếp xúc với các đặc trưng môi trường từ dễ đến khó là giải pháp tự nhiên và bền vững nhất cho Physical AI.

#### Bảng Cấu Hình Ma Trận 4 ($V_6$):

| Chiến Lược Curriculum ($V_6$) | Cơ Chế Điều Tiết Môi Trường | Môi Trường Thưởng Thưa (R1) | Môi Trường Thưởng Shaped (R6) |
| :--- | :--- | :---: | :---: |
| **C0: Không có Curriculum** | Lấy mẫu ngẫu nhiên đồng đều trên toàn bộ không gian task | Config 4.1 | Config 4.2 |
| **C1: Level Curriculum** | Chuyển tầng (`easy` $\to$ `medium` $\to$ `hard` $\to$ `starved`) khi $SR \ge 70\%$ | Config 4.3 | Config 4.4 |
| **C2: Adaptive SA Curriculum**| Hạ ngưỡng $sa_{min}$ liên tục từ $0.30 \to 0.0$ theo tiến độ training | Config 4.5 | Config 4.6 |

- **Thông số cố định:** Robot = `turtlebot4`. Đánh giá hiệu năng trực tiếp trên tập kiểm thử gồm **100% Starved Tasks**.
- **Quy mô thực nghiệm:** 6 ô cấu hình $\times$ 2 thuật toán $\times$ 5 seeds = **60 runs**.
- **Giả thuyết khoa học cần kiểm chứng ($H_4$):**
  1. Dưới môi trường thưởng thưa tự nhiên (R1), tác tử không có curriculum (C0) hoàn toàn thất bại ($SR = 0\%$) trên tập Starved Tasks do không bao giờ nhận được tín hiệu thưởng ban đầu.
  2. Chiến lược Adaptive SA Curriculum (C2) giúp tác tử tích lũy được biểu diễn không gian từ các vùng giàu tín hiệu, qua đó đạt $SR > 45\%$ trên tập Starved ngay cả khi chỉ sử dụng phần thưởng thuần R1 (Sparse), thu hẹp khoảng cách hiệu năng mà không cần dùng đến Dense Reward.

---

## 4. TỔNG HỢP KẾ HOẠCH PHÂN BỔ TÀI NGUYÊN & THỜI GIAN CHẠY

Toàn bộ ma trận được thiết kế tối ưu để có thể chạy tự động, lưu kết quả liên tục và tận dụng tối đa năng lực phần cứng hiện có của máy trạm:

| Nhóm Thực Nghiệm | Ma Trận Khảo Sát | Cấu Hình | Thuật Toán | Số Seeds | Tổng Số Runs | Thời Gian Ước Tính (GPU RTX 3050) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP 1: Task $\times$ Reward** | $V_1 \times V_2$ | 16 | PPO, DQN | 5 | **160 runs** | ~2.5 giờ |
| **EXP 2: Topology $\times$ Task**| $V_3 \times V_1$ | 12 | PPO, DQN | 5 | **120 runs** | ~2.0 giờ |
| **EXP 3: Horizon $\times$ Robot** | $V_4 \times V_5$ | 12 | PPO, DQN | 5 | **120 runs** | ~2.0 giờ |
| **EXP 4: Curriculum** | $V_6 \times V_2$ | 6 | PPO, DQN | 5 | **60 runs** | ~1.0 giờ |
| **TỔNG CỘNG TOÀN DỰ ÁN** | — | **46 configs** | **2 algos** | **5** | **460 runs** | **~7.5 giờ** |

> [!TIP]
> **Khả năng hoàn thành:** Với tốc độ thực thi ~36,000 FPS của NavSim và khả năng tính toán song song 5 tiến trình worker trên GPU RTX 3050 (PyTorch 2.5.1 CUDA), toàn bộ 460 runs có thể hoàn tất trong **~7.5 giờ chạy liên tục**. Hệ thống runner sẽ ghi trực tiếp kết quả vào file CSV sau mỗi run để đảm bảo tính an toàn (crash-safe) và hỗ trợ tiếp tục (resume) nếu bị gián đoạn.

---

## 5. QUY TRÌNH PHÂN TÍCH THỐNG KÊ & CHỨNG MINH GIẢ THUYẾT

Nhằm đáp ứng yêu cầu của một báo cáo khoa học chuẩn mực, dữ liệu thu thập từ 460 runs sẽ được phân tích qua quy trình thống kê 3 bước:

1. **Ước Lượng Khoảng Tin Cậy (Confidence Interval Estimation):**
   - Với mọi độ đo thuộc 3 trụ cột hiệu suất ($M_1 \dots M_5$), công bố giá trị trung bình kèm khoảng tin cậy 95%:
     $$\text{CI}_{95\%} = \bar{X} \pm 1.96 \cdot \frac{s}{\sqrt{N}} \quad (\text{với } N=5\text{ random seeds})$$
   - Áp dụng kỹ thuật Bootstrap phi tham số ($B = 10,000$ resamples) cho các chỉ số có phân phối lệch như $SPL$ và Generalization Gap.
2. **Kiểm Định Ý Nghĩa Thống Kê & Phân Tích Tương Tác (Hypothesis Testing):**
   - **Two-way ANOVA:** Phân tích ảnh hưởng chính (*main effects*) và tương tác chéo (*interaction effects*) của các cặp đặc trưng môi trường ($V_1 \times V_2$, $V_3 \times V_1$, $V_4 \times V_5$).
   - **Tukey’s Honestly Significant Difference (HSD):** Kiểm định hậu nghiệm (*post-hoc*) để so sánh định lượng sự khác biệt giữa các mức phần thưởng (R1 vs R4 vs R6) và các kiểu robot với mức ý nghĩa $p < 0.01$.
   - **Mann-Whitney U Test:** Kiểm định phi tham số khẳng định sự vượt trội có ý nghĩa thống kê của phương pháp Curriculum Learning (C2) so với baseline (C0) trên các task Starved.
3. **Phân Tích Cơ Chế Định Lượng (Mechanistic Validation):**
   - Lập đồ thị hồi quy và tính hệ số tương quan tuyến tính $R^2$ giữa nhãn `sa_uniform` và tỷ lệ thành công thực nghiệm $SR$ nhằm khẳng định tính chuẩn xác của proxy đo lường Signal Starvation sinh ra từ cấu trúc môi trường.

---

## 6. TIÊU CHÍ HOÀN THÀNH (DEFINITION OF DONE - DoD CHECKLIST)

- [x] **Tập trung vào mục tiêu chính của đề tài:** Khảo sát và định lượng ảnh hưởng của các đặc trưng môi trường (*environmental properties*) đến hiệu suất cốt lõi của DRL agents trong Physical AI.
- [x] **Định vị đúng vai trò của Signal Starvation:** Đóng vai trò là cơ chế lý giải khoa học trung tâm để giải thích nguyên nhân gây suy giảm và sụp đổ hiệu suất khi cấu trúc môi trường thay đổi.
- [x] **Cấu trúc rõ ràng 3 trụ cột hiệu suất chuẩn:**
  1. *Độ chính xác / Tỷ lệ thành công (Success Rate & Cumulative Reward, kèm SPL).*
  2. *Khả năng tổng quát hóa (Generalization Gap trên môi trường Test chưa từng biết).*
  3. *Hiệu suất mẫu & Tốc độ huấn luyện (Sample Efficiency $SE_{50}$ & FPS).*
- [x] **Thiết kế ma trận chi tiết 4 nhóm thực nghiệm ($V_1 \times V_2$, $V_3 \times V_1$, $V_4 \times V_5$, $V_6$)** với đầy đủ các giả thuyết khoa học ($H_1 \dots H_4$).
- [x] **Kế hoạch kiểm thử chặt chẽ với $N = 5$ random seeds độc lập**, đảm bảo độ tin cậy thống kê và khả năng thực thi khả thi trên GPU RTX 3050 (~7.5 giờ).
- [ ] **Người dùng review và chốt phê duyệt thiết kế ma trận.**
