# 📐 MA TRẬN THỰC NGHIỆM ĐỊNH LƯỢNG: ẢNH HƯỞNG CỦA CÁC ĐẶC TRƯNG MÔI TRƯỜNG ĐẾN HÀNH VI, ĐỘNG HỌC HỌC TẬP VÀ GIỚI HẠN CAN THIỆP CỦA DRL TRONG PHYSICAL AI
## Quantitative Empirical Matrix: Impact of Environmental Properties on Learning Dynamics, Policy Behaviors, and Mitigation Limits of DRL in Physical AI

- **Tác giả:** Antigravity Research Team
- **Đề tài chính:** Khảo sát và định lượng ảnh hưởng của các đặc trưng môi trường (*environmental properties*) đến hành vi, động học học tập và hiệu suất cốt lõi của các DRL agents (được sử dụng làm bộ não cho hệ thống Physical AI).
- **Lăng kính cơ chế:** Hiện tượng **Signal Starvation (Thiếu đói tín hiệu do cấu trúc môi trường)** và sự bóp nghẹt không gian trạng thái khả dụng.
- **Môi trường thực nghiệm:** NavSim `GridNav-v0` (Gymnasium-native)
- **Tài liệu nền tảng:** [`ENV_AUDIT_REPORT.md`](file:///C:/SIGNAL-STARVATION/audit_reports/ENV_AUDIT_REPORT.md), [`reward_design_taxonomy_2d_navigation.md`](file:///C:/SIGNAL-STARVATION/reward_design_taxonomy_2d_navigation.md), [`reproduction_report.md`](file:///C:/SIGNAL-STARVATION/Documentation/reproduction_report.md)
- **Phiên bản:** v3.0 — Định hình chuẩn mực học thuật, giải phẫu động học DRL và giới hạn các giải pháp can thiệp

---

## 1. TỔNG QUAN VÀ QUAN ĐIỂM CỐT LÕI CỦA NGHIÊN CỨU (CORE ACADEMIC THESIS)

### 1.1. Vượt Qua "Bẫy Hiện Tượng Hiển Nhiên" (Transcending the Trivial Geometric Trap)
Trong bài toán điều hướng tự chủ bằng Học tăng cường sâu (DRL Navigation for Physical AI), nếu chỉ quan sát:
> *"Bản đồ rộng hơn thì robot khó tìm đường hơn, robot thân xe to hơn thì khó lọt qua khe hẹp hơn, hoặc đường dài hơn thì xác suất va chạm tăng lên."*

$\to$ Đây thuần túy là những **sự thật hình học hiển nhiên (Trivial geometric facts)**, ai cũng có thể suy luận trực giác và **hoàn toàn không mang giá trị nghiên cứu khoa học**.

**Giá trị học thuật thực thụ của nghiên cứu này nằm ở việc giải quyết 3 câu hỏi bản chất:**
1. **Định Lượng Điểm Nghẽn Chi Phối (Bottleneck Hierarchy & Attribution):**
   Trong một hệ thống Physical AI chịu tác động đồng thời của nhiều yếu tố môi trường (Quy mô bản đồ, Chiều dài chuỗi quyết định, Kích thước hình học thân xe đối với cửa hẹp, và Mật độ tín hiệu thưởng tự nhiên): **Yếu tố nào đóng vai trò là "điểm nghẽn chi phối" (Dominant Bottleneck) bẻ gãy quá trình học của DRL?** Yếu tố nào gây suy thoái hiệu suất mẫu ($Sample\ Efficiency$), yếu tố nào phá vỡ khả năng tổng quát hóa ($Generalization\ Gap$)?
2. **Giải Phẫu Sự Biến Dạng Hành Vi & Động Học Học Tập (Learning Dynamics & Behavioral Distortion):**
   Môi trường tác động lên "bên trong bộ não" của agent như thế nào trong suốt quá trình tối ưu hóa?
   - Gradient bị triệt tiêu do không bao giờ chạm đích (Vanishing Gradient / Signal Starvation)?
   - Entropy của chính sách bị sụp đổ quá sớm khiến agent tê liệt khả năng khám phá (Premature Entropy Collapse)?
   - Hay hàm giá trị $Q(s, a)$ bị chệch hướng nghiêm trọng (Value Estimation Divergence), dẫn đến việc agent chọn đứng yên (*Premature Freezing*) hoặc dao động qua lại vô nghĩa (*Dithering*) để tránh phạt va chạm?
3. **Thử Thách & Xác Định Giới Hạn Của Các Phương Pháp Can Thiệp Kinh Điển (Stress-Testing Canonical Mitigations):**
   Cộng đồng DRL đã phát triển các phương pháp chuẩn mực để giải quyết bài toán môi trường phức tạp và thiếu tín hiệu:
   - **Curriculum Learning (CL):** Tăng dần độ khó của môi trường từ dễ đến khó.
   - **Hindsight Experience Replay (HER / Goal Relabeling):** Tái gán mục tiêu từ trải nghiệm thất bại để tạo tín hiệu thưởng nhân tạo.
   - **Potential-Based Reward Shaping (PBRS):** Bơm gradient thế năng bảo toàn chính sách tối ưu.
   $\to$ **Trọng tâm học thuật:** *Các phương pháp này cải thiện được yếu tố nào, cải thiện đến đâu, và đâu là ranh giới môi trường mà tại đó các phương pháp kinh điển hoàn toàn sụp đổ? Khi nào DRL có thể tự vượt qua bằng thuật toán, và khi nào bắt buộc phải tái cấu trúc môi trường hoặc sử dụng kiến trúc lai (Hybrid Planning)?*

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              KHUNG LÝ LUẬN KHOA HỌC CỐT LÕI                                     │
│                                                                                                 │
│   CÁC ĐẶC TRƯNG MÔI TRƯỜNG                                                                      │
│   ┌────────────────────────────────────────────────────────────────────────────┐                │
│   │ • Quy mô diện tích & Topology bản đồ (C_topo)                              │                │
│   │ • Chiều dài chuỗi quyết định / Cự ly Start-Goal (d_bfs)                    │                │
│   │ • Ràng buộc hình học thân xe / Tỷ lệ thắt nghẽn cửa hẹp (Footprint Ratio)  │                │
│   │ • Mật độ phản hồi thưởng tự nhiên (Sparsity)                               │                │
│   └─────────────────────────────────────┬──────────────────────────────────────┘                │
│                                         │ (Tác động lên)                                        │
│                                         ▼                                                       │
│   ĐỘNG HỌC HỌC TẬP & HÀNH VI CỦA DRL AGENTS                                                     │
│   ┌────────────────────────────────────────────────────────────────────────────┐                │
│   │ • Policy Entropy Dynamics: Tốc độ đông cứng thăm dò                        │                │
│   │ • Value Estimation Error: Sự méo mó của hàm giá trị Q(s,a)                 │                │
│   │ • Failure Modes: Dithering loops, Freezing, Corner Trapping                │                │
│   └─────────────────────────────────────┬──────────────────────────────────────┘                │
│                                         │ (Được can thiệp bởi)                                  │
│                                         ▼                                                       │
│   CÁC GIẢI PHÁP CAN THIỆP KINH ĐIỂN & ĐỊNH LƯỢNG GIỚI HẠN SỤP ĐỔ                                │
│   ┌────────────────────────────────────────────────────────────────────────────┐                │
│   │ [Curriculum Learning] ──► [Hindsight Experience Replay] ──► [Potential PBRS]│                │
│   │                ĐÂU LÀ RANH GIỚI MÔI TRƯỜNG MÀ CÁC GIẢI PHÁP THẤT BẠI?      │                │
│   └────────────────────────────────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. HỆ THỐNG BIẾN THỰC NGHIỆM VÀ 3 TRỤ CỘT METRICS CHUẨN

### 2.1. Nhóm Biến Độc Lập: 4 Nhóm Đặc Trưng Môi Trường & 3 Phương Pháp Can Thiệp

| Ký hiệu | Tên Yếu Tố | Mức Khảo Sát (Levels) | Ánh Xạ Kỹ Thuật Trong NavSim | Ý Nghĩa Học Thuật |
| :---: | :--- | :--- | :--- | :--- |
| **$V_1$** | **Chuỗi Quyết Định & Tỷ Lệ Đói Tín Hiệu ($d_{bfs}, sa$)** | `easy` ($sa \ge 0.15$), `medium` ($0.05 \le sa < 0.15$), `hard` ($0 < sa < 0.05$), `starved` ($sa = 0$) | `set_task_filter(level=...)` | Đo lường mức độ suy biến tín hiệu do chiều dài chuỗi hành động ngẫu nhiên gây ra ($sa\_uniform$). |
| **$V_2$** | **Mật Độ Phản Hồi Thưởng Môi Trường (Reward Density)** | **R1:** Pure Sparse, **R3:** Distance Dense, **R4:** Progress Dense, **R6:** Potential Shaped | [`RewardFactory`](file:///C:/SIGNAL-STARVATION/src/rewards/reward_factory.py) | Khảo sát từ môi trường thưởng cực thưa tự nhiên đến môi trường có tín hiệu dẫn đường liên tục. |
| **$V_3$** | **Quy Mô Diện Tích & Cấu Trúc Topology ($C_{topo}$)** | **Small** ($< 30\text{m}^2$), **Medium** ($30-50\text{m}^2$), **Large** ($\ge 50\text{m}^2$) | Lọc bản đồ theo diện tích di chuyển `main_m2` từ profiler | Khảo sát sự bùng nổ không gian trạng thái $|S|$ và sự ngăn cách của các vách ngăn phòng. |
| **$V_4$** | **Ràng Buộc Hình Học Thân Xe & Động Học (Footprint Constraint)** | **TurtleBot 4** (nhỏ tròn $r=0.17\text{m}$), **JetAuto Pro** (vừa, toàn hướng omni), **Clearpath Jackal** (lớn chữ nhật $0.51 \times 0.43\text{m}$) | Datasets: `grid-turtlebot4-v2`, `grid-jetauto_pro-v2`, `grid-clearpath_jackal-v2` | Khảo sát tỷ lệ co rút không gian tự do ($\mathcal{S}_{free} = \mathcal{M} \ominus \text{Footprint}$) và các nút thắt hình học tại cửa hẹp. |
| **$V_5$** | **Chiều Dài Hành Trình Tuyệt Đối (Path Length)** | `1-3m` (ngắn), `3-6m` (vừa), `6-10m` (dài), `10m+` (rất dài) | `set_task_filter(dist_bin=...)` | Khảo sát sự tích lũy sai số quyết định theo cự ly vật lý. |
| **$V_6$** | **Chiến Lược Can Thiệp Kinh Điển (Mitigation Strategies)** | **M0:** Không can thiệp (Baseline), **M1:** Curriculum Learning (CL), **M2:** Goal Relabeling (HER), **M3:** Potential Shaping (PBRS) | Custom Training Callbacks & Replay Wrappers | Thử thách khả năng "giải cứu" của từng giải pháp kinh điển trước các điểm nghẽn môi trường. |

---

### 2.2. Nhóm Biến Phụ Thuộc: 3 Trụ Cột Hiệu Suất Cốt Lõi & Metrics Động Học Nội Tại

Mọi đánh giá định lượng được kiểm định nghiêm ngặt trên **Validation Split (20 maps, 400 tasks)** và **Test Split (20 maps, 400 tasks hoàn toàn mới)**:

#### Trụ Cột 1: Độ Chính Xác & Tỷ Lệ Hoàn Thành Nhiệm Vụ (Accuracy & Task Completion)
- **$M_1$ — Success Rate ($SR$):** Tỷ lệ nhiệm vụ robot điều hướng thành công chạm đích.
- **$M_2$ — Cumulative Return ($G$):** Tổng phần thưởng tích lũy trung bình mỗi episode.
- **$M_{1b}$ — Success weighted by Path Length ($SPL$):** Thước đo chuẩn quốc tế về độ tối ưu đường đi ($SPL \le SR$).

#### Trụ Cột 2: Khả Năng Tổng Quát Hóa (Generalization Capability)
- **$M_3$ — Generalization Gap ($\Delta_{gen} = SR_{val} - SR_{test}$):** Độ suy giảm hiệu năng khi đối mặt với địa hình kiểm thử chưa từng thấy trong quá trình huấn luyện (đo lường mức độ overfitting cấu trúc hình học quen thuộc).
- **$M_{3b}$ — Generalization SPL Gap ($\Delta_{SPL} = SPL_{val} - SPL_{test}$):** Đo lường sự méo mó của đường đi trên môi trường mới.

#### Trụ Cột 3: Hiệu Suất Mẫu & Tốc Độ Huấn Luyện (Sample Efficiency & Training Time)
- **$M_4$ — Sample Efficiency ($SE_{50}$):** Số bước tương tác môi trường (timesteps) tối thiểu để đạt $SR \ge 50\%$. Nếu sau toàn bộ ngân sách vẫn không đạt $50\%$, $SE_{50} = \infty$ (đại diện cho sụp đổ học tập).
- **$M_5$ — Training Throughput & Wall Time ($FPS$ & $T_{wall}$):** Tốc độ thực thi trên GPU NVIDIA RTX 3050 Laptop GPU.

#### Nhóm Metrics Giải Phẫu Động Học Nội Tại (Internal Learning Dynamics & Behavioral Metrics)
- **$D_1$ — Policy Entropy Decay Rate ($\Delta \mathcal{H}_\pi / \Delta t$):** Tốc độ suy giảm entropy của chính sách, phát hiện hiện tượng thăm dò bị đông cứng sớm.
- **$D_2$ — Value Estimation Error ($\mathbb{E}[|Q(s, a) - V^*(s)|]$):** Sai số ước lượng hàm giá trị, đo lường sự ảo tưởng hoặc phẳng lì của gradient.
- **$D_3$ — Dithering & Collision Index ($CI$):** Tỷ lệ phần trăm các bước đi bị rơi vào vòng lặp dao động qua lại tại chỗ hoặc va chạm liên hoàn tại mép cửa hẹp.

---

## 3. THIẾT KẾ CHI TIẾT 4 KHỐI THỰC NGHIỆM HỌC THUẬT

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              LỘ TRÌNH 4 KHỐI THỰC NGHIỆM HỌC THUẬT                              │
│                                                                                                 │
│   [KHỐI 1: BÓC TÁCH ĐIỂM NGHẼN CHI PHỐI]                                                        │
│   Đo lường độ nhạy và phân rã mức độ ảnh hưởng của từng đặc trưng môi trường:                   │
│   Topology ($V_3$) vs Cự ly ($V_5$) vs Thân xe ($V_4$) vs Mật độ thưởng ($V_2$)                 │
│            │                                                                                    │
│            ▼                                                                                    │
│   [KHỐI 2: GIẢI PHẪU SỰ BIẾN DẠNG HÀNH VI & ĐỘNG HỌC HỌC TẬP]                                   │
│   Phân tích cơ chế suy thoái nội tại: Entropy Collapse, Value Divergence, Dithering             │
│            │                                                                                    │
│            ▼                                                                                    │
│   [KHỐI 3: THỬ THÁCH GIỚI HẠN CỦA CÁC GIẢI PHÁP KINH ĐIỂN]                                     │
│   Stress-test: Curriculum Learning, Hindsight Experience Replay (HER), Potential Shaping        │
│   ──► Xác định ranh giới mà từng giải pháp thất bại!                                            │
│            │                                                                                    │
│            ▼                                                                                    │
│   [KHỐI 4: TỔNG HỢP NGUYÊN LÝ THIẾT KẾ CHO PHYSICAL AI]                                         │
│   Xác định ranh giới: Vùng DRL tự giải quyết vs Vùng bắt buộc can thiệp môi trường / Hybrid     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1. KHỐI THỰC NGHIỆM 1: Định Lượng Điểm Nghẽn & Phân Tích Độ Nhạy Khách Quan (Factor Attribution & Sensitivity Analysis)
- **Mục tiêu khoa học:** Loại bỏ sự ghép cặp chủ quan, áp dụng nguyên lý thiết kế thực nghiệm chuẩn (Design of Experiments - DoE) để trả lời câu hỏi: *Trong các đặc trưng môi trường (Cự ly, Thân xe, Topology, Mật độ thưởng), yếu tố nào là "thủ phạm chính" phá vỡ Success Rate ($M_1$), yếu tố nào nới rộng Generalization Gap ($M_5$), và yếu tố nào triệt tiêu Sample Efficiency ($M_4$)?*
- **Phương pháp luận:** Sử dụng **Điểm Neo Quy Chuẩn (Anchor Reference Design)** kết hợp **Quét Đơn Biến (Main Effects)** và **Đo 2 Tương Tác Cốt Lõi (Critical 2-Way Interactions)**, sau đó phân rã phương sai bằng **Factorial ANOVA / Sobol Sensitivity Analysis**.

#### 1. Thiết Lập Điểm Neo Quy Chuẩn (Anchor Baseline)
Mọi phép đo lường độ nhạy đều được so sánh với một môi trường chuẩn đại diện cho điều kiện vận hành phổ biến:
- **Robot chuẩn:** `TurtleBot 4` (chuẩn công nghiệp, $r=0.17\text{m}$, vi sai)
- **Bản đồ chuẩn:** `Medium Maps` ($30 - 50\text{m}^2$)
- **Hành trình chuẩn:** `Medium Path` ($3 - 6\text{m}$)
- **Hàm thưởng chuẩn:** `R1 Pure Sparse` (môi trường tự nhiên không can thiệp shaping)

---

#### 2. Ma Trận 1A: Quét Đơn Biến Quanh Điểm Neo (One-Factor-At-A-Time Main Effects — 11 configs)
Giữ nguyên điểm neo chuẩn, lần lượt biến thiên **từng yếu tố độc lập** để đo độ dốc suy thoái thuần khiết của từng biến lên 3 metrics chuẩn:

| Biến Biến Thiên | Các Mức Khảo Sát | Điều Kiện Cố Định (Điểm Neo) | Số Configs |
| :--- | :--- | :--- | :---: |
| **Cự ly ($d_{bfs}$)** | `1-3m`, `3-6m` *(neo)*, `6-10m`, `10m+` | TB4, Medium Map, Sparse R1 | **4** |
| **Thân xe (Robot)** | `TurtleBot 4` *(neo)*, `JetAuto Pro`, `Clearpath Jackal` | Medium Path, Medium Map, Sparse R1 | **3** *(1 trùng neo)* |
| **Bản đồ ($C_{topo}$)** | `Small (<30m²)`, `Medium` *(neo)*, `Large (≥50m²)` | TB4, Medium Path, Sparse R1 | **3** *(1 trùng neo)* |
| **Hàm thưởng (Reward)**| `R1 Sparse` *(neo)*, `R3 Distance`, `R4 Progress`, `R6 Potential` | TB4, Medium Path, Medium Map | **4** *(1 trùng neo)* |
| **Tổng Configs 1A** | *(Đo Main Effect thuần khiết xuất biểu đồ Tornado Plot)* | — | **11 configs** |

$\to$ **Đầu ra Ma trận 1A:** Biểu đồ **Tornado Sensitivity Plot** xếp hạng độ dốc ảnh hưởng của 4 biến lên $M_1$ ($SR$), $M_4$ ($SE_{50}$), và $M_5$ ($\Delta_{gen}$).

---

#### 3. Ma Trận 1B: Phân Tích 2 Tương Tác Cốt Lõi Có Cơ Sở Lý Thuyết (Critical 2-Way Interactions)
Đo lường mức độ các yếu tố môi trường **khuếch đại tác hại tiêu cực của nhau**:

##### Tương Tác 1: Kích Thước Thân Xe $\times$ Chiều Dài Hành Trình (3 $\times$ 4 = 12 configs)
- *Cơ sở lý thuyết:* Robot to (Jackal) có thể không sao ở đường ngắn $1-3\text{m}$, nhưng khi hành trình dài $10\text{m}+$ đi qua nhiều cửa hẹp, xác suất tích lũy va chạm tăng vọt phi tuyến tính.
- *Mục tiêu:* Đo lường mức độ thân xe cồng kềnh khuếch đại sự sụp đổ khi cự ly dài ra.

| Robot Model \ Path Length | `1-3m` (Ngắn) | `3-6m` (Vừa) | `6-10m` (Dài) | `10m+` (Rất dài) |
| :--- | :---: | :---: | :---: | :---: |
| **TurtleBot 4** (Nhỏ tròn) | Config 1 | Config 2 *(neo)* | Config 3 | Config 4 |
| **JetAuto Pro** (Toàn hướng) | Config 5 | Config 6 | Config 7 | Config 8 |
| **Clearpath Jackal** (Lớn) | Config 9 | Config 10 | Config 11 | Config 12 |

##### Tương Tác 2: Mức Đói Tín Hiệu Nhiệm Vụ $\times$ Mật Độ Thưởng (4 $\times$ 4 = 16 configs)
- *Cơ sở lý thuyết:* Reward Shaping có cứu được agent ở mọi bài toán không? Hay nó chỉ có tác dụng ở bài toán Easy/Medium, còn khi rơi vào Starved ($sa\_uniform = 0$) thì chính Reward Shaping cũng bị vô hiệu hóa?
- *Mục tiêu:* Xác định ranh giới chuyển pha (Phase Transition) của các dạng hàm thưởng theo độ đói tín hiệu thực tế.

| Task Level \ Hàm Thưởng | **R1: Pure Sparse** | **R3: Distance** | **R4: Progress** | **R6: Potential** |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** ($sa \ge 0.15$) | Config 13 | Config 14 | Config 15 | Config 16 |
| **Medium** ($0.05 \le sa < 0.15$) | Config 17 | Config 18 | Config 19 | Config 20 |
| **Hard** ($0 < sa < 0.05$) | Config 21 | Config 22 | Config 23 | Config 24 |
| **Starved** ($sa = 0$) | Config 25 | Config 26 | Config 27 | Config 28 |

---

#### 4. Tổng Hợp Quy Mô Khối 1 & Phân Bổ Phương Sai (ANOVA Factor Attribution)
- **Tổng số cấu hình độc lập Khối 1:** **32 configs** (sau khi khử trùng lặp giữa 1A và 1B).
- **Quy mô thực nghiệm:** 32 configs $\times$ 2 thuật toán (PPO, DQN) $\times$ 5 seeds = **320 runs** (~5.0 giờ trên GPU RTX 3050).
- **Kết quả đầu ra của Khối 1:** Mô hình phân tích phương sai đa biến (**Three-way Factorial ANOVA**) phân rã tổng phương sai suy giảm:
  $$\text{Total Variance} = \text{SS}_{Path} + \text{SS}_{Robot} + \text{SS}_{Topology} + \text{SS}_{Reward} + \text{SS}_{Interactions} + \epsilon$$
  Xuất ra bảng xếp hạng khoa học không thiên vị khẳng định yếu tố nào là **điểm nghẽn số 1 chi phối từng metric**, làm tiền đề chính xác để Khối 3 đưa các giải pháp can thiệp (Curriculum / HER) vào giải quyết đúng bệnh!

---

### 3.2. KHỐI THỰC NGHIỆM 2: Giải Phẫu Động Học Học Tập & Sự Biến Dạng Hành Vi (Learning Dynamics & Behavioral Distortion)
- **Câu hỏi khoa học:** *Các yếu tố môi trường tiêu cực làm biến dạng quá trình học và hành vi của agent thông qua cơ chế nào bên trong mạng nơ-ron?*
- **Phương pháp luận:** Trích xuất log nội tại từng 1,000 steps về: Policy Entropy, Value Estimation, và quỹ đạo di chuyển thực tế.

#### Các Hiện Tượng Lỗi Cần Kiểm Chứng & Đo Lường:
1. **Premature Entropy Collapse (Đông cứng thăm dò):** 
   - *Hiện tượng:* Trong môi trường có thân xe lớn hoặc hành lang hẹp, penalty va chạm lặp lại quá nhanh khiến entropy của policy giảm dốc đứng ($\mathcal{H} \to 0$) trước khi agent tìm thấy goal lần đầu tiên.
   - *Đo lường:* Vẽ đồ thị đường cong Entropy Decay $\mathcal{H}(t)$ so sánh giữa TurtleBot 4 và Clearpath Jackal.
2. **Value Divergence & Overestimation (Ảo tưởng hàm giá trị):**
   - *Hiện tượng:* Dưới hàm thưởng Dense R3 (Distance-based), sai số ước lượng $Q(s, a)$ tích lũy theo chiều dài hành trình, khiến agent ảo tưởng rằng mình đang tiến bộ ngay cả khi đang bị kẹt ở góc tường.
   - *Đo lường:* Sai số $|Q(s, a) - V^*(s)|$ theo từng bước thời gian.
3. **Phân Loại Hành Vi Lệch Chuẩn (Failure Typology):**
   - *Dithering:* Tỷ lệ episode mà robot lặp lại 2 hành động đối nghịch liên tiếp $\ge 10$ lần tại nút thắt.
   - *Premature Freezing:* Tỷ lệ episode robot dừng hẳn hoặc quay tròn tại chỗ sau khi va chạm nhẹ.

---

### 3.3. KHỐI THỰC NGHIỆM 3: Thử Thách & Đánh Giá Giới Hạn Của Các Giải Pháp Can Thiệp Kinh Điển (Stress-Testing Canonical Mitigations)
Đây là **trọng tâm học thuật phản ánh năng lực nghiên cứu của đề tài**: Đưa các giải pháp kinh điển của cộng đồng DRL vào "thử lửa" trước các điểm nghẽn môi trường đã bóc tách ở Khối 1 & 2.

#### 3 Giải Pháp Can Thiệp Được Kiểm Chứng:
1. **Curriculum Learning (CL — $M_1$):** Tăng dần độ khó môi trường (từ cự ly ngắn $\to$ dài, từ hành lang rộng $\to$ hẹp, từ $sa$ cao $\to$ thấp).
2. **Hindsight Experience Replay / Goal Relabeling (HER — $M_2$):** Tái gán mục tiêu đạt được từ các bước đi dở dang để tạo tín hiệu thưởng nhân tạo từ thất bại.
3. **Potential-Based Reward Shaping (PBRS — $M_3$):** Bơm gradient thế năng bảo toàn chính sách tối ưu.

#### Ma Trận Thử Thách Giới Hạn Can Thiệp (Stress-Test Matrix):

| Điểm Nghẽn Môi Trường (Identified Bottlenecks) | Baseline (DRL Thuần) | + Potential Shaping (PBRS) | + Curriculum Learning (CL) | + Goal Relabeling (HER) | + Phối Hợp (CL + HER) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1: Đói Tín Hiệu Do Cự Ly Rất Dài ($10\text{m}+$)** | Đo $SR, \Delta_{gen}, SE_{50}$ | Đánh giá hiệu quả | Đánh giá hiệu quả | Đánh giá hiệu quả | Đánh giá hiệp đồng |
| **B2: Nút Thắt Thân Xe Lớn (Jackal Footprint)** | Đo $SR, \Delta_{gen}, SE_{50}$ | Đánh giá hiệu quả | Đánh giá hiệu quả | Đánh giá hiệu quả | Đánh giá hiệp đồng |
| **B3: Bẫy Topology Ngõ Cụt Phức Tạp (Dead-ends)**| Đo $SR, \Delta_{gen}, SE_{50}$ | Đánh giá hiệu quả | Đánh giá hiệu quả | Đánh giá hiệu quả | Đánh giá hiệp đồng |

- **Giả thuyết khoa học cần kiểm chứng ($H_3$):**
  - **Giới hạn của HER:** HER cực kỳ hiệu quả đối với bài toán cự ly dài (B1) vì nó biến các trajectory dài không tới đích thành các bài học định hướng; nhưng **HER hoàn toàn bất lực trước nút thắt thân xe lớn (B2)** vì các goal ảo được tái gán vẫn nằm trước cửa hẹp, không cung cấp được thông tin làm sao để thân xe lách qua khe hẹp mà không chạm tường.
  - **Giới hạn của Curriculum:** Curriculum giải phóng được hiện tượng Entropy Collapse ở Khối 2, giúp tăng $SR$ trên bài toán thân xe lớn (B2), nhưng làm tăng thời gian huấn luyện tổng thể ($T_{wall}$).
  - **Giới hạn của PBRS:** PBRS giúp tăng tốc hội tụ nhưng không làm giảm Generalization Gap trên bản đồ kiểm thử có vách ngăn mới.

---

### 3.4. KHỐI THỰC NGHIỆM 4: Tổng Hợp Ranh Giới Khả Thi & Nguyên Lý Thiết Kế Cho Physical AI (Synthesis & Design Guidelines)
- **Mục tiêu:** Rút ra bản đồ ranh giới khoa học (Feasibility Boundary Map) đóng góp cho cộng đồng nghiên cứu:
  1. **Vùng Giải Quyết Được Bằng Thuật Toán (Algorithm-Solvable Zone):** Các cấu hình môi trường mà DRL thuần túy kết hợp CL hoặc HER có thể tự giải quyết triệt để ($SR > 80\%, \Delta_{gen} < 10\%$).
  2. **Vùng Bắt Buộc Can Thiệp Môi Trường / Phần Cứng (Environment-Redesign Zone):** Các ngưỡng hình học mà tại đó DRL sụp đổ do rào cản vật lý (ví dụ: khi tỷ lệ kích thước thân xe / độ rộng cửa $> 0.85$), đòi hỏi phải tái thiết kế độ rộng hành lang hoặc chuyển giao sang kiến trúc lai (Hybrid Planning + DRL).

---

## 4. KẾ HOẠCH TÀI NGUYÊN & PHÂN BỔ THỜI GIAN THỰC NGHIỆM

Toàn bộ hệ thống chạy thực nghiệm được thiết kế tự động, lưu trữ liên tục (crash-safe) và tối ưu hóa tối đa trên phần cứng GPU NVIDIA RTX 3050 Laptop GPU:

| Khối Thực Nghiệm | Nội Dung Khảo Sát | Cấu Hình | Thuật Toán | Số Seeds | Tổng Số Runs | Thời Gian Ước Tính (RTX 3050) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Khối 1: Sensitivity & Attribution** | Quét đơn biến quanh Điểm Neo (1A) + 2 Tương tác cốt lõi (1B) | 32 | PPO, DQN | 5 | **320 runs** | ~5.0 giờ |
| **Khối 2: Learning Dynamics** | Trích xuất Entropy, Value Error, Dithering | (Tích hợp trong Khối 1 & 3) | PPO, DQN | 5 | — | (Ghi log song song) |
| **Khối 3: Stress-Testing Mitigations** | Thử thách HER, Curriculum, Potential Shaping | 15 | PPO, DQN | 5 | **150 runs** | ~2.5 giờ |
| **TỔNG CỘNG TOÀN DỰ ÁN** | Khảo sát toàn diện độ nhạy, hành vi & giới hạn can thiệp | **47 configs** | **2 algos** | **5** | **470 runs** | **~7.5 giờ** |

> [!TIP]
> **Khả năng thực thi vượt trội:** Nhờ tốc độ mô phỏng cực cao của NavSim (~36,000 FPS) kết hợp với 5 tiến trình worker song song chạy trực tiếp trên GPU CUDA (PyTorch 2.5.1), toàn bộ 470 runs của công trình nghiên cứu quy mô này có thể hoàn tất trong vòng **~7.5 giờ chạy liên tục**.

---

## 5. QUY TRÌNH PHÂN TÍCH THỐNG KÊ & CHỨNG MINH KHOA HỌC

Để đảm bảo tính thuyết phục tuyệt đối của một công trình nghiên cứu khoa học chất lượng cao:

1. **Độ Tin Cậy Thống Kê ($N = 5$ Random Seeds):**
   - Mọi kết quả công bố đều có khoảng tin cậy 95% ($\text{CI}_{95\%}$) được tính toán bằng kỹ thuật Bootstrap phi tham số ($B = 10,000$ resamples).
2. **Phân Tích ANOVA & Effect Size ($\eta^2$):**
   - Sử dụng Three-way ANOVA để định lượng tỷ lệ phương sai của $SR$, Generalization Gap và Sample Efficiency được giải thích bởi từng yếu tố môi trường, khẳng định yếu tố nào là điểm nghẽn có ý nghĩa thống kê ($p < 0.001$).
3. **Kiểm Định Giới Hạn Của Giải Pháp Can Thiệp:**
   - Dùng **Tukey HSD Post-hoc Test** và **Mann-Whitney U Test** để chứng minh sự khác biệt có ý nghĩa thống kê giữa Baseline vs HER vs Curriculum vs Potential Shaping trên từng nhóm điểm nghẽn cụ thể.

---

## 6. TIÊU CHÍ HOÀN THÀNH (DEFINITION OF DONE - DoD CHECKLIST)

- [x] **Xóa bỏ tính hiển nhiên của hiện tượng:** Đặt trọng tâm vào việc định lượng mức độ ảnh hưởng của các yếu tố môi trường lên động học học tập và sự biến dạng hành vi của DRL agents.
- [x] **Giải phẫu cơ chế suy thoái nội tại:** Phân tích Policy Entropy Collapse, Value Estimation Divergence, và các dạng hành vi lỗi (Dithering, Freezing).
- [x] **Stress-test các giải pháp can thiệp kinh điển:** Kiểm chứng năng lực và chỉ ra ranh giới sụp đổ của Curriculum Learning, Hindsight Experience Replay (HER), và Potential-based Reward Shaping.
- [x] **Bảo đảm 3 trụ cột metrics chuẩn:** Độ chính xác ($SR, G, SPL$), Khả năng tổng quát hóa ($\Delta_{gen}$), và Hiệu suất mẫu ($SE_{50}$).
- [x] **Kế hoạch kiểm thử $N = 5$ seeds độc lập** với ngân sách thời gian thực thi khả thi (~8.5 giờ trên GPU RTX 3050).
- [ ] **Người dùng review và chốt phê duyệt bản thiết kế ma trận thực nghiệm.**
