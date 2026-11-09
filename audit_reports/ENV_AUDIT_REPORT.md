# BÁO CÁO AUDIT MÔI TRƯỜNG & BASELINE PROFILING
## Environment Profiling & Baseline Verification Report (TASK-104)

- **Dự án:** Signal Starvation — Khảo sát và Giảm thiểu Hiện tượng Thiếu đói Tín hiệu Thưởng trong Học tăng cường Định vị Robot (DRL Physical AI)
- **Git Branch:** `feature/env-audit-baseline`
- **Mục tiêu:** Tổng hợp kết quả audit kỹ thuật từ TASK-101 (Resource Discovery), TASK-102 (Smoke Test), và TASK-103 (Map Profiler & Topological Complexity).
- **Trạng thái:** Hoàn tất, sẵn sàng phê duyệt và merge vào `dev`.

---

## 1. BẢNG THÔNG SỐ ĐỊNH LƯỢNG MÔI TRƯỜNG TỔNG HỢP

Bảng sau đây tổng hợp các đặc tính kỹ thuật cơ bản và các tham số thực nghiệm được đo lường trực tiếp từ hệ thống thực tế:

| Thuộc tính | Giá trị Thực tế | Nguồn gốc / Phương pháp xác minh | Ghi chú & Ý nghĩa thực nghiệm |
| :--- | :--- | :--- | :--- |
| **Env type** | `GridNav-v0` (Discrete) | Smoke test ([`audit_reports/smoke_test_gridnav.py`](file:///C:/SIGNAL-STARVATION/audit_reports/smoke_test_gridnav.py)) | Mô phỏng lưới 2D tối ưu hóa cao độ, tương thích Gymnasium chuẩn |
| **Obs shape** | `Box(0.0, 10.0, (22,), float32)` | Smoke test | 22 tia LiDAR mô phỏng 1D vector (khoảng cách vật cản gần nhất) |
| **Action space** | `Discrete(4)` | Smoke test | 4 hướng di chuyển: 0: Lên, 1: Phải, 2: Xuống, 3: Trái |
| **Step FPS** | **~36,000 FPS** (Pure sim) <br> ~1,000 FPS (với full step logging) | Benchmark đo thời gian 1,000 steps | Cho phép train 500k timesteps chỉ mất ~15–20 giây CPU |
| **Reset Time** | ~0.0098 s (~9.8 ms) | 50 lần đo reset liên tiếp | Chi phí overhead thấp cho episode ngắn |
| **Reward** | `0.0` by default (100% sparse) | Thiết kế MDP mặc định của NavSim | Chỉ nhận `+1.0` khi chạm đích; hoàn hảo để khảo sát Signal Starvation |
| **$\rho_{sig}$ proxy** | `1 - sa_uniform` per task | Dataset labels ([`audit_reports/task_complexity_profile.csv`](file:///C:/SIGNAL-STARVATION/audit_reports/task_complexity_profile.csv)) | Tỷ lệ "đói tín hiệu": xác suất tác tử ngẫu nhiên thất bại trong việc chạm goal |
| **$C_{topo}$ proxy** | `1 - main_frac` per map | Grid JSON ([`audit_reports/map_complexity_profile.csv`](file:///C:/SIGNAL-STARVATION/audit_reports/map_complexity_profile.csv)) | Tỷ lệ ô bị cô lập/chia cắt khỏi thành phần liên thông chính |
| **Level labels** | `easy` / `medium` / `hard` / `starved` | Dataset labels pre-computed | 4 phân tầng độ khó dựa trên xác suất chạm đích ngẫu nhiên |
| **$d_{bfs}$ range** | **[8, 67]** cells, mean = **23.1** cells | Smoke test & Task Profiler | Đường đi ngắn nhất (BFS steps) từ vị trí Start đến Goal |
| **$sa\_uniform$ range** | **[~0, ~0.59]**, mean = **0.111** (11.1%) | Smoke test & Task Profiler | Xác suất policy ngẫu nhiên đều chạm goal trước khi hết budget |

---

## 2. ĐIỂM CHỐT QUAN TRỌNG VỀ MẶT KHOA HỌC: $\rho_{sig}$ ĐÃ ĐƯỢC PRE-COMPUTE TỪ MDP

> [!IMPORTANT]
> **Điểm mấu chốt của nghiên cứu:**
> Tập dữ liệu NavSim **đã tính toán sẵn (pre-computed)** chỉ số `sa_uniform` cho từng cặp task (Start, Goal). 
> 
> Đây chính là độ đo định lượng chính xác của **Signal Starvation ($\rho_{sig}$)** xuất phát từ bản chất hình học và xác suất chuyển trạng thái của MDP:
> $$\rho_{sig}(\tau) = 1 - P(\text{chạm đích} \mid \pi_{uniform}, \text{budget})$$
> 
> Việc này mang lại lợi thế khoa học vượt trội: **Không cần ước tính thủ công hoặc xấp xỉ mò mẫm**, ta có một biến độc lập khách quan, chuẩn xác để đo lường ngưỡng gãy đổ (collapse threshold) của các thuật toán DRL.

### Hai Trục Khảo Sát Độc Lập của Nghiên Cứu
Nhờ đặc thù này, nghiên cứu Signal Starvation được phân tách thành 2 trục độc lập:
1. **Trục A (Task Difficulty Selection - $\rho_{sig}^{task}$):** Lọc task theo các mức `easy`, `medium`, `hard`, `starved` thông qua API `env.unwrapped.set_task_filter(level=...)`.
2. **Trục B (Reward Signal Density - $\rho_{sig}^{reward}$):** Thiết kế và biến thiên các hàm thưởng từ cực thưa (Pure Sparse) đến cực dày (Distance-dense / Potential-shaped).

```
                     TRỤC B: MẬT ĐỘ TÍN HIỆU THƯỞNG (Reward Function)
                          Sparse (R1) ───► Potential/Dense (R6)
                           ┌───────────────────────────────┐
             easy (34.5%)  │ Baseline dễ                   │
TRỤC A:      medium (34.0%)│ Chuyển tiếp                   │
TASK LEVEL   hard (9.6%)   │ Ranh giới bắt đầu collapse    │
(sa_uniform) starved (21.9%)│ CRITICAL STARVATION ZONE      │
                           └───────────────────────────────┘
```

---

## 3. TỔNG HỢP TASK-101: TÀI NGUYÊN VÀ CẤU TRÚC DỮ LIỆU NAVSIM

### 3.1. Phân bổ dữ liệu 9 Datasets
Dataset chuẩn được lưu tại `F:\navsim\datasets`. Toàn bộ 9 robot models đều có các dataset tiền xử lý tương ứng:

| Dataset | Robot Model | Maps (Train / Val / Test / Tổng) | Tasks (Train / Val / Test / Tổng) | Tỷ lệ Starved (Train) |
| :--- | :--- | :---: | :---: | :---: |
| `grid-turtlebot4-v2` *(Dataset chính)* | TurtleBot 4 | 98 / 20 / 20 / **138** | 1,960 / 400 / 400 / **2,760** | **22.4%** |
| `grid-turtlebot4_lite-v2` | TB4 Lite | 97 / 22 / 21 / **140** | 1,940 / 440 / 420 / **2,800** | **22.3%** |
| `grid-turtlebot3_burger-v2` | TB3 Burger | 105 / 24 / 22 / **151** | 2,100 / 480 / 440 / **3,020** | **53.4%** |
| `grid-turtlebot3_waffle_pi-v2` | TB3 Waffle Pi | 102 / 23 / 22 / **147** | 2,040 / 460 / 440 / **2,940** | **27.0%** |
| `grid-jetauto_pro-v2` | JetAuto Pro | 102 / 21 / 20 / **143** | 2,040 / 420 / 400 / **2,860** | **23.9%** |
| `grid-agilex_limo-v2` | AgileX LIMO | 101 / 23 / 20 / **144** | 2,020 / 460 / 400 / **2,880** | **24.1%** |
| `grid-kobuki-v2` | Kobuki Base | 99 / 22 / 21 / **142** | 1,980 / 440 / 420 / **2,840** | **21.9%** |
| `grid-clearpath_jackal-v2` | Clearpath Jackal | 76 / 14 / 13 / **103** | 1,520 / 280 / 260 / **2,060** | **2.8%** |
| `grid-pal_tiago-v2` | PAL TIAGo | 60 / 12 / 8 / **80** | 1,200 / 240 / 160 / **1,600** | **1.0%** |
| **Tổng cộng** | **9 Robot Datasets** | **840 / 173 / 176 / 1,189** | **16,800 / 3,460 / 3,520 / 23,780** | — |

### 3.2. Đặc tính phần cứng của Robot chính (`turtlebot4`)
- **Footprint:** Dạng tròn, bán kính $r = 0.171\ \text{m}$ (đường kính $\approx 0.34\ \text{m}$, khớp hoàn hảo với kích thước ô lưới `cell_m = 0.34m`).
- **Chiều cao:** $0.351\ \text{m}$.
- **Vận tốc tối đa:** $v_{max} = 0.31\ \text{m/s}$, $\omega_{max} = 1.90\ \text{rad/s}$.
- **Cảm biến LiDAR:** Tầm quét $0.15\ \text{m} - 12.0\ \text{m}$, góc nhìn $360^\circ$ ($2\pi$).

---

## 4. TỔNG HỢP TASK-102: KẾT QUẢ SMOKE TEST CHÍNH THỨC TRÊN GRIDNAV-V0

Script kiểm thử [`audit_reports/smoke_test_gridnav.py`](file:///C:/SIGNAL-STARVATION/audit_reports/smoke_test_gridnav.py) đã thực thi kiểm chứng 1,000 bước agent ngẫu nhiên trên `grid-turtlebot4-v2`:

### 4.1. Kết quả thực thi
- **Thông lượng mô phỏng thuần:** Đạt **~36,000 FPS**.
- **Thông lượng khi ghi log đầy đủ (11 cột chỉ số):** Đạt **~1,000 FPS** (1,000 steps / 0.9996 s).
- **Tổng phần thưởng nhận được:** **0.0** (Xác nhận 100% sparse reward; agent ngẫu nhiên không thể tình cờ chạm đích trong 1,000 bước nếu không có shaping).
- **Tỷ lệ va chạm vật cản:** **34.7%** (347 / 1,000 bước).
- **Thời gian reset:** Trung bình $9.8\ \text{ms}$.

### 4.2. Bằng chứng kiểm định đã xuất
1. File dữ liệu bước: [`audit_reports/logs/gridnav_random_1000steps.csv`](file:///C:/SIGNAL-STARVATION/audit_reports/logs/gridnav_random_1000steps.csv) (1,000 dòng log chi tiết).
2. Hình ảnh render môi trường: [`audit_reports/logs/gridnav_episode.png`](file:///C:/SIGNAL-STARVATION/audit_reports/logs/gridnav_episode.png).
3. Đồ thị phân phối $sa\_uniform$ & $d_{bfs}$: [`audit_reports/logs/sa_uniform_distribution.png`](file:///C:/SIGNAL-STARVATION/audit_reports/logs/sa_uniform_distribution.png).

---

## 5. TỔNG HỢP TASK-103: PHÂN TÍCH ĐỘ PHỨC TẠP HÌNH HỌC VÀ NHIỆM VỤ

Bộ trích xuất [`src/experiment/map_profiler.py`](file:///C:/SIGNAL-STARVATION/src/experiment/map_profiler.py) đã xử lý toàn bộ $138$ maps và $2,760$ tasks trong `grid-turtlebot4-v2`:

### 5.1. Bảng phân loại quy mô không gian bản đồ ($C_{topo}$ theo `main_m2`)

| Nhóm Map | Ngưỡng phân vị (Tertile) | Ngưỡng tròn trực quan | Số maps (Tertile / Tròn) | Tỷ lệ (%) |
|---|---|---|---|---|
| **Small** | $\text{main\_m2} < 28.34\ \text{m}^2$ | $\text{main\_m2} < 30.00\ \text{m}^2$ | **46** / **51** | 33.3% / 37.0% |
| **Medium** | $28.34 \le \text{main\_m2} < 46.27\ \text{m}^2$ | $30.00 \le \text{main\_m2} < 50.00\ \text{m}^2$ | **46** / **50** | 33.3% / 36.2% |
| **Large** | $\text{main\_m2} \ge 46.27\ \text{m}^2$ | $\text{main\_m2} \ge 50.00\ \text{m}^2$ | **46** / **37** | 33.3% / 26.8% |

- **Diện tích di chuyển được (`main_m2`):** Dao động từ $11.02\ \text{m}^2$ đến $145.41\ \text{m}^2$ (Trung bình: $42.66 \pm 26.51\ \text{m}^2$).
- **Mật độ vật cản / phân mảnh (`obstacle_density = 1 - main_frac`):** Trung bình $0.0907$ (cao nhất $0.3812$).

### 5.2. Bảng phân loại Tasks theo độ khó môi trường ($\rho_{sig}$ / `sa_uniform`)
> **Nhắc lại:** Map lớn không đồng nghĩa với task khó (Start và Goal có thể cách nhau 2 bước trong map $100\ \text{m}^2$). Do đó, task-level là đơn vị phân loại độ khó thực chất.

| Cấp độ (Level) | Tiêu chí $sa\_uniform$ | Số lượng Tasks | Tỷ lệ (%) | $d_{bfs}$ trung bình | $sa\_uniform$ trung bình |
|---|---|---|---|---|---|
| **`easy`** | $sa\_uniform \ge 0.10$ | **953** | 34.5% | 9.4 ô | 0.294562 |
| **`medium`** | $0.001 \le sa\_uniform < 0.10$ | **938** | 34.0% | 21.1 ô | 0.026784 |
| **`hard`** | $0.0001 \le sa\_uniform < 0.001$ | **265** | 9.6% | 31.8 ô | 0.000433 |
| **`starved`** | $sa\_uniform < 0.0001$ | **604** | 21.9% | 44.2 ô | 0.000015 |
| **Tổng cộng** | — | **2,760** | **100.0%** | **23.1 ô** | **0.111420** |

### 5.3. Bằng chứng phân tích
- Profile bản đồ: [`audit_reports/map_complexity_profile.csv`](file:///C:/SIGNAL-STARVATION/audit_reports/map_complexity_profile.csv)
- Profile nhiệm vụ: [`audit_reports/task_complexity_profile.csv`](file:///C:/SIGNAL-STARVATION/audit_reports/task_complexity_profile.csv)
- Đồ thị phân phối bản đồ: [`audit_reports/map_distribution.png`](file:///C:/SIGNAL-STARVATION/audit_reports/map_distribution.png)
- Đồ thị phân phối nhiệm vụ: [`audit_reports/task_distribution.png`](file:///C:/SIGNAL-STARVATION/audit_reports/task_distribution.png)

---

## 6. KHUYẾN NGHỊ THỰC NGHIỆM CHO CÁC EPIC TIẾP THEO

1. **Thiết kế Reward (EPIC 2):**
   - Không cần thiết kế các cơ chế reward định tuyến nhân tạo phức tạp để tạo độ khó, vì dataset đã chia sẵn 4 tầng độ khó rõ rệt.
   - Tập trung so sánh 4 họ reward chính: **Sparse ($R_1$)**, **Collision-Penalized ($R_2$)**, **Progress Dense ($R_4$)**, và **Potential-Based Shaped ($R_6$)**.
2. **Setup Thực nghiệm Phase 1 (EPIC 5 & 6):**
   - Sử dụng `level` làm yếu tố phân tầng trong ma trận thực nghiệm: 
     $$\text{Task Level (4)} \times \text{Reward Function (4)} \times \text{Algorithms (2: PPO, DQN)} \times \text{Seeds (5)} = 160\ \text{runs}$$
   - Ngưỡng gãy đổ (policy collapse) được kỳ vọng sẽ xuất hiện rõ nhất tại nhóm `hard` ($sa\_uniform < 10^{-3}$) và `starved` ($sa\_uniform < 10^{-4}$) dưới hàm thưởng thưa.

---

## 7. DEFINITION OF DONE (DoD) CHECKLIST

- [x] Có bảng định lượng đối chiếu thuộc tính môi trường đầy đủ (Env, Obs, Action, FPS, Reward, $\rho_{sig}$, $C_{topo}$, $d_{bfs}$, $sa\_uniform$).
- [x] Làm rõ luận điểm khoa học về việc `sa_uniform` được pre-compute từ MDP đại diện cho $\rho_{sig}$.
- [x] Xuất đầy đủ dữ liệu audit CSV và đồ thị distribution trực quan.
- [x] Bộ unit test [`tests/test_map_profiler.py`](file:///C:/SIGNAL-STARVATION/tests/test_map_profiler.py) vượt qua 100% (3/3 passed).
- [ ] Báo cáo được Người dùng (User) xem xét và phê duyệt.
- [ ] Merge branch `feature/env-audit-baseline` vào branch `dev`.
