# Reward Design Taxonomy cho Deep RL Navigation 2D (A→B)

> Bản tổng hợp dựa trên các nguồn đã kiểm tra và kiến thức nền. Đây **không** phải taxonomy chính thức của một bài báo nào. Mục 7 ghi rõ nguồn nào đã đọc toàn văn và nguồn nào cần tự xác minh trước khi trích dẫn.

**Ký hiệu:** $d_t=\lVert p_t-p_B\rVert$ là khoảng cách tới đích; $\varepsilon$ là bán kính chấp nhận đích; $d_{obs}$ là khoảng cách tới chướng ngại gần nhất (LiDAR hoặc occupancy map); $\gamma$ là hệ số chiết khấu.

---

## 1. Tổng quan: 6 trục phân loại

| Trục | Câu hỏi | Các giá trị |
|---|---|---|
| A. Chức năng | Reward thưởng/phạt điều gì? | Goal, Progress, Safety, Efficiency, Smoothness, Social, Exploration |
| B. Mật độ | Phản hồi dày cỡ nào? | Sparse, Semi-sparse, Dense |
| C. Cách xây dựng | Reward đến từ đâu? | Hand-crafted, Potential-based, Curriculum/Relabel, Intrinsic, Learned |
| D. Cách kết hợp | Gộp các thành phần ra sao? | Additive, Multiplicative, Hierarchical, Constrained, Multi-objective |
| E. Bảo đảm lý thuyết | Có giữ policy tối ưu không? | Policy-invariant, Xấp xỉ, Không bảo đảm |
| F. Ngữ cảnh | Bài toán thuộc loại nào? | Quan sát, hành động, môi trường tĩnh/động, có/không bản đồ |

---

## 2. Trục A: Thành phần theo chức năng

| Thành phần | Công thức điển hình | Vai trò | Rủi ro |
|---|---|---|---|
| **Goal / terminal** | $+R_g$ nếu $d<\varepsilon$; $-R_c$ nếu va chạm; $-R_t$ nếu hết thời gian | Định nghĩa nhiệm vụ thật | Thưa; scale $R_g, R_c$ ảnh hưởng mạnh |
| **Progress** | $\alpha(d_{t-1}-d_t)$ | Dẫn hướng tới B | Dạng một chiều gây đi vòng (xem mục 5) |
| **Heading** | $-\beta\,\lvert\theta_t\rvert$ ($\theta$ là sai lệch hướng tới B) | Định hướng | Có thể kẹt trước chướng ngại |
| **Safety** | $-\lambda\max\big(0,\,(d_{safe}-d_{obs})/d_{safe}\big)$ | Giữ khoảng cách an toàn | Agent đứng yên hoặc quá thận trọng |
| **Efficiency** | $-c$ mỗi bước; phạt độ dài quỹ đạo | Ép đi nhanh, ngắn | Gây va chạm vội nếu $c$ lớn |
| **Smoothness** | $-\beta_1\lvert\Delta v\rvert-\beta_2\lvert\Delta\omega\rvert$ | Chuyển động trơn, khả thi cơ học | Làm chậm phản ứng né |
| **Social** | Phạt khi vào vùng proxemics của người/agent | Điều hướng đông người | Chỉ cần khi môi trường động |
| **Exploration** | $\beta/\sqrt{\hat N(x)}$ | Khám phá trạng thái mới | Không phải shaping; cần tinh chỉnh $\beta$ |

Cấu trúc ba tình huống (tới đích, va chạm, tiến về đích) là lõi của hầu hết thiết kế reward cho navigation dùng LiDAR (Lee & Jeong, 2023).

---

## 3. Trục B: Mật độ phản hồi

- **Sparse:** chỉ phản hồi ở terminal. Dạng nhị phân của HER: $r=-[\lVert p'-g\rVert>\varepsilon]$. Cần relabel hoặc exploration đủ mạnh.
- **Semi-sparse:** có mốc trung gian (waypoint, subgoal). Theo Ng et al. (1999), thưởng khi tới subgoal có thể suy ra từ một potential tăng bước nhảy mỗi khi qua subgoal.
- **Dense:** phản hồi mỗi bước (progress, safety, smoothness). Học nhanh nhưng dễ bị reward hacking.

---

## 4. Trục C: Cách xây dựng

1. **Hand-crafted:** tự thiết kế và cân chỉnh trọng số (phổ biến nhất).
2. **Potential-based shaping:** $F=\gamma\Phi(s')-\Phi(s)$. Với navigation: $\Phi(s)=-d(s,B)$, hoặc $-\text{geodesic}(s,B)$ khi có bản đồ. Biến thể: dynamic potential $\Phi(s,t)$, potential-based advice.
3. **Curriculum / goal relabeling:** HER (chiến lược `future`, $k=4$); curriculum theo khoảng cách hoặc độ dày chướng ngại.
4. **Intrinsic motivation:** count/pseudo-count, ICM, RND. Với ô lưới rời rạc, pseudo-count trùng số lần ghé thăm thật.
5. **Learned reward:** IRL/GAIL từ demonstration của planner (A*, DWA), preference learning, reward do LLM sinh (hướng gần đây, cần xác minh nguồn).

---

## 5. Trục D và E: Kết hợp và bảo đảm lý thuyết

### Cách kết hợp

- **Additive:** $r=\sum_i w_i r_i$ (phổ biến, nhạy với trọng số).
- **Multiplicative / gating:** ví dụ progress × hệ số an toàn.
- **Hierarchical / lexicographic:** an toàn ưu tiên hơn tiến độ.
- **Constrained (CMDP):** tối đa return với ràng buộc chi phí va chạm $\le d$ (Lagrangian, CPO).
- **Multi-objective:** Pareto giữa tốc độ, an toàn, độ mượt.

### Bảo đảm lý thuyết

- **Policy-invariant:** potential-based (Ng 1999) và biến đổi affine dương.
- **Không bảo đảm nhưng có chủ ý:** safety penalty, step penalty, smoothness. Chúng *thay đổi mục tiêu* thật.

> **Lưu ý:** $\alpha(d_{t-1}-d_t)$ thực chất là potential-based với $\Phi=-d$ và $\gamma=1$ vì nó telescoping. Dạng bị lỗi là **một chiều**, ví dụ $\alpha\max(0,d_{t-1}-d_t)$, vì chu trình khi đó có tổng dương. Đây chính là lỗi "đi vòng tròn" trong bài Ng 1999. Đây là suy luận từ định lý; nên tự kiểm tra bằng thực nghiệm.

---

## 6. Trục F: Ngữ cảnh 2D và các lỗi thường gặp

### Ngữ cảnh cần khai báo trong bài

- **Quan sát:** LiDAR, occupancy grid, hoặc tọa độ tương đối tới B.
- **Hành động:** rời rạc (DQN) hay liên tục $(v,\omega)$ (DDPG/SAC/PPO).
- **Môi trường:** tĩnh hay động; một agent hay đông người; có bản đồ hay mapless.
- **Điều kiện kết thúc:** $\varepsilon$, số bước tối đa, va chạm.

### Lỗi reward thường gặp (nên làm ablation)

| Hiện tượng | Nguyên nhân thường gặp |
|---|---|
| Đi vòng tròn gần xuất phát | Progress một chiều (non-potential) |
| Đứng yên | Phạt va chạm/safety quá lớn so với phần thưởng đích |
| Lao thẳng vào chướng ngại | Step penalty lớn hoặc $R_c$ nhỏ |
| Dao động trái–phải | Thiếu smoothness, hoặc heading xung đột safety |
| Không bao giờ chạm đích | Sparse thuần, không có relabel hoặc exploration |

### Thiết kế tham chiếu (điểm xuất phát để ablation)

$$
r_t = \underbrace{R_g\,[d_t<\varepsilon]-R_c\,[\text{collision}]}_{\text{goal/terminal}}
+ \underbrace{\alpha\,(d_{t-1}-d_t)}_{\text{progress}}
- \underbrace{\lambda\,\max\!\Big(0,\tfrac{d_{safe}-d_{obs}}{d_{safe}}\Big)}_{\text{safety}}
- \underbrace{c}_{\text{efficiency}}
- \underbrace{\beta\,\lvert\Delta\omega\rvert}_{\text{smoothness}}
$$

Các hệ số $R_g, R_c, \alpha, \lambda, c, \beta$ là siêu tham số: cần báo cáo giá trị và phân tích độ nhạy.

**Gợi ý ablation:** bỏ từng thành phần; so sánh sparse+HER với dense; thử dạng potential-based với $\gamma<1$.

---

## 7. Nguồn và trạng thái xác minh

| Nhóm | Nguồn | Link | Trạng thái |
|---|---|---|---|
| Shaping, invariance | Ng, Harada, Russell (1999), *Policy invariance under reward transformations: Theory and application to reward shaping*, ICML | https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf | ✅ Đã đọc toàn văn |
| Intrinsic, pseudo-count | Bellemare et al. (2016), *Unifying Count-Based Exploration and Intrinsic Motivation*, NeurIPS | https://arxiv.org/abs/1606.01868 | ✅ Đã đọc toàn văn |
| Sparse + relabel | Andrychowicz et al. (2017), *Hindsight Experience Replay*, NeurIPS | https://arxiv.org/abs/1707.01495 | ✅ Đã đọc toàn văn |
| Reward navigation (3 tình huống, velocity-range) | Lee & Jeong (2023), Front. Neurorobot., DOI 10.3389/fnbot.2023.1210442 | https://pubmed.ncbi.nlm.nih.gov/37744086/ | 🔎 Đã thấy abstract/snippet |
| Khảo sát hàm reward path following | SBR/LARS 2025 | https://sol.sbc.org.br/index.php/sbrlars/article/view/39227 | 🔎 Đã thấy abstract |
| Review DRL navigation, môi trường động | Sensors 2025, DOI 10.3390/s25113394 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12158180/ | 🔎 Đã thấy abstract |
| Review DRL path planning | Singh, Ren, Lin (2023), DOI 10.3390/vehicles5040078 | https://doaj.org/article/2773335a5438497c8c0a94682033a954 | 🔎 Đã thấy abstract |
| Review môi trường đông người | Le, Saeedvand, Hsu (2024), J. Intell. Robot. Syst. 110(4) | https://scholar.lib.ntnu.edu.tw/en/publications/a-comprehensive-review-of-mobile-robot-navigation-using-deep-rein/ | 🔎 Đã thấy abstract |
| Social navigation | Kabir & Mysorewala, arXiv 2512.00049 | https://arxiv.org/abs/2512.00049v1 | 🔎 Đã thấy abstract |
| ICM, RND, GAIL, IRL (Abbeel & Ng 2004), CPO (Achiam 2017), CADRL (Chen 2017), dynamic potential (Devlin & Kudenko 2012), preference learning (Christiano 2017) | Từ kiến thức nền | (chưa có) | ⚠️ Chưa kiểm tra link, cần xác minh trước khi trích |

**Chú giải:** ✅ đã mở và đối chiếu toàn văn; 🔎 mới đọc abstract/snippet, cần mở bản đầy đủ; ⚠️ chưa xác minh.

---

## 8. Ghi chú sửa lỗi so với bản tóm tắt gốc

- Bài Ng et al. có tiêu đề "...application to **reward shaping**", không phải "...Success Reward".
- Kết quả chính của Ng et al. là **Định lý 1** (potential-based là điều kiện cần và đủ); dạng chuẩn $F=\gamma\Phi(s')-\Phi(s)$, cần đối chiếu trực tiếp trên PDF.
- Pseudo-count: $\rho'_n(x)$ là *recoding probability* (xác suất gán cho $x$ sau khi quan sát thêm một lần $x$). Bài chỉ thử trên Atari và thừa nhận chưa rõ với không gian liên tục.
- HER: yêu cầu thuật toán off-policy và chính sách goal-conditioned. Kết luận "shaping kém hơn sparse" chỉ đến từ thí nghiệm manipulation trong bài, không nên khái quát cho navigation.
