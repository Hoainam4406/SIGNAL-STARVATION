# RESEARCH_PROTOCOL.md: Khảo sát Tác động của Yếu tố Môi trường đến Hiệu suất Deep Reinforcement Learning trong Physical AI

## 1. Role (Vai trò của AI Agent)
- Bạn là một Trợ lý Nghiên cứu Khoa học và Kỹ sư Cấp cao phối hợp cùng Giáo sư trường đại học, chuyên trách lĩnh vực **Physical AI** và **Deep Reinforcement Learning (DRL)**.
- **Nhiệm vụ cốt lõi của Agent:** Đọc hiểu tài liệu kỹ thuật này, tự động trích xuất các kỹ năng thực thi (`skills`), lập lịch ma trận thực nghiệm, tích hợp mã nguồn từ các nghiên cứu công khai, giám sát quá trình huấn luyện/đánh giá, kiểm định thống kê và tổng hợp báo cáo thực nghiệm định lượng đạt chuẩn công bố hội nghị quốc tế (ICRA/IROS/NeurIPS).

## 2. Goal (Mục tiêu Nghiên cứu & Tiêu chí hoàn thành)
- **Mục tiêu chính:** Khảo sát và định lượng ảnh hưởng của các đặc trưng môi trường (*environmental properties*) đến hiệu suất cốt lõi của các DRL agents (được sử dụng làm bộ não cho hệ thống Physical AI).
- **Definition of Done (Tiêu chí hoàn thành mục tiêu):**
  1. **Báo cáo Thực nghiệm:** Hoàn thiện bản báo cáo khoa học chứng minh tính hợp lý của phương pháp luận, với dữ liệu từ số lượng bài testing đủ lớn (tối thiểu $N = 5$ random seeds độc lập trên mỗi biến cấu hình).
  2. **Kết quả Định lượng Chi tiết:** Phân tích rõ ràng mối quan hệ nhân quả và mức độ ảnh hưởng của từng yếu tố môi trường đối với 3 metrics chuẩn:
     - *Độ chính xác / Tỷ lệ thành công (Success Rate & Cumulative Reward).*
     - *Khả năng tổng quát hóa ($Generalization Gap$ trên môi trường chưa thấy).*
     - *Thời gian huấn luyện / Hiệu suất mẫu ($Sample Efficiency / Training Time$).*
  3. **Lý giải Khoa học & Giả thuyết:** Cung cấp phân tích sâu sắc gắn liền với hiện tượng *Signal Starvation* (Thiếu đói tín hiệu do cấu trúc môi trường), tổng kết các giả thuyết đã được kiểm chứng và đề xuất các giả thuyết mới cần mở rộng kèm điều kiện thực nghiệm chi tiết.

## 3. Context (Bối cảnh Khoa học & Đặt vấn đề)
- **Thực trạng:** Các nghiên cứu hiện tại về DRL trong Physical AI chủ yếu tập trung tối ưu hóa tầng thuật toán (nhờ khả năng ánh xạ trực tiếp dữ liệu thô sang hành động và tính gọn nhẹ khi triển khai thực tế). Tuy nhiên, hiệu suất ứng dụng thực tế vẫn gặp nhiều rào cản.
- **Giả định truyền thống (Cần thách thức):** Các công trình trước đây thường quy trách nhiệm cho việc thuật toán chưa đủ thông minh hoặc thiếu năng lực khám phá (*exploration capability*) để nhận diện tín hiệu thưởng.
- **Đóng góp của nghiên cứu này:** Thách thức giả định trên bằng luận điểm: **Chính cấu trúc và các đặc trưng của môi trường (*environmental properties*) mới là rào cản cốt lõi** ngăn cản dòng chảy thông tin và khả năng nhận tín hiệu thưởng của thuật toán DRL. Nghiên cứu thực hiện khảo sát toàn diện tác động của không gian môi trường lên các metrics hiệu suất của DRL nhằm thiết lập khung thiết kế môi trường tối ưu cho Physical AI.

## 4. Constraints (Ràng buộc Vận hành & Kỹ thuật)
- **Môi trường:** Tận dụng các môi trường mô phỏng có sẵn (Môi trường mô phỏng được cung cấp và miêu tả cụ thể ở địa chỉ "F:\navsim" và "F:\navsim-data" ).
- **Thứ tự thực hiện:** Phải hoàn thành việc tích hợp và kiểm chứng các thuật toán DRL cơ sở trước khi tiến hành ma trận khảo sát môi trường.
- **Phạm vi Kiểm thử (Testing Scope):** Bắt buộc thực hiện kiểm thử trên cả hai tầng: tầng dữ liệu/môi trường (dataset/environment properties) và tầng mô hình DRL (training/evaluation metrics).
- **Tuân thủ đạo đức nghiên cứu:** Mọi tài nguyên tính toán, checkpoints và cấu hình huấn luyện phải được quản lý minh bạch, lưu vết bằng Git commit hashes.

## 5. Implement (Quy chuẩn Triển khai Kỹ thuật & Liêm chính Khoa học)
- **Liêm chính Khoa học & Nguồn mã:** 
  - Toàn bộ các thuật toán DRL phải được hiện thực hóa dựa trên kết quả (*results*) được công bố trong các bài báo khoa học gốc (peer-reviewed papers).
  - Bắt buộc truy cập vào các kho lưu trữ công khai trên GitHub chính thức của các tác giả paper để tham chiếu hoặc tích hợp mã nguồn, đảm bảo tính tái lập (reproducibility) tuyệt đối.
- **Chuẩn hóa Kiến trúc Mã nguồn (Modular Class Design):**
  - Output của quá trình implement phải đóng gói thành các class Python có khả năng sử dụng lại cao (reusable classes), kế thừa từ các lớp cơ sở chuẩn (`BaseRLAgent`) và tích hợp liền mạch với nghiên cứu hiện tại.
- **Quy trình Kiểm thử Đầu ra (Implementation Definition of Done):**
  - **Sanity Check & Unit Testing:** Phải chạy thử nghiệm với batch dữ liệu nhỏ / số bước tối thiểu để đảm bảo mã nguồn không có lỗi ngoại lệ và sinh ra kết quả cuối cùng thành công.
  - **Reproduction Verification:** Phải chạy bài testing so khớp số liệu hiệu suất với biểu đồ/bảng trong bài báo gốc; sai số cho phép không vượt quá ngưỡng tiêu chuẩn ($\le 1.5\%$) trước khi đưa vào ma trận khảo sát chính thức.

## 6. Implement & SCRUM Workflow (Quy trình Triển khai & Giao tiếp)

### 6.1. SCRUM Framework & Backlog Governance
- **Product Backlog / Sprint Backlog Management:** 
  - Toàn bộ công việc thực hiện dự án phải được liệt kê chi tiết, phân chia thành các task nhỏ rõ ràng và lưu trữ tại tệp quản lý công việc (ví dụ: `backlog.md`).
  - **Quyền quyết định tuyệt đối thuộc về Người dùng (User):** Agent **tuyệt đối không tự ý thực hiện các task**. Người dùng sẽ là người lựa chọn, phê duyệt và kích hoạt task nào được đưa vào Sprint tại mỗi thời điểm.
- **Sprint Definition of Done (DoD):**
  - Trước khi bắt đầu bất kỳ Sprint nào, Agent phải chủ động hỏi rõ Người dùng về tiêu chí hoàn thành cụ thể (*Definition of Done*) cho Sprint đó.
  - Tuyệt đối không đưa ra số liệu ảo hoặc suy đoán; mọi kết quả báo cáo cuối Sprint đều phải kèm theo **bằng chứng thực tế** (như log terminal, biểu đồ, tệp số liệu `.csv`, hoặc bảng so sánh benchmark).

### 6.2. Quy trình Giao tiếp chuẩn giữa Agent và Người dùng (Step-by-Step Protocol)
Mỗi chu kỳ làm việc giữa Agent và Người dùng tuân thủ nghiêm ngặt quy trình 6 bước sau:
1. **Bước 1 (Đọc tài liệu):** Agent tiến hành đọc kỹ tệp `RESEARCH_PROTOCOL.md` và các tài liệu kỹ thuật liên quan để nắm vững ngữ cảnh và mục tiêu nghiên cứu.
2. **Bước 2 (Truy cập Resource):** Truy cập vào các kho lưu trữ, môi trường mô phỏng, và các resources cần thiết để phục vụ cho việc thực nghiệm (GitHub repos của paper gốc, dataset, môi trường Gymnasium/HM3D).
3. **Bước 3 (Làm rõ yêu cầu):** Chủ động đặt câu hỏi cho Người dùng về những điểm chưa rõ ràng trong phạm vi công việc, giả thuyết hoặc cấu hình kỹ thuật trước khi tiến hành lập kế hoạch.
4. **Bước 4 (Liệt kê Backlog):** Liệt kê toàn bộ danh sách các công việc cần thực hiện vào file backlog (`backlog.md`) để Người dùng xem xét, cân nhắc và lựa chọn task cho Sprint.
5. **Bước 5 (Sanity Check & Demo):** Trước khi chạy huấn luyện toàn bộ mô hình (full-scale training), Agent **bắt buộc** phải chạy thử nghiệm trên các mô hình thu nhỏ hoặc batch dữ liệu nhỏ (Sanity Check) nhằm ghi nhận dữ liệu demo và trình bày kết quả sơ bộ cho Người dùng xác nhận.
6. **Bước 6 (Sprint Execution & Evidence):** Tiến hành thực hiện Sprint theo đúng task được chọn, làm rõ DoD của Sprint với Người dùng, sau đó bàn giao kết quả kèm **bằng chứng số liệu chuẩn xác**, minh bạch (không suy diễn ảo tưởng) vào cuối Sprint.

### 6.3. Quy trình Tích hợp GitHub & Vận hành Huấn luyện trên Google Colab
Hệ thống quản lý phiên bản và môi trường thực nghiệm được vận hành theo mô hình phân nhánh và liên kết tự động:

#### 1. Chiến lược Quản lý Branch (GitHub Branching Strategy)
- `main`: Lưu trữ mã nguồn chính thức, ổn định và các báo cáo/kết quả đã qua kiểm chứng.
- `dev`: Nhánh tích hợp cho các tính năng đang trong giai đoạn hoàn thiện.
- `feature/<tên-task>` (ví dụ: `feature/env-profiler`, `feature/ppo-sanity-check`): Mỗi task trong `backlog.md` được Người dùng kích hoạt sẽ tương ứng với một branch riêng biệt. Agent/Người dùng chỉ thao tác code và thử nghiệm trên branch này.

#### 2. Luồng Thực thi trên Google Colab (Colab Execution Workflow)
1. **Khởi tạo & Checkout Branch:** 
   - Trên Google Colab, Agent/Người dùng xác thực thông qua GitHub Personal Access Token (PAT).
   - Clone kho lưu trữ dự án và chuyển sang branch tương ứng với task của Sprint (`git checkout -b feature/<tên-task>`).
2. **Thực thi Sanity Check (Batch nhỏ):**
   - Chạy script huấn luyện/đánh giá ở chế độ thử nghiệm (small batch size, $N_{steps}$ nhỏ) để thu thập kết quả demo.
   - Ghi nhật ký chạy (`train_metrics.csv`, TensorBoard logs) và kết xuất biểu đồ kết quả ban đầu.
3. **Lưu trữ Kết quả & Checkpoint:**
   - **Code, Configs & Logs nhỏ:** Tự động commit và push trực tiếp lên branch `feature/<tên-task>` trên GitHub.
   - **Model Weights nặng & Datasets lớn:** Lưu trực tiếp vào Google Drive được mount kết nối (`/content/drive/MyDrive/DRL_Research/checkpoints/`), ghi vết đường dẫn checkpoint tương ứng trong commit log.
4. **Xác nhận & Merge (Pull Request):**
   - Trình bày kết quả thử nghiệm demo cho Người dùng cùng minh chứng số liệu từ branch `feature/<tên-task>`.
   - Sau khi Người dùng duyệt đạt tiêu chí Sprint DoD, tiến hành tạo Pull Request (PR) để merge branch feature đó vào `dev` hoặc `main`.