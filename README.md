# Credit Scoring

Dự án xây dựng mô hình xếp hạng tín dụng khách hàng dựa trên dữ liệu lịch sử tín dụng, thông tin tài chính và thông tin cá nhân. Notebook chính sử dụng Python và scikit-learn để tiền xử lý dữ liệu, huấn luyện mô hình Random Forest, đánh giá kết quả và lưu mô hình.

## Bài toán

Mục tiêu của dự án là dự đoán `Hạng tín dụng của khách hàng` dựa trên các đặc trưng như:

- Tài khoản vãng lai
- Thời hạn vay
- Lịch sử tín dụng
- Mục đích vay
- Tài khoản tiết kiệm
- Số năm kinh nghiệm
- Giới tính
- Tuổi
- Tình trạng nhà ở
- Nghề nghiệp

## Quy trình thực hiện

1. Đọc dữ liệu từ `train.xlsx` và `test.xlsx`.
2. Khám phá kích thước, kiểu dữ liệu và một số dòng mẫu.
3. Mã hóa các biến phân loại bằng `LabelEncoder`.
4. Tách biến đầu vào `X` và biến mục tiêu `y`.
5. Huấn luyện mô hình `RandomForestClassifier`.
6. Đánh giá mô hình bằng Accuracy và F1-score.
7. Dự đoán hạng tín dụng cho khách hàng mới.
8. Lưu và tải lại mô hình bằng `joblib`.


## Yêu cầu môi trường

Nên chạy notebook bằng Jupyter Notebook, JupyterLab hoặc VS Code với Python 3.

Các thư viện cần có:

```bash
pip install pandas scikit-learn openpyxl joblib
```

## Cách chạy

1. Mở terminal tại thư mục `CreditScoring`.
2. Cài các thư viện cần thiết nếu chưa có.
3. Mở `CreditScoring.ipynb`.
4. Chạy lần lượt các cell từ trên xuống dưới.

Vì notebook đọc dữ liệu bằng đường dẫn tương đối:

```python
pd.read_excel("train.xlsx")
pd.read_excel("test.xlsx")
```

hãy đảm bảo kernel/notebook đang chạy trong đúng thư mục `CreditScoring`.
