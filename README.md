# YOLOv8 Web Demo

Ứng dụng web demo sử dụng **YOLOv8** để phát hiện đối tượng trong ảnh,
xây dựng bằng **Streamlit**.

## 🚀 Tính năng

-   Tải ảnh lên từ **Clipboard**, **URL**, hoặc **Upload file**.
-   Sử dụng mô hình YOLOv8 để phát hiện đối tượng.
-   Hiển thị ảnh gốc và ảnh đã detect.
-   Cho phép lọc kết quả detect theo **nhãn (label)** có trong ảnh.
-   Cắt và hiển thị các vùng ảnh chứa nhãn đã chọn.
-   Cho phép tải về ảnh gốc hoặc ảnh đã detect.

## 📂 Cấu trúc thư mục

    deploy-yolov8/
    │
    ├── app/
    │   └── streamlit_app.py      # File chính chạy ứng dụng Streamlit
    │
    ├── yolo/
    │   └── detector.py           # Lớp Detector xử lý YOLOv8
    │
    ├── utils/
    │   └── images_io.py          # Các hàm hỗ trợ xử lý ảnh
    │
    ├── configs/
    │   ├── app.yaml              # Cấu hình ứng dụng
    │   └── model.yaml            # Cấu hình mô hình YOLO
    │
    └── requirements.txt          # Danh sách thư viện cần thiết

## ⚙️ Cài đặt

### 1. Clone dự án

``` bash
git clone https://github.com/nguyenduyan28/YOLOv8-Detection.git
cd YOLOv8-Detection/
```

### 2. Tạo môi trường ảo và cài đặt thư viện

``` bash
conda create -n yolov8 python=3.10 -y
conda activate yolov8
pip install -r requirements.txt
```

### 3. Cấu hình

-   Chỉnh sửa file `configs/app.yaml` để thay đổi tham số ứng dụng.
-   Chỉnh sửa file `configs/model.yaml` để trỏ tới mô hình YOLO mong
    muốn (`yolov8n.pt` hoặc custom `.pt`).

### 4. Chạy ứng dụng

``` bash
streamlit run streamlit_app.py
```

## 🖼️ Sử dụng

1.  Bấm **Upload image** để chọn nguồn ảnh.
2.  Chọn cách upload: **Paste (Clipboard)**, **URL**, hoặc **Upload
    file**.
3.  Xem kết quả ở tab **Detected** hoặc **Original**.
4.  Chọn **label** từ danh sách để lọc kết quả và xem ảnh crop.
5.  Tải về ảnh bằng các nút **Download original** hoặc **Download
    detected**.

## 🛠️ Công nghệ sử dụng

-   [Streamlit](https://streamlit.io/) --- xây dựng giao diện web.
-   [YOLOv8](https://docs.ultralytics.com/) --- phát hiện đối tượng.
-   [Pillow](https://pillow.readthedocs.io/) --- xử lý ảnh.
-   [PyYAML](https://pyyaml.org/) --- đọc file cấu hình.


