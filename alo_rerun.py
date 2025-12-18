import rerun as rr
import numpy as np

# 1. Khởi động Rerun
rr.init("Embedding_Space_3D", spawn=True)

# 2. Tạo dữ liệu giả (Thay bằng vector t-SNE của bạn)
num_points = 5000
positions = np.random.normal(size=(num_points, 3)) # Vector 3D
colors = np.random.uniform(0, 255, size=(num_points, 3)) # Màu ngẫu nhiên
labels = [f"Doc {i}" for i in range(num_points)]

# 3. Log dữ liệu lên Rerun (Nó sẽ bật app riêng lên)
rr.log(
    "embeddings", 
    rr.Points3D(positions, colors=colors, radii=0.05, labels=labels)
)