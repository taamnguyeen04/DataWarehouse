import time
# 👇 QUAN TRỌNG: Phải import thêm 'Collection' ở đây
from pymilvus import connections, utility, Collection

# 1. Kết nối
try:
    connections.connect(
        alias="default",
        host="0.tcp.ap.ngrok.io",
        port="11421",
        user="root",
        password="aiostorm"
    )
    print("✅ Đã kết nối thành công!\n")
except Exception as e:
    print(f"❌ Kết nối thất bại: {e}")
    exit()

# Danh sách các collection cần load
target_collections = [
    'arch_beit3_image_v100', 
    'arch_clip_image_v100', 
    'arch_object_name_v100'
]

print(f"Danh sách collection hiện có: {utility.list_collections()}")
print("--- Bắt đầu đo thời gian Load ---")

for name in target_collections:
    if not utility.has_collection(name):
        print(f"⚠️ Collection '{name}' không tồn tại. Bỏ qua.")
        continue

    print(f"🔄 Đang load '{name}'...", end=" ", flush=True)
    
    # Khởi tạo object Collection
    coll = Collection(name)
    
    # Trước khi load, gọi release để đảm bảo đo thời gian load "nguội" (từ ổ cứng lên RAM)
    # Nếu không release, nếu data đã ở trên RAM rồi thì thời gian sẽ = 0s
    coll.release() 
    
    start_time = time.time()
    
    try:
        # Load collection vào bộ nhớ
        coll.load()
        
        # Chờ cho đến khi load xong hoàn toàn
        utility.wait_for_loading_complete(name)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ Xong! Thời gian: {elapsed:.4f} giây")
        
        # (Tuỳ chọn) In ra số lượng vector để tham khảo
        print(f"   -> Số lượng entities: {coll.num_entities}")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi load '{name}': {e}")

print("\n--- Hoàn tất ---")
# col = Collection("pmc_papers_v2")
# print("Description:", col.describe())
# print("Number of entities:", col.num_entities)
# print("Index:", col.indexes)
# # # import json
# print(utility.loading_progress("pmc_papers_v2"))

# # file_path = "data.jsonl"
# # target_line = 1  # dòng bạn muốn lấy (bắt đầu từ 1)

# # with open(file_path, "r", encoding="utf-8") as f:
# #     for i, line in enumerate(f, start=1):
# #         if i == target_line:
# #             data = json.loads(line)
# #             print(data)
# #             break


# from elasticsearch import Elasticsearch
# from elasticsearch.exceptions import ConnectionError, AuthenticationException

# # Cấu hình
# ES_HOST = "100.98.10.24"
# ES_PORT = 9200
# ES_USERNAME = "elastic"
# ES_PASSWORD = "aiostorm"
# ES_USE_SSL = False

# # Tạo URL kết nối
# protocol = "https" if ES_USE_SSL else "http"
# url = f"{protocol}://{ES_USERNAME}:{ES_PASSWORD}@{ES_HOST}:{ES_PORT}"

# # Tạo client Elasticsearch
# es = Elasticsearch(url, verify_certs=ES_USE_SSL)

# # Kiểm tra kết nối
# try:
#     info = es.info()
#     print("✅ Kết nối Elasticsearch thành công!")
#     print("Cluster name:", info.get("cluster_name"))
#     print("Elasticsearch version:", info.get("version", {}).get("number"))
# except AuthenticationException:
#     print("❌ Sai username hoặc password.")
# except ConnectionError:
#     print("❌ Không thể kết nối tới Elasticsearch (kiểm tra host/port).")
# except Exception as e:
#     print("❌ Lỗi khác:", str(e))

# import os
# import subprocess
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import multiprocessing
# import sys
# sys.stdout.reconfigure(encoding='utf-8')
# sys.stderr.reconfigure(encoding='utf-8')
# BASE_DIR = "C:/Users/tam/Desktop/Data/preprocessed/corpus_chunks"
# VOLUME_NAME = "par"   # tên volume trên Modal
# REMOTE_BASE_PATH = "/mnt/par/Data Warehouse/ReCDS_benchmark/PAR/preprocessed/corpus_chunks"
# START = 106                   # file bắt đầu (ví dụ: 0 cho corpus_chunk_0000.pt)
# END = 106                   # file kết thúc (ví dụ: 117 cho corpus_chunk_0117.pt)

# # Tính số core và gợi ý số luồng
# cpu_cores = multiprocessing.cpu_count()
# MAX_WORKERS = min(cpu_cores * 2, 8)  # tối đa 8 luồng cho ổn định

# def upload_file(file_index, retries=3):
#     """Upload một file .pt lên Modal storage"""
#     filename = f"corpus_chunk_{file_index:04d}.pt"
#     local_path = os.path.join(BASE_DIR, filename)

#     # Kiểm tra file có tồn tại không
#     if not os.path.exists(local_path):
#         return f"SKIPPED {filename}: file not found"

#     # Remote path trên Modal volume
#     remote_path = f"{REMOTE_BASE_PATH}/{filename}"
#     cmd = ["modal", "volume", "put", "-f", VOLUME_NAME, local_path, remote_path]

#     # Set environment variables để Modal CLI không gặp lỗi encoding
#     env = os.environ.copy()
#     env['PYTHONIOENCODING'] = 'utf-8'
#     env['PYTHONUTF8'] = '1'

#     for attempt in range(1, retries+1):
#         try:
#             # Chạy với env modified và suppress output
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 env=env,
#                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
#             )

#             if result.returncode == 0:
#                 file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
#                 return f"OK Uploaded {filename} ({file_size_mb:.2f} MB)"
#             else:
#                 if attempt < retries:
#                     print(f"Retry {filename} (lan {attempt})...")
#                 else:
#                     # Decode với error handling
#                     try:
#                         error_msg = result.stderr.decode('utf-8', errors='replace').strip()
#                         # Loại bỏ các box drawing characters
#                         error_msg = error_msg.replace('+-', '').replace('-+', '').replace('|', '').strip()
#                     except:
#                         error_msg = "Unknown error"
#                     return f"FAILED {filename}: {error_msg}"
#         except Exception as e:
#             if attempt < retries:
#                 print(f"Retry {filename} (lan {attempt}) - Exception: {str(e)}")
#             else:
#                 return f"FAILED {filename}: Exception - {str(e)}"
#     return None


# def main():
#     # Tạo danh sách các file index cần upload
#     file_indices = list(range(START, END + 1))

#     print(f"Upload corpus_chunk_{START:04d}.pt den corpus_chunk_{END:04d}.pt")
#     print(f"Local: {BASE_DIR}")
#     print(f"Remote: {REMOTE_BASE_PATH}")
#     print(f"Tong {len(file_indices)} files")
#     print(f"Dung {MAX_WORKERS} luong song song\n")

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(upload_file, idx): idx for idx in file_indices}
#         for future in as_completed(futures):
#             result = future.result()
#             if result:
#                 print(result)

# if __name__ == "__main__":
#     main()


# from pymilvus import connections, utility

# COLLECTION_NAME = "pmc_papers_v1"
# MILVUS_HOST = "127.0.0.1"
# MILVUS_PORT = "19530"

# # Kết nối Milvus
# connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# # Kiểm tra collection tồn tại
# if utility.has_collection(COLLECTION_NAME):
#     utility.drop_collection(COLLECTION_NAME)
#     print(f"Đã xoá toàn bộ collection: {COLLECTION_NAME}")
# else:
#     print("Collection không tồn tại!")

