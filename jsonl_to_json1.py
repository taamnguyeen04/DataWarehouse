import os
import json
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Cấu hình
INPUT_DIR = r"C:\Users\tam\Documents\data\Data Warehouse\embed\results (12)\embeddings_output"
OUTPUT_MERGED_DIR = r"C:\Users\tam\Documents\data\Data Warehouse\embed\results (12)\merged_output"
MAX_WORKERS = 16  # Số luồng xử lý song song
FILES_PER_MERGE = 4  # Số file JSON ghép thành 1

# Lock để in log an toàn
print_lock = Lock()


def safe_print(message):
    """In log thread-safe"""
    with print_lock:
        print(message)


def process_jsonl_file(jsonl_path):
    """
    Xử lý một file JSONL: đọc, chuyển đổi sang JSON, lưu và xóa file gốc

    Args:
        jsonl_path: Đường dẫn đến file JSONL

    Returns:
        tuple: (success, file_name, json_path, message)
    """
    try:
        jsonl_path = Path(jsonl_path)
        file_name = jsonl_path.name

        safe_print(f"[START] Đang xử lý: {file_name}")

        # Đọc file JSONL
        data = []
        line_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                        line_count += 1
                    except json.JSONDecodeError as e:
                        safe_print(f"[WARNING] Lỗi parse JSON tại dòng {line_count + 1} trong {file_name}: {e}")

        if not data:
            safe_print(f"[WARNING] File {file_name} không có dữ liệu hợp lệ")
            return (False, file_name, None, "No valid data")

        # Tạo tên file JSON output
        json_path = jsonl_path.with_suffix('.json')

        # Ghi file JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Kiểm tra kích thước file output
        file_size_mb = os.path.getsize(json_path) / (1024 * 1024)

        if file_size_mb > 1000:
            safe_print(f"[WARNING] File {json_path.name} có kích thước {file_size_mb:.2f} MB (> 1GB)")

        # Xóa file JSONL gốc
        os.remove(jsonl_path)

        safe_print(
            f"[SUCCESS] Hoàn thành: {file_name} -> {json_path.name} ({line_count} records, {file_size_mb:.2f} MB)")

        return (True, file_name, str(json_path), f"{line_count} records, {file_size_mb:.2f} MB")

    except Exception as e:
        safe_print(f"[ERROR] Lỗi khi xử lý {jsonl_path}: {str(e)}")
        return (False, str(jsonl_path), None, str(e))


def merge_json_files(json_files, output_path, batch_index):
    """
    Ghép nhiều file JSON thành một file

    Args:
        json_files: List các đường dẫn file JSON cần ghép
        output_path: Đường dẫn file output
        batch_index: Chỉ số batch để đặt tên

    Returns:
        tuple: (success, output_file, message)
    """
    try:
        safe_print(f"\n[MERGE START] Ghép {len(json_files)} files vào batch_{batch_index}.json")

        merged_data = []
        total_records = 0

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Xử lý cả trường hợp data là list hoặc dict
                    if isinstance(data, list):
                        merged_data.extend(data)
                        total_records += len(data)
                    else:
                        merged_data.append(data)
                        total_records += 1

                safe_print(
                    f"  [+] Đã đọc: {Path(json_file).name} ({len(data) if isinstance(data, list) else 1} records)")

            except Exception as e:
                safe_print(f"  [ERROR] Không đọc được {json_file}: {str(e)}")
                continue

        if not merged_data:
            return (False, output_path, "No data to merge")

        # Tạo thư mục output nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ghi file merged
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        safe_print(f"[MERGE SUCCESS] Đã tạo: {Path(output_path).name} ({total_records} records, {file_size_mb:.2f} MB)")

        # Xóa các file JSON gốc sau khi ghép thành công
        for json_file in json_files:
            try:
                os.remove(json_file)
                safe_print(f"  [-] Đã xóa: {Path(json_file).name}")
            except Exception as e:
                safe_print(f"  [WARNING] Không xóa được {json_file}: {str(e)}")

        return (True, output_path, f"{total_records} records, {file_size_mb:.2f} MB")

    except Exception as e:
        safe_print(f"[MERGE ERROR] Lỗi khi ghép files: {str(e)}")
        return (False, output_path, str(e))


def convert_and_merge_jsonl_files():
    """
    Tìm và chuyển đổi tất cả file JSONL, sau đó ghép thành các batch
    """
    # BƯỚC 1: Chuyển đổi JSONL sang JSON
    print("=" * 80)
    print("BƯỚC 1: CHUYỂN ĐỔI JSONL SANG JSON")
    print("=" * 80)

    jsonl_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))

    if not jsonl_files:
        print(f"Không tìm thấy file JSONL nào trong: {INPUT_DIR}")
        return

    print(f"Tìm thấy {len(jsonl_files)} file JSONL")
    print(f"Số luồng xử lý: {MAX_WORKERS}")
    print("=" * 80)

    # Xử lý song song chuyển đổi
    conversion_results = {
        'success': [],
        'failed': []
    }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_jsonl_file, file_path): file_path
            for file_path in jsonl_files
        }

        for future in as_completed(future_to_file):
            success, file_name, json_path, message = future.result()

            if success and json_path:
                conversion_results['success'].append(json_path)
            else:
                conversion_results['failed'].append((file_name, message))

    # In báo cáo chuyển đổi
    print("\n" + "=" * 80)
    print("KẾT QUẢ CHUYỂN ĐỔI:")
    print(f"Thành công: {len(conversion_results['success'])}/{len(jsonl_files)}")
    print(f"Thất bại: {len(conversion_results['failed'])}/{len(jsonl_files)}")

    if conversion_results['failed']:
        print("\nCác file thất bại:")
        for file_name, error in conversion_results['failed']:
            print(f"  - {file_name}: {error}")

    # BƯỚC 2: Ghép các file JSON
    if not conversion_results['success']:
        print("\nKhông có file JSON nào để ghép!")
        return

    print("\n" + "=" * 80)
    print("BƯỚC 2: GHÉP CÁC FILE JSON")
    print("=" * 80)
    print(f"Số file JSON cần ghép: {len(conversion_results['success'])}")
    print(f"Ghép {FILES_PER_MERGE} files thành 1 batch")
    print("=" * 80)

    # Chia thành các batch
    json_files = sorted(conversion_results['success'])
    batches = [json_files[i:i + FILES_PER_MERGE] for i in range(0, len(json_files), FILES_PER_MERGE)]

    print(f"Tổng số batch: {len(batches)}")

    # Tạo thư mục output
    os.makedirs(OUTPUT_MERGED_DIR, exist_ok=True)

    # Ghép từng batch
    merge_results = {
        'success': [],
        'failed': []
    }

    for batch_idx, batch_files in enumerate(batches, start=1):
        output_file = os.path.join(OUTPUT_MERGED_DIR, f"merged_batch_{batch_idx:03d}.json")

        success, output_path, message = merge_json_files(batch_files, output_file, batch_idx)

        if success:
            merge_results['success'].append((output_path, message))
        else:
            merge_results['failed'].append((output_path, message))

    # In báo cáo tổng kết
    print("\n" + "=" * 80)
    print("TỔNG KẾT CUỐI CÙNG:")
    print("=" * 80)
    print(f"Files JSONL chuyển đổi: {len(conversion_results['success'])}/{len(jsonl_files)}")
    print(f"Batches merged: {len(merge_results['success'])}/{len(batches)}")

    if merge_results['failed']:
        print("\nCác batch thất bại:")
        for file_name, error in merge_results['failed']:
            print(f"  - {file_name}: {error}")

    if merge_results['success']:
        print("\nCác file merged đã tạo:")
        for file_path, info in merge_results['success']:
            print(f"  ✓ {Path(file_path).name}: {info}")


if __name__ == "__main__":
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(INPUT_DIR):
        print(f"Thư mục không tồn tại: {INPUT_DIR}")
    else:
        convert_and_merge_jsonl_files()
        print("\n🎉 Hoàn thành tất cả!")