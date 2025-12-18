import os
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa file gốc (giống đường dẫn trong code cũ của bạn)
SOURCE_DIR = "C:/Users/tam/Desktop/Data/par/results (15)/embeddings_output"

# Số dòng tối đa trong mỗi file con
# Với 15 workers, nếu file gốc có 1 triệu dòng, 
# để 20000 dòng/file sẽ tạo ra khoảng 50 file -> Tận dụng tốt đa luồng
LINES_PER_FILE = 20000 

def split_jsonl_file():
    source_path = Path(SOURCE_DIR)
    
    # 1. Tìm file .jsonl trong thư mục
    files = list(source_path.glob("*.jsonl"))
    if not files:
        print("Không tìm thấy file .jsonl nào trong thư mục!")
        return
    
    # Lấy file đầu tiên tìm thấy (giả sử chỉ có 1 file lớn)
    large_file = files[0]
    print(f"Đang xử lý file: {large_file.name}")

    # 2. Tạo thư mục output riêng để không bị lẫn với file gốc
    output_dir = source_path / "split_files"
    output_dir.mkdir(exist_ok=True)
    print(f"File nhỏ sẽ được lưu tại: {output_dir}")

    # 3. Bắt đầu tách file
    file_count = 1
    line_count = 0
    current_out_file = None
    
    # Đếm tổng số dòng trước để hiển thị thanh tiến trình (tùy chọn, có thể bỏ qua bước này nếu file quá lớn)
    print("Đang đếm tổng số dòng...")
    total_lines = sum(1 for _ in open(large_file, 'r', encoding='utf-8'))
    
    try:
        with open(large_file, 'r', encoding='utf-8') as f_in:
            pbar = tqdm(total=total_lines, desc="Splitting")
            
            for line in f_in:
                # Nếu chưa mở file hoặc đã đủ số lượng dòng -> mở file mới
                if current_out_file is None:
                    new_file_name = f"{large_file.stem}_part_{file_count:03d}.jsonl"
                    new_file_path = output_dir / new_file_name
                    current_out_file = open(new_file_path, 'w', encoding='utf-8')
                
                # Ghi dòng vào file con
                current_out_file.write(line)
                line_count += 1
                pbar.update(1)
                
                # Nếu đủ dòng quy định -> đóng file hiện tại
                if line_count >= LINES_PER_FILE:
                    current_out_file.close()
                    current_out_file = None
                    line_count = 0
                    file_count += 1
            
            # Đóng file cuối cùng nếu còn mở
            if current_out_file:
                current_out_file.close()
                
        print("\n" + "="*30)
        print("HOÀN THÀNH!")
        print(f"Đã tách thành {file_count} file nhỏ trong thư mục: {output_dir}")
        print("="*30)

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    split_jsonl_file()