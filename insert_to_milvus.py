import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from tqdm import tqdm

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Cấu hình logging với UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('milvus_insert.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cấu hình Milvus
COLLECTION_NAME = "pmc_papers_v2"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
BATCH_SIZE = 2000  # Số lượng records mỗi batch
MAX_WORKERS = 15    # Số thread song song
EMBEDDING_DIM = 768  # Thay đổi theo dimension của embedding của bạn

class MilvusInserter:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.collection = None
        self.existing_ids: Set[str] = set()
        
    def connect_milvus(self):
        """Kết nối đến Milvus"""
        try:
            connections.connect(alias="default", uri = "https://in05-d7d375279b93833.serverless.aws-eu-central-1.cloud.zilliz.com", token = "35d5a7bc1f263bd6cfdfad6f548fe4f598ec4776cfdb2dfcae4074b27d39bcdcd5938aedd965e895d40fa77b5fe80cc21f6e533f")
            # connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            logger.info("Connected to Milvus successfully")

            if not utility.has_collection(COLLECTION_NAME):
                logger.info(f"Collection {COLLECTION_NAME} does not exist. Creating new collection...")
                self.create_collection()

            # Không cần load collection nếu chỉ insert
            self.collection = Collection(COLLECTION_NAME)
            logger.info(f"Connected to collection: {COLLECTION_NAME}")

        except Exception as e:
            logger.error(f"Milvus connection error: {e}")
            raise

    
    def create_collection(self):
        """Tạo collection mới với schema phù hợp"""
        try:
            # Định nghĩa schema
            fields = [
                FieldSchema(name="pmid", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="PMC papers embeddings collection"
            )
            
            # Tạo collection
            self.collection = Collection(
                name=COLLECTION_NAME,
                schema=schema
            )
            
            # Tạo index cho vector field để tăng tốc độ tìm kiếm
            index_params = {
                "metric_type": "L2",
                "index_type": "DISKANN",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # Load collection vào memory
            self.collection.load()
            
            logger.info(f"Successfully created collection with schema: {schema}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def get_existing_ids(self) -> Set[str]:
        """Get list of existing IDs in collection"""
        try:
            logger.info("Getting existing IDs...")
            # Query all IDs - adjust field name according to your schema
            expr = "pmid != ''"  # Query all
            results = self.collection.query(
                expr=expr,
                output_fields=["pmid"],
                limit=16384  # Milvus limit
            )
            
            existing_ids = {str(r['pmid']) for r in results}
            logger.info(f"Found {len(existing_ids)} existing IDs")
            return existing_ids
            
        except Exception as e:
            logger.warning(f"Cannot get existing IDs: {e}")
            return set()
    
    def read_jsonl_file(self, file_path: Path) -> List[Dict]:
        """Read JSONL file and return list of records"""
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        if 'pmid' in record and 'embedding' in record:
                            records.append(record)
                        else:
                            logger.warning(f"File {file_path.name} line {line_num}: missing pmid or embedding field")
                    except json.JSONDecodeError as e:
                        logger.warning(f"File {file_path.name} line {line_num}: JSON parse error - {e}")
            
            logger.info(f"Read {len(records)} records from {file_path.name}")
            return records
            
        except Exception as e:
            logger.error(f"Error reading file {file_path.name}: {e}")
            return []
    
    def filter_duplicates(self, records: List[Dict]) -> List[Dict]:
        """Filter out records with existing IDs"""
        filtered = []
        for record in records:
            pmid = str(record['pmid'])
            if pmid not in self.existing_ids:
                filtered.append(record)
            else:
                logger.debug(f"Skip duplicate PMID: {pmid}")
        
        if len(records) > len(filtered):
            logger.info(f"Filtered {len(records) - len(filtered)} duplicate records")
        
        return filtered
    
    def insert_batch(self, records: List[Dict]) -> bool:
        """Insert a batch of records into Milvus"""
        if not records:
            return True
            
        try:
            # Prepare data according to Milvus format
            pmids = [str(r['pmid']) for r in records]
            embeddings = [r['embedding'] for r in records]
            
            # Insert into Milvus
            entities = [pmids, embeddings]
            self.collection.insert(entities)
            
            # Update existing_ids
            self.existing_ids.update(pmids)
            
            logger.info(f"Successfully inserted {len(records)} records")
            return True
            
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            return False
    
    def process_file(self, file_path: Path) -> bool:
        """Process one JSONL file"""
        try:
            logger.info(f"Processing file: {file_path.name}")
            
            # Read file
            records = self.read_jsonl_file(file_path)
            if not records:
                logger.warning(f"File {file_path.name} has no valid data")
                return False
            
            # Filter duplicates
            records = self.filter_duplicates(records)
            if not records:
                logger.info(f"File {file_path.name}: all records already exist, skipping")
                return True  # Still consider success and delete file
            
            # Insert in batches
            total_inserted = 0
            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i:i + BATCH_SIZE]
                if not self.insert_batch(batch):
                    logger.error(f"Batch insert error for file {file_path.name}")
                    return False
                total_inserted += len(batch)
            
            # Flush to ensure data is saved
            self.collection.flush()
            
            logger.info(f"Completed file {file_path.name}: inserted {total_inserted} records")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            return False
    
    def process_all_files(self):
        """Process all JSONL files in directory"""
        # Get file list
        files = list(self.data_dir.glob("*.jsonl"))
        if not files:
            logger.warning("No JSONL files found in directory")
            return
        
        logger.info(f"Found {len(files)} JSONL files")
        
        # Get existing IDs
        self.existing_ids = self.get_existing_ids()
        
        # Process in parallel with ThreadPoolExecutor
        success_count = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in files
            }
            
            # Process results with progress bar
            with tqdm(total=len(files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            # Delete file after successful insert
                            file_path.unlink()
                            logger.info(f"Deleted file: {file_path.name}")
                            success_count += 1
                        else:
                            failed_files.append(file_path.name)
                            logger.error(f"Keeping file due to error: {file_path.name}")
                    except Exception as e:
                        failed_files.append(file_path.name)
                        logger.error(f"Exception processing {file_path.name}: {e}")
                    
                    pbar.update(1)
        
        # Summary
        logger.info("=" * 50)
        logger.info(f"Processing completed!")
        logger.info(f"Success: {success_count}/{len(files)} files")
        logger.info(f"Failed: {len(failed_files)} files")
        if failed_files:
            logger.info(f"Failed files: {', '.join(failed_files)}")
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            
            # Tạo index cho vector field để tăng tốc độ tìm kiếm
            index_params = {
                "metric_type": "L2",
                "index_type": "DISKANN",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # Load collection vào memory
            self.collection.load()
            
            logger.info(f"Successfully created collection with schema: {schema}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def get_existing_ids(self) -> Set[str]:
        """Get list of existing IDs in collection"""
        try:
            logger.info("Getting existing IDs...")
            # Query all IDs - adjust field name according to your schema
            expr = "pmid != ''"  # Query all
            results = self.collection.query(
                expr=expr,
                output_fields=["pmid"],
                limit=16384  # Milvus limit
            )
            
            existing_ids = {str(r['pmid']) for r in results}
            logger.info(f"Found {len(existing_ids)} existing IDs")
            return existing_ids
            
        except Exception as e:
            logger.warning(f"Cannot get existing IDs: {e}")
            return set()
    
    def read_jsonl_file(self, file_path: Path) -> List[Dict]:
        """Read JSONL file and return list of records"""
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        if 'pmid' in record and 'embedding' in record:
                            records.append(record)
                        else:
                            logger.warning(f"File {file_path.name} line {line_num}: missing pmid or embedding field")
                    except json.JSONDecodeError as e:
                        logger.warning(f"File {file_path.name} line {line_num}: JSON parse error - {e}")
            
            logger.info(f"Read {len(records)} records from {file_path.name}")
            return records
            
        except Exception as e:
            logger.error(f"Error reading file {file_path.name}: {e}")
            return []
    
    def filter_duplicates(self, records: List[Dict]) -> List[Dict]:
        """Filter out records with existing IDs"""
        filtered = []
        for record in records:
            pmid = str(record['pmid'])
            if pmid not in self.existing_ids:
                filtered.append(record)
            else:
                logger.debug(f"Skip duplicate PMID: {pmid}")
        
        if len(records) > len(filtered):
            logger.info(f"Filtered {len(records) - len(filtered)} duplicate records")
        
        return filtered
    
    def insert_batch(self, records: List[Dict]) -> bool:
        """Insert a batch of records into Milvus"""
        if not records:
            return True
            
        try:
            # Prepare data according to Milvus format
            pmids = [str(r['pmid']) for r in records]
            embeddings = [r['embedding'] for r in records]
            
            # Insert into Milvus
            entities = [pmids, embeddings]
            self.collection.insert(entities)
            
            # Update existing_ids
            self.existing_ids.update(pmids)
            
            logger.info(f"Successfully inserted {len(records)} records")
            return True
            
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            return False
    
    def process_file(self, file_path: Path) -> bool:
        """Process one JSONL file"""
        try:
            logger.info(f"Processing file: {file_path.name}")
            
            # Read file
            records = self.read_jsonl_file(file_path)
            if not records:
                logger.warning(f"File {file_path.name} has no valid data")
                return False
            
            # Filter duplicates
            records = self.filter_duplicates(records)
            if not records:
                logger.info(f"File {file_path.name}: all records already exist, skipping")
                return True  # Still consider success and delete file
            
            # Insert in batches
            total_inserted = 0
            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i:i + BATCH_SIZE]
                if not self.insert_batch(batch):
                    logger.error(f"Batch insert error for file {file_path.name}")
                    return False
                total_inserted += len(batch)
            
            # Flush to ensure data is saved
            self.collection.flush()
            
            logger.info(f"Completed file {file_path.name}: inserted {total_inserted} records")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            return False
    
    def process_all_files(self):
        """Process all JSONL files in directory"""
        # Get file list
        files = list(self.data_dir.glob("*.jsonl"))
        if not files:
            logger.warning("No JSONL files found in directory")
            return
        
        logger.info(f"Found {len(files)} JSONL files")
        
        # Get existing IDs
        self.existing_ids = self.get_existing_ids()
        
        # Process in parallel with ThreadPoolExecutor
        success_count = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in files
            }
            
            # Process results with progress bar
            with tqdm(total=len(files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            # Delete file after successful insert
                            file_path.unlink()
                            logger.info(f"Deleted file: {file_path.name}")
                            success_count += 1
                        else:
                            failed_files.append(file_path.name)
                            logger.error(f"Keeping file due to error: {file_path.name}")
                    except Exception as e:
                        failed_files.append(file_path.name)
                        logger.error(f"Exception processing {file_path.name}: {e}")
                    
                    pbar.update(1)
        
        # Summary
        logger.info("=" * 50)
        logger.info(f"Processing completed!")
        logger.info(f"Success: {success_count}/{len(files)} files")
        logger.info(f"Failed: {len(failed_files)} files")
        if failed_files:
            logger.info(f"Failed files: {', '.join(failed_files)}")
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Insert embeddings into Milvus")
    parser.add_argument("--data_dir", type=str, default="embeddings_output", help="Directory containing JSONL files")
    parser.add_argument("--drop_collection", action="store_true", help="Drop existing collection before inserting")
    args = parser.parse_args()

    data_dir = "C:/Users/tam/Desktop/Data/par/results (15)/embeddings_output/split_files"
    
    inserter = MilvusInserter(data_dir)
    
    try:
        # Connect to Milvus
        inserter.connect_milvus()
        
        if args.drop_collection and utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
            logger.info(f"Dropped collection {COLLECTION_NAME}")
            # Re-create
            inserter.create_collection()
        
        # Process all files
        inserter.process_all_files()
        
    except Exception as e:
        logger.error(f"Execution error: {e}")
    finally:
        # Ensure disconnection
        inserter.disconnect()


if __name__ == "__main__":
    main()