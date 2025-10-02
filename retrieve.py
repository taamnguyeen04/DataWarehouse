import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pymilvus import connections, Collection
import numpy as np
from typing import List, Dict, Union
import warnings
warnings.filterwarnings("ignore")

from model import BiEncoder
from config import Config


class PaperRetriever:
    def __init__(self,
                 model_path: str = None,
                 use_pretrained: bool = False,
                 device: str = None):
        """
        Khởi tạo retriever cho việc tìm kiếm bài báo

        Args:
            model_path: Đường dẫn đến model đã fine-tune (checkpoint file)
            use_pretrained: Nếu True, sử dụng model gốc chưa fine-tune
            device: Device để chạy model ('cuda', 'cpu', hoặc None để tự động)
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # Load model
        self.model = BiEncoder(
            model_name=Config.MODEL_NAME,
            embedding_dim=Config.EMBEDDING_DIM,
            pooling=Config.POOLING
        ).to(self.device)

        # Load checkpoint nếu có
        if model_path and not use_pretrained:
            print(f"Loading model from checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # Xử lý trường hợp model được lưu bằng DataParallel
            state_dict = checkpoint['model_state_dict']

            # Nếu state_dict có prefix 'module.', loại bỏ nó
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        elif use_pretrained:
            print("Using pretrained model without fine-tuning")

        self.model.eval()

        # Connect to Milvus
        print(f"Connecting to Milvus at {Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
            user=Config.MILVUS_USER,
            password=Config.MILVUS_PASSWORD
        )

        # Load collection
        self.collection = Collection(Config.COLLECTION_NAME)
        self.collection.load()
        print(f"Loaded collection: {Config.COLLECTION_NAME}")
        print(f"Total entities in collection: {self.collection.num_entities}")

    def encode_patient_description(self, patient_text: str) -> np.ndarray:
        """
        Encode mô tả bệnh nhân thành embedding vector

        Args:
            patient_text: Mô tả bệnh của bệnh nhân

        Returns:
            Embedding vector (numpy array)
        """
        # Tokenize
        encoded = self.tokenizer(
            patient_text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Encode
        with torch.no_grad():
            embedding = self.model.encode_query(input_ids, attention_mask)

        # Convert to numpy
        embedding_np = embedding.cpu().numpy()[0]

        return embedding_np

    def search(self,
               patient_text: str,
               top_k: int = 10,
               metric_type: str = "COSINE") -> List[Dict]:
        """
        Tìm kiếm top-k bài báo liên quan đến mô tả bệnh nhân

        Args:
            patient_text: Mô tả bệnh của bệnh nhân
            top_k: Số lượng bài báo cần trả về
            metric_type: Loại metric để tính similarity ("IP" cho Inner Product, "L2" cho Euclidean)

        Returns:
            List các dict chứa pmid và score
        """
        # Encode patient description
        query_embedding = self.encode_patient_description(patient_text)

        # Search trong Milvus
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["pmid"]
        )

        # Parse results
        retrieved_papers = []
        for hits in results:
            for hit in hits:
                retrieved_papers.append({
                    "pmid": hit.entity.get('pmid'),
                    "score": float(hit.score),
                    "distance": float(hit.distance)
                })

        return retrieved_papers

    def batch_search(self,
                     patient_texts: List[str],
                     top_k: int = 10,
                     metric_type: str = "COSINE") -> List[List[Dict]]:
        """
        Tìm kiếm batch cho nhiều mô tả bệnh nhân

        Args:
            patient_texts: List các mô tả bệnh của bệnh nhân
            top_k: Số lượng bài báo cần trả về cho mỗi query
            metric_type: Loại metric để tính similarity

        Returns:
            List of lists, mỗi list chứa top-k results cho một query
        """
        results = []
        for text in patient_texts:
            result = self.search(text, top_k, metric_type)
            results.append(result)

        return results

    def __del__(self):
        """Disconnect from Milvus when object is destroyed"""
        try:
            connections.disconnect("default")
            print("Disconnected from Milvus")
        except:
            pass


def main():
    """
    Ví dụ sử dụng
    """
    # Khởi tạo retriever với model đã fine-tune
    # Option 1: Sử dụng model đã fine-tune
    # retriever = PaperRetriever(
    #     model_path="./PAR/checkpoints/best_model.pt",
    #     use_pretrained=False
    # )

    # Option 2: Sử dụng model gốc chưa fine-tune
    retriever = PaperRetriever(use_pretrained=True)

    # Mô tả bệnh nhân (ví dụ)
    patient_description = """
    A 60-year-old female patient with a medical history of hypertension came to our attention because of several neurological deficits that had developed over the last few years, significantly impairing her daily life. Four years earlier, she developed sudden weakness and hypoesthesia of the right hand. The symptoms resolved in a few days and no specific diagnostic tests were performed. Two months later, she developed hypoesthesia and weakness of the right lower limb. On neurological examination at the time, she had spastic gait, ataxia, slight pronation of the right upper limb and bilateral Babinski sign. Brain MRI showed extensive white matter hyperintensities (WMHs), so leukodystrophy was suspected. However, these WMHs were located bilaterally in the corona radiata, basal ganglia, the anterior part of the temporal lobes and the medium cerebellar peduncle (A–D), and were highly suggestive of CADASIL. Genetic testing was performed, showing heterozygous mutation of the NOTCH3 gene (c.994 C<T; exon 6). The diagnosis of CADASIL was confirmed and antiplatelet prevention therapy was started. Since then, her clinical conditions remained stable, and the lesion load was unchanged at follow-up brain MRIs for 4 years until November 2020, when the patient was diagnosed with COVID-19 after a PCR nasal swab. The patient developed only mild respiratory symptoms, not requiring hospitalization or any specific treatment. Fifteen days after the COVID-19 diagnosis, she suddenly developed aphasia, agraphia and worsened right upper limb motor deficit, but she did not seek medical attention. Some days later, she reported these symptoms to her family medical doctor, and a new brain MRI was performed, showing a subacute ischemic area in the left corona radiata (E,F). Therapy with acetylsalicylic acid was switched to clopidogrel as secondary prevention, while her symptoms improved in the next few weeks. The patient underwent a carotid doppler ultrasound and an echocardiogram, which did not reveal any pathological changes. The review of the blood pressure log, both in-hospital and the personal one the patient had kept, excluded uncontrolled hypertension.
    """

    # Tìm kiếm top 10 bài báo liên quan
    print("\n" + "="*80)
    print("Patient Description:")
    print(patient_description)
    print("="*80)

    results = retriever.search(patient_description, top_k=10)

    print(f"\nTop 10 related papers:")
    print("-"*80)
    for i, paper in enumerate(results, 1):
        print(f"{i}. PMID: {paper['pmid']:<15} Score: {paper['score']:.4f}")

    # Ví dụ batch search
    print("\n" + "="*80)
    print("Batch Search Example:")
    print("="*80)

    patient_descriptions = [
        "Patient with diabetes and hypertension",
        "Child with fever and rash symptoms",
        "Elderly patient with memory loss and confusion"
    ]

    batch_results = retriever.batch_search(patient_descriptions, top_k=5)

    for i, (query, results) in enumerate(zip(patient_descriptions, batch_results), 1):
        print(f"\nQuery {i}: {query}")
        print(f"Top 5 papers:")
        for j, paper in enumerate(results, 1):
            print(f"  {j}. PMID: {paper['pmid']:<15} Score: {paper['score']:.4f}")


if __name__ == "__main__":
    main()
