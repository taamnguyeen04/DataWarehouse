import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pymilvus import connections, Collection
import numpy as np
from typing import List, Dict, Union, Optional
import json
from sentence_transformers import CrossEncoder
import os
import warnings
warnings.filterwarnings("ignore")

from model import BiEncoder
from config import Config
from corpus_loader import IndexedCorpusDataset


class PaperRetriever:
    def __init__(self,
                 model_path: str = None,
                 use_pretrained: bool = False,
                 device: str = None,
                 cross_encoder_path: str = None,
                 corpus_file: str = None):
        """
        Khởi tạo retriever cho việc tìm kiếm bài báo

        Args:
            model_path: Đường dẫn đến model đã fine-tune (checkpoint file)
            use_pretrained: Nếu True, sử dụng model gốc chưa fine-tune
            device: Device để chạy model ('cuda', 'cpu', hoặc None để tự động)
            cross_encoder_path: Đường dẫn đến model Cross-Encoder (nếu có)
            corpus_file: Đường dẫn đến file corpus (jsonl) để lấy text cho reranking
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
        connections.connect(alias="default", uri = "https://in05-d7d375279b93833.serverless.aws-eu-central-1.cloud.zilliz.com", token = "35d5a7bc1f263bd6cfdfad6f548fe4f598ec4776cfdb2dfcae4074b27d39bcdcd5938aedd965e895d40fa77b5fe80cc21f6e533f")
        # Load collection
        self.collection = Collection(Config.COLLECTION_NAME)
        self.collection.load()
        print(f"Loaded collection: {Config.COLLECTION_NAME}")
        print(f"Total entities in collection: {self.collection.num_entities}")

        # Load Cross-Encoder
        self.cross_encoder = None
        if cross_encoder_path:
            print(f"Loading Cross-Encoder from {cross_encoder_path}...")
            self.cross_encoder = CrossEncoder(cross_encoder_path, num_labels=1, max_length=512, device=self.device)
        
        # Load Corpus using IndexedCorpusDataset (instant loading with pre-built index)
        self.corpus = None
        if corpus_file:
            print(f"Loading corpus index...")
            self.corpus = IndexedCorpusDataset(corpus_file)

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
               metric_type: str = "L2",
               rerank: bool = True,
               top_k_candidates: int = 50,
               score_threshold: float = 0.5) -> List[Dict]:
        """
        Tìm kiếm top-k bài báo liên quan đến mô tả bệnh nhân

        Args:
            patient_text: Mô tả bệnh của bệnh nhân
            top_k: Số lượng bài báo cần trả về (sau khi filter/rerank)
            metric_type: Loại metric để tính similarity ("IP" cho Inner Product, "L2" cho Euclidean)
            rerank: Có thực hiện rerank bằng Cross-Encoder hay không
            top_k_candidates: Số lượng ứng viên lấy từ Bi-Encoder để rerank
            score_threshold: Ngưỡng điểm Cross-Encoder để lọc bài


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

        # Nếu rerank, lấy nhiều candidates hơn
        search_k = top_k_candidates if (rerank and self.cross_encoder) else top_k

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=search_k,
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

        # Rerank logic with IndexedCorpusDataset and batch prediction
        if rerank and self.cross_encoder and self.corpus:
            # Prepare pairs using IndexedCorpusDataset
            pairs = []
            valid_indices = []
            
            for i, paper in enumerate(retrieved_papers):
                pmid = str(paper['pmid'])
                doc_text = self.corpus.__getitembyid__(pmid)
                if doc_text:
                    pairs.append([patient_text, doc_text])
                    valid_indices.append(i)
            
            if pairs:
                # Predict scores with batch processing for better GPU utilization
                # batch_size=32 is a good default for most GPUs
                cross_scores = self.cross_encoder.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False
                )
                
                # Update scores
                reranked_papers = []
                for idx, score in zip(valid_indices, cross_scores):
                    paper = retrieved_papers[idx]
                    paper['cross_score'] = float(score)
                    
                    # Filter by threshold
                    if paper['cross_score'] >= score_threshold:
                        reranked_papers.append(paper)
                
                # Sort by cross_score
                reranked_papers.sort(key=lambda x: x['cross_score'], reverse=True)
                
                # Take top_k
                retrieved_papers = reranked_papers[:top_k]
            else:
                # Fallback if no text found in corpus for reranking, just slice top_k from initial results
                retrieved_papers = retrieved_papers[:top_k]
        else:
             # No rerank, just slice top_k
             retrieved_papers = retrieved_papers[:top_k]

        return retrieved_papers

    def batch_search(self,
                     patient_texts: List[str],
                     top_k: int = 10,
                     metric_type: str = "L2") -> List[List[Dict]]:
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
    retriever = PaperRetriever(
        model_path="best_model.pt",
        use_pretrained=False,
        cross_encoder_path="./output/cross-encoder-pubmedbert", # Đường dẫn đến Cross-Encoder
        corpus_file=Config.CORPUS_FILE # Đường dẫn đến Corpus
    )

    # Option 2: Sử dụng model gốc chưa fine-tune
    # retriever = PaperRetriever(use_pretrained=True)

    # Mô tả bệnh nhân (ví dụ)
    patient_description = "The mother, 34-years old, primigravida (G0P0), underwent all recommended tests. The first-trimester morphology scan revealed normal crown-rump length, visible nasal bone, and normal nuchal translucency value. Moreover, the double marker for chromosomal aneuploidies (13, 18, and 21) indicated a low-level risk. The TORCH IgM and IgG screening showed no acute or recent infection (negative IgM), and the IgG titer was high. The woman had not been previously exposed to harmful factors that would have justified placing the pregnancy in the high-risk category. The second-trimester morphology scan performed at 22 weeks confirmed the normal development of a female fetus.\nHowever, at 33 weeks of pregnancy, the first abnormal sign was noted. The amniotic fluid quantity started to increase, leading to the diagnosis of polyhydramnios. Another visible alteration was the shape and position of the lower fetal limbs, indicating minor clubfoot and altered fetal biophysical profile. By the time the pregnancy reached 36 weeks, the biophysical variables were severely modified. The fetal heart rate monitored using the non-stress test was worrying. There were significant decelerations, abnormal fetal movement, and poor muscular tonus. Additionally, the quantity of amniotic fluid continued to rise. Cumulatively, these observations led to the decision to deliver the baby prematurely via emergency C-section, 36 weeks into the pregnancy.\nThe C-section was uneventful, and the mother made a fast recovery, but the female newborn weighing 2200 g received an APGAR score of 3. Unfortunately, when thoroughly examined by our team, it was noticeable that the fetus's movement, breathing, and swallowing capacity were impaired, and she was unable to sustain spontaneous breathing. The newborn was constantly and fully dependent on assisted mechanical ventilation. Her condition continued to deteriorate despite all the efforts. Unfortunately, at two months of age, the baby succumbed to respiratory failure and multiple associated complications.\nBased on the clinical signs and paraclinical tests, we were able to establish the following diagnostics: generalized congenital muscular atony, right diaphragmatic hernia, cerebral atrophy, neonatal anemia, bilateral varus equinus, neonatal hypocalcemia, prematurity and low birth weight, ostium secundum atrial heart defect, and tricuspid valve dysplasia.\nThoracic X-rays show reduced ribcage expansion of the right hemithorax, suggestive of right diaphragmatic hernia. The transfontanelar ultrasound and head CT showed moderate cerebral atrophy, mostly in the frontal lobe. The generalized and severe muscular hypotonia was investigated using a muscular biopsy that showed no significant alterations. Both tests for aminoacidopathies and spinal muscular atrophy were negative. Serum levels of creatine kinase (CK) and lactate dehydrogenase (LDH) were high but with a tendency to normalize. The karyotype showed a normal profile 46XX.\nGiven the multitude of heterogenic clinical symptoms, we suspected a genetic syndrome yet to be diagnosed, so we proceeded to perform an Array Comparative Genome Hybridization (aCGH) with Single Nucleotide Polymorphism (SNP). aCGH+SNP was conducted on a blade with 4 ∗ 180,000 (180 K) samples (110.112 CGH samples, 59.647 SNP samples, 3000 replicated samples and 8121 control samples) covering the entire human genome with a spatial resolution of ~25.3 kb DNA (G4890A, design ID: 029830, UCSC hg19, Agilent). The scans were interpreted with the CytoGenomics Agilent software, using standard interpretation parameters with a SureScan Microarray Scanner. The resulting profile was abnormal: there were three areas associated with loss of heterozygosity on chromosomes 1 (q25.1–q25.3) of 6115 kb, 5 (p15.2–p15.1) of 2589 kb and 8 (q11.21–q11.23) of 4830 kb, a duplication of 1104 kb on chromosome 10 in the position q11.22, and duplication of 1193 kb on chromosome 16 in the position p11.2p11.1.\nConsidering this abnormal genetic profile, the parental couple received genetic counseling. Furthermore, we continued to test both partners through Next Generation Sequencing (NGS) by Illumina.\nThe results confirmed the following abnormal genetic profile; TTN (NM_001267550.1, sequencing): heterozygous variant on Chr2(GRCh37):g.179479653G>C—TTN variant c.48681C>G p.(Tyr16227*)—exon 260, heterozygous variant on Chr2(GRCh37):g.179396832_179396833del—TTN variant c.104509_104510del p.(Leu34837Glufs*12)—exon 358 (TTN: NM_001267550.1—reference sequence). The TTN variant c.48681C>G p.(Tyr16227*) creates a premature stop codon. Sanger sequencing also confirmed this variant and was classified as likely pathogenic (class 2). The TTN variant c.104509_104510del p.(Leu34837Glufs*12) creates a shift in the reading frame starting at codon 34837. The new reading frame ends in a stop codon 11 positions downstream. This variant has been confirmed by Sanger sequencing, and it is also classified as likely pathogenic (class 2) ().\nIn light of the clinical outcomes, based on the previous unfortunate experience, the couple agreed to receive genetic counseling. The couple was advised to pursue in vitro fertilization (IVF) with preimplantation genetic testing (PGT-M). Considering the mother's age and weight, her Anti-Müllerian hormone (AMH) serum level and antral follicle count, we used a short-antagonist ovarian stimulation (OS) protocol with 200 UI of FSH (follitropin beta) concomitantly with 150 UI of combined FSH and LH (menotropin). Ten days later, seven oocytes were retrieved through transvaginal, sonographically controlled follicle puncture. Five of them were injected through intracytoplasmic intracytoplasmatic sperm injection (ICSI), resulting in two blastocysts. The embryonic biopsy was performed on day 6 of the blastocyst stage for these two embryos ().\nThe amplification of the entire genome was performed using the SurePlex DNA Amplification System by Illumina Inc. 2018, California US. Using the BlueFuse Multi Analysis Software (Illumina Inc. 2018, San Diego, CA, USA), all 24 chromosomes were detected euploid for embryos. The identification of the mutation TTN gene on exon 358 (father's mutation) and exon 260 (mother's mutation) was performed only for euploid embryos using Sanger sequencing with specific primers on ABI 3500. PCR products for both embryos were purified and sequenced in both senses with a BigDye Terminator v3.1 Cycle Sequencing Kit by Thermo Fisher Scientific. Specific primers were manually designed according to both mutations and tested afterwards using blood samples from the parents. Both embryos tested by PGT-A were euploid. One of them was a carrier of the mother's mutation c.48681C>G p.(Tyr16227), exon 260, and the other was a wild type (WT) for both mutations ().\nWe performed a frozen-thawed embryo transfer in the following cycle, transferring the WT euploid embryo after endometrial preparation with exogenous estrogen. The result was positive, and we confirmed the ongoing viable pregnancy via ultrasound 14 days after.\nThroughout the pregnancy, we performed the non-invasive double marker test (low-risk result) and fetal DNA analysis using maternal blood (low-risk result) and an invasive amniocentesis at 17 weeks of gestation, indicating a normal genetic profile. To test whether or not the second fetus presents a genetic abnormality, we extracted the DNA directly from the amniotic fluid. Targeted sequencing was performed on both DNA strands of the relevant TTN region. The reference sequence is TTN: NM_001267550.2. To exclude maternal cell contamination (MCC), we analyzed 15 STR autosomal markers plus amelogenin using the PowerPlex 16HS multiplex kit (Promega, Madison, Wisconsin, USA). Moreover, all the non-invasive ultrasound scans showed a normal growth rate and organ development. The evolution of the pregnancy was uneventful, and at 38 weeks, we carried out the C-section delivery of a healthy female baby of 2990 g, receiving an APGAR score of 9."


    # Tìm kiếm top 10 bài báo liên quan
    print("\n" + "="*80)
    print("Patient Description:")
    print(patient_description)
    print("="*80)

    # Search với rerank
    results = retriever.search(
        patient_description, 
        top_k=10, 
        rerank=True, 
        top_k_candidates=50, 
        score_threshold=0.0 # Điều chỉnh ngưỡng tùy model
    )

    print(f"\nTop 10 related papers:")
    print("-"*80)
    for i, paper in enumerate(results, 1):
        score_display = f"Cross-Score: {paper.get('cross_score', 0):.4f}" if 'cross_score' in paper else f"Bi-Score: {paper['score']:.4f}"
        print(f"{i}. PMID: {paper['pmid']:<15} {score_display}")

if __name__ == "__main__":
    main()
