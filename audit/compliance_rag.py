import chromadb
from chromadb.utils import embedding_functions

class FDARAGAuditor:
    def __init__(self, db_path="./fda_vector_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="fda_compliance_standards",
            embedding_function=self.embedding_fn
        )
        if self.collection.count() == 0:
            self._seed_regulatory_corpus()

    def _seed_regulatory_corpus(self):
        regulations = [
            {
                "id": "samd_risk_categorization",
                "text": "Software as a Medical Device (SaMD): Clinical Evaluation - The manufacturer must categorize the SaMD based on the healthcare situation or condition and the significance of information provided.",
                "source": "FDA Guidance: Software as a Medical Device (SaMD) Clinical Evaluation"
            },
            {
                "id": "iec_62304_software_safety",
                "text": "IEC 62304 Medical Device Software - Software Safety Classification: Class A, Class B, Class C (death or serious injury possible). Software architecture must segregate safety-critical components.",
                "source": "IEC 62304 / FDA Recognition Number 9-106"
            },
            {
                "id": "explainability_clinical_review",
                "text": "Medical Device Software Explainability: High-risk diagnostic algorithms (such as CAD systems for mammography) must provide interpretable feature attribution maps (Grad-CAM) to assist clinicians.",
                "source": "FDA Good Machine Learning Practice (GMLP)"
            }
        ]
        self.collection.add(
            documents=[r["text"] for r in regulations],
            metadatas=[{"source": r["source"]} for r in regulations],
            ids=[r["id"] for r in regulations]
        )

    def audit_documentation(self, project_summary: str, n_results: int = 2):
        results = self.collection.query(
            query_texts=[project_summary],
            n_results=n_results
        )
        retrieved_docs = results['documents'][0]
        retrieved_sources = results['metadatas'][0]
        
        audit_report = []
        for doc, meta in zip(retrieved_docs, retrieved_sources):
            audit_report.append({"source": meta['source'], "clause": doc})
        return audit_report
