
import os
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


class PatentRetrieval:

    def __init__(self):
        print("Loading SBERT model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.index = None
        self.patents = None

    def build_index(self, csv_path):

        print("Loading patent dataset...")

        df = pd.read_csv(csv_path)

        self.patents = df

        texts = (
            df["abstract"].fillna("").astype(str).tolist()
        )

        print("Generating patent embeddings...")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print("FAISS index created!")
        print("Total patents indexed:", self.index.ntotal)

        os.makedirs("models", exist_ok=True)

        faiss.write_index(
            self.index,
            "models/patent_faiss.index"
        )

        self.patents.to_csv(
            "models/patent_metadata.csv",
            index=False
        )

        print("Index and metadata saved.")


    def load_index(self):

        print("Loading saved FAISS index...")

        self.index = faiss.read_index(
            "models/patent_faiss.index"
        )

        self.patents = pd.read_csv(
            "models/patent_metadata.csv"
        )

        print("FAISS index loaded successfully!")
        print("Total patents:", self.index.ntotal)


    def search(self, query, top_k=5):

        if self.index is None:
            raise ValueError("Build the index before searching.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding, top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            patent = self.patents.iloc[idx]

            results.append({
                "category": patent["category"],
                "abstract": patent["abstract"],
                "similarity_score": float(score)
            })

        return results




if __name__ == "__main__":

    retrieval = PatentRetrieval()

    retrieval.load_index()

    results = retrieval.search(
        "A battery system using advanced electrical components",
        top_k=5
    )

    for result in results:
        print(result)
