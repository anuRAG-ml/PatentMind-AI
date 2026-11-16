
import logging
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ner.patent_ner import PatentNER
from src.retrieval.patent_retrieval import PatentRetrieval
from src.rag.patent_rag import PatentRAG


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("patentmind")
CPC_INFO = {
    "A": {
        "name": "Human Necessities",
        "description": "Technologies related to human needs, agriculture, health, and daily life."
    },
    "B": {
        "name": "Performing Operations; Transporting",
        "description": "Technologies involving manufacturing, processing, and transportation."
    },
    "C": {
        "name": "Chemistry; Metallurgy",
        "description": "Chemical processes, materials, and metallurgy."
    },
    "D": {
        "name": "Textiles; Paper",
        "description": "Technologies involving textiles, fabrics, and paper."
    },
    "E": {
        "name": "Fixed Constructions",
        "description": "Buildings, construction, and structural engineering."
    },
    "F": {
        "name": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "description": "Mechanical systems, engines, pumps, lighting, heating, and related technologies."
    },
    "G": {
        "name": "Physics",
        "description": "Physics-related technologies, measurement, optics, and computing."
    },
    "H": {
        "name": "Electricity",
        "description": "Electrical systems, electronic circuits, and electric communication."
    },
    "Y": {
        "name": "General Tagging of New Technological Developments",
        "description": "Emerging and cross-sectional technologies."
    }
}
app = FastAPI(
    title="PatentMind-AI",
    description="AI-powered patent research and analysis API",
    version="1.0.0"
)

print("Loading PatentMind-AI components...")

classifier = joblib.load("models/patent_classifier_pipeline.joblib")

ner = PatentNER()

retrieval = PatentRetrieval()
retrieval.load_index()

rag = PatentRAG()

print("All components loaded successfully!")


class PatentRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "PatentMind-AI API is running!"}


@app.post("/classify")
def classify_patent(request: PatentRequest):
    try:
        prediction = str(classifier.predict([request.text])[0])

        category_info = CPC_INFO.get(
            prediction,
            {
                "name": "Unknown Category",
                "description": "No description available."
            }
        )

        top_predictions = []

        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba([request.text])[0]
            classes = classifier.classes_

            ranked = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            )[:3]

            top_predictions = [
                {
                    "category": str(category),
                    "name": CPC_INFO.get(str(category), {}).get(
                        "name", "Unknown Category"
                    ),
                    "probability": round(float(probability) * 100, 2)
                }
                for category, probability in ranked
            ]

        return {
            "predicted_category": prediction,
            "category_name": category_info["name"],
            "description": category_info["description"],
            "top_predictions": top_predictions
        }

    except Exception as e:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/extract-entities")
def extract_entities(request: PatentRequest):
    try:
        entities = ner.extract_entities(request.text)
        return {"entities": entities}
    except Exception as e:
        logger.exception("Entity extraction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_patents(request: PatentRequest):
    try:
        results = retrieval.search(request.text, top_k=5)
        return {"similar_patents": results}
    except Exception as e:
        logger.exception("Patent retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze_patent(request: PatentRequest):
    try:
        analysis = rag.analyze_patent(request.text)
        return {"analysis": analysis}
    except Exception as e:
        logger.exception("RAG patent analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
