
import os
import logging

from dotenv import load_dotenv
from groq import Groq, APIStatusError

from src.retrieval.patent_retrieval import PatentRetrieval


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentRAG:

    MODEL = "openai/gpt-oss-120b"

    # Keep the total request comfortably below the 8,000 TPM limit.
    MAX_QUERY_CHARS = 3500
    MAX_ABSTRACT_CHARS = 1000
    TOP_K = 3
    MAX_OUTPUT_TOKENS = 1200

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("Groq API key not found in .env")

        self.client = Groq(api_key=api_key)

        self.retrieval = PatentRetrieval()
        self.retrieval.load_index()

    def analyze_patent(self, query):

        if not query or not query.strip():
            raise ValueError("Patent text cannot be empty.")

        # Limit the uploaded patent text before sending it to Groq.
        query = query.strip()[:self.MAX_QUERY_CHARS]

        logger.info("Retrieving similar patents...")

        patents = self.retrieval.search(
            query,
            top_k=self.TOP_K
        )

        context_parts = []

        for i, patent in enumerate(patents, 1):

            abstract = str(patent.get("abstract", ""))

            # Keep only a concise portion of each retrieved abstract.
            abstract = abstract[:self.MAX_ABSTRACT_CHARS]

            context_parts.append(
                f"""
Patent {i}
CPC Category: {patent.get('category', 'Unknown')}
Similarity: {patent.get('similarity_score', 0):.3f}
Abstract: {abstract}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a patent research assistant.

Analyze the patent using only the supplied patent text
and retrieved patent abstracts.

PATENT TEXT:
{query}

RETRIEVED PATENTS:
{context}

Provide a concise report with these sections:

1. Patent Summary
2. Common Technical Features
3. Differences Between Retrieved Patents
4. Potential Areas for Further Research

Rules:
- Use evidence from the supplied text.
- Do not invent patent numbers, titles, or technical details.
- Clearly distinguish evidence from suggestions.
- If evidence is insufficient, say so.
- Do not make legal conclusions about patentability or infringement.
"""

        logger.info("Generating concise patent analysis...")

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful patent research assistant. "
                            "Be concise, accurate and evidence-based."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=self.MAX_OUTPUT_TOKENS
            )

            return response.choices[0].message.content

        except APIStatusError as e:
            logger.exception("Groq API request failed")

            if e.status_code == 413:
                raise RuntimeError(
                    "The patent analysis request exceeded Groq's token "
                    "limit. Please try a shorter patent text."
                ) from e

            raise RuntimeError(
                f"Groq API error ({e.status_code}). "
                "Please check the API configuration and limits."
            ) from e


if __name__ == "__main__":

    rag = PatentRAG()

    query = """
    A rechargeable battery system with improved
    cooling and electrical power management.
    """

    result = rag.analyze_patent(query)

    print("\n===== PATENT ANALYSIS =====\n")
    print(result)
