import json
import logging
import os

import requests
from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

RXNFP_URL = os.environ.get("RXNFP_URL", "http://localhost:8087/api/v1/run")


class RXNFPEmbed(EmbeddingFunction):
    def __call__(self, inps: Documents) -> Embeddings:
        response = requests.post(
            RXNFP_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"rxns": list(inps)}),
        )
        try:
            return response.json()["fps"]
        except Exception:
            logger.error("Failed to get RXNFP embeddings: %s", response.text)
            return None
