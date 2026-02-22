import chromadb
from chromadb.config import Settings
import json

from dotenv import load_dotenv
import os
from .rxnfp import RXNFPEmbed
from pydantic import BaseModel, Field, model_validator
from typing import Optional

load_dotenv()
CHROMA_REAXYS_KEY = os.getenv("CHROMA_REAXYS_KEY")


def create_chroma_client():
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8067")),
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
            chroma_client_auth_credentials=f"admin:{CHROMA_REAXYS_KEY}",
        ),
    )


class SimilarRxn(BaseModel):
    client: chromadb.HttpClient = Field(default_factory=create_chroma_client)
    collection: Optional[chromadb.Collection] = None

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, rxn: str):
        rxns = self.most_similar_rxns(rxn)

        # Prioritize results with dois
        for r in rxns["metadatas"][0]:
            if "doi" in r:
                return r

        return rxns["metadatas"][0][0]

    def most_similar_rxns(self, rxn: str, n_results: int = 5):
        results = self.collection.query(
            query_texts=[rxn],
            n_results=n_results,
        )
        return results

    @model_validator(mode="after")
    def setup_collection(self):
        if self.collection is None:
            self.collection = self.client.get_collection(
                name="rxns_doi",
                embedding_function=RXNFPEmbed(),
            )
        return self


if __name__ == "__main__":
    sr = SimilarRxn()
    a = sr(
        "CC(C)C(O)[C@@]1(C)CC[C@@H](C(=O)O)C(C)(C)O1>>CC(C)C(=O)C1(C)CCC(C(=O)O)C(C)(C)O1"
    )
    print(a)
