import pandas as pd
import numpy as np
import json
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Optional, Union

from the_retry import retry

try:
    from openai import OpenAI

    openai_available = True
except ImportError:
    openai_available = False
    OpenAI = None

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastTemplateSearchEngine:
    """Template search using pre-computed embeddings from CSV - Auto-Model Detection Version"""

    def __init__(self, csv_with_embeddings: str, openai_key: Optional[str] = None):
        logger.info(
            f"Initializing Fast Template Search Engine from {csv_with_embeddings}"
        )

        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        self.templatefile = csv_with_embeddings
        self.df = pd.read_csv(csv_with_embeddings, low_memory=False)
        self.embedding_model_name = self._detect_embedding_model()
        logger.info(f"Detected embedding model: {self.embedding_model_name}")

        self._prepare_embeddings()
        logger.info(
            f"Search Engine Ready: {len(self.embeddings)} templates, {self.embeddings.shape[1]}D, model type: {self.model_type}"
        )

        self._cache = {}

    def _detect_embedding_model(self) -> str:
        if "embedding_metadata" in self.df.columns:
            metadata = self.df["embedding_metadata"].iloc[0]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            model_name = metadata.get("model", "").strip()
            if model_name:
                return model_name

        if "embedding_model" in self.df.columns:
            model_name = str(self.df["embedding_model"].iloc[0]).strip()
            if model_name:
                return model_name

        embedding_dim = self.df.filter(regex=r"^embedding_\\d+$").shape[1]
        if embedding_dim == 1536:
            return "text-embedding-ada-002"
        elif embedding_dim == 3072:
            return "text-embedding-3-large"
        elif embedding_dim == 768:
            return "bge-large-en-v1.5"
        elif embedding_dim == 384:
            return "all-MiniLM-L6-v2"
        else:
            raise ValueError(f"Unknown embedding dimension: {embedding_dim}")

    def _prepare_embeddings(self):
        def safe_json_to_array(x):
            if isinstance(x, str):
                try:
                    arr = np.array(json.loads(x), dtype=np.float32)
                    if arr.ndim == 1:
                        return arr
                except Exception as e:
                    logger.warning(f"Skipping row due to JSON error: {e}")
            return None

        embedding_cols = [c for c in self.df.columns if re.match(r"embedding_\d+$", c)]
        has_embedding_columns = len(embedding_cols) > 0
        has_embedding_vector = "embedding_vector" in self.df.columns

        if (
            "openai" in self.embedding_model_name
            or self.embedding_model_name.startswith("text-embedding")
        ):
            if not openai_available:
                raise ImportError(
                    "OpenAI module not found. Please install openai>=1.0.0 to use OpenAI embeddings."
                )
            self.model_type = "openai"
            self.openai_model = self.embedding_model_name
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.query_encoder = None
        else:
            self.model_type = "sentence_transformers"
            self.query_encoder = SentenceTransformer(self.embedding_model_name)
            self.openai_model = None
            self.openai_client = None

        if has_embedding_columns:
            embedding_df = self.df[embedding_cols].apply(pd.to_numeric, errors="coerce")
            embedding_df = embedding_df.dropna(how="any")
            if embedding_df.empty:
                raise ValueError(
                    "No valid embedding rows found in CSV (all embedding columns contain NaN or non-numeric values)."
                )
            self.embeddings = embedding_df.values.astype(np.float32)
            self.templates = self.df.loc[embedding_df.index].reset_index(drop=True)
        elif has_embedding_vector:
            embeddings_list = self.df["embedding_vector"].apply(safe_json_to_array)
            valid_idx = embeddings_list.dropna().index
            if len(valid_idx) == 0:
                raise ValueError(
                    "No valid embedding_vector rows found in CSV (all rows failed to parse as JSON)."
                )
            self.templates = self.df.loc[valid_idx].reset_index(drop=True)
            self.embeddings = np.stack(embeddings_list.dropna().to_numpy())
        else:
            raise ValueError(
                "No embedding columns found in CSV (neither embedding_0... nor embedding_vector)."
            )

    def encode_query(self, query: Union[str, List[str]]) -> np.ndarray:
        if self.model_type == "openai":
            response = self.openai_client.embeddings.create(
                model=self.openai_model, input=query, encoding_format="float"
            )
            res = np.array([each.embedding for each in response.data], dtype=np.float32)
            return res
        else:
            return self.query_encoder.encode([query]).astype(np.float32)

    def _parse_query(self, query: str) -> Optional[List[str]]:
        pattern = re.compile(
            r"reaction (involves|represents) (?P<rxn_name>.+?) (of|between|from|within|converting|to convert) (?P<substrate>.+?), (focusing on|where) (?P<fgs>.+?),.* classified as (?P<rxn_class>.+?)\."
        )
        match = re.search(pattern, query)
        if match is None:
            return None
        return list(match.groupdict().values())

    def encode_query_by_components(self, query: str) -> np.ndarray:
        text_components = self._parse_query(query)
        if text_components is None:
            logger.warning(
                "Description does not match expected format. Encoding full description."
            )
            return self.encode_query(query)

        embeddings = self.encode_query(text_components)
        return embeddings

    def search_scores(
        self,
        query_similarity: float,
        template_frequency: int,
        C: int = 100,
        alpha: float = 0.5,
    ) -> float:
        return alpha * query_similarity + (1 - alpha) * (
            template_frequency / (template_frequency + C)
        )

    @retry(attempts=3, backoff=5)
    def search_templates(
        self,
        query: str,
        similarity_threshold: float = 0.1,
        max_results: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Dict:

        # Create cache key that includes is_ring_break
        cache_key = f"{query}_{similarity_threshold}_{max_results}_{alpha}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        query_embedding = self.encode_query(query)

        if query_embedding.shape[1] != self.embeddings.shape[1]:
            logger.error("Dimension mismatch: Query vs. Template embeddings")
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "error": "dimension_mismatch",
            }

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        above_threshold_indices = np.where(similarities >= similarity_threshold)[0]

        # Apply ring_break filter if specified
        # if is_ring_break and 'ring_break' in self.templates.columns:
        #     ring_break_values = self.templates.iloc[above_threshold_indices]['ring_break'].values
        #     ring_break_mask = (ring_break_values == 1)
        #     above_threshold_indices = above_threshold_indices[ring_break_mask]
        # elif is_ring_break and 'ring_break' not in self.templates.columns:
        #     logger.warning("Ring break is set to True, but 'ring_break' column is not found in the templates. Ignoring ring break filter.")

        if len(above_threshold_indices) > 0:
            sorted_order = np.argsort(similarities[above_threshold_indices])[::-1]
            final_indices = above_threshold_indices[sorted_order]
            # if max_results:
            #     final_indices = final_indices[:max_results]
        else:
            final_indices = []

        results = []
        for i, idx in enumerate(final_indices):
            row = self.templates.iloc[idx]
            similarity_score = similarities[idx]
            library_occurence = int(row.get("library_occurence", 1))
            search_score = self.search_scores(
                similarity_score, library_occurence, alpha=alpha
            )
            results.append(
                {
                    "rank": i + 1,
                    "template_code": str(row["template_code"]),
                    "description": str(row.get("llm_description", "")),
                    "similarity_score": round(float(similarity_score), 3),
                    "library_occurence": library_occurence,
                    "search_score": round(float(search_score), 3),
                    "retro_template": str(row["retro_template"]),
                    "ring_break": bool(
                        row.get("ring_break", 0)
                    ),  # Default to 0 if column doesn't exist
                }
            )

        # sort results by search_score
        results = sorted(results, key=lambda x: x["search_score"], reverse=True)

        if max_results:
            results = results[:max_results]

        for i, res in enumerate(results):
            res["rank"] = i + 1

        res = {
            "query": query,
            "results": results,
            "total_found": len(above_threshold_indices),
        }
        self._cache[cache_key] = res
        return res

    def print_results(self, results: Dict):
        print(f"\n Search results for: '{results['query']}'")
        print(f"Found: {results['total_found']} templates")
        print("-" * 60)
        for result in results["results"]:
            print(
                f"{result['rank']}. {result['template_code']} (Score: {result['similarity_score']})"
            )
            print(f"   {result['description'][:200]}...")
            print(f"   SMARTS: {result['retro_template']}")
            print()

    def get_model_info(self) -> Dict:
        return {
            "model_type": self.model_type,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embeddings.shape[1],
            "template_count": len(self.embeddings),
        }


def create_engine_from_csv(
    csv_path: str, openai_key: Optional[str] = None
) -> FastTemplateSearchEngine:
    return FastTemplateSearchEngine(csv_path, openai_key=openai_key)
