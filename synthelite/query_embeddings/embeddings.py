import asyncio
import os
from typing import List, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import logging
from pathlib import Path
import argparse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str, batch_size: int = 32):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.embedding_dim = self.get_embedding_dim()
        self.batch_size = batch_size
        try:
            self.model = SentenceTransformer(model_name)
        except:
            from openai import OpenAI

            self.model = OpenAI(max_retries=3)

    def get_embedding_dim(self):
        if self.model_name == "all-MiniLM-L6-v2":
            return 384
        elif self.model_name == "text-embedding-ada-002":
            return 1536
        elif self.model_name == "text-embedding-3-large":
            return 3072
        elif self.model_name == "bge-large-en-v1.5":
            return 768
        else:
            raise ValueError(f"Unknown embedding model: {self.model_name}")

    async def _openai_batch_embed(self, texts: List[str], batch_id: int = 0):
        """Generate embeddings using OpenAI API."""
        try:
            response = self.model.embeddings.create(
                model=self.model_name, input=texts, encoding_format="float"
            )
            embeddings = np.array(
                [e.embedding for e in response.data], dtype=np.float32
            )
            return embeddings, True
        except Exception as e:
            logger.error(
                f"Error generating embeddings with OpenAI API for batch {batch_id}: {e}"
            )
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32), False

    def embed_texts(self, texts: List[str], batch_id: int = 0) -> np.ndarray:
        if self.model_name.startswith("text-embedding"):
            embeddings, success = asyncio.run(self._openai_batch_embed(texts, batch_id))
        else:
            embeddings = self.model.encode(texts)
            success = True

        return embeddings, success

    def embed_df(
        self, description_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> pd.DataFrame:
        result_df = description_df.copy()
        result_df = result_df.dropna(subset=["llm_description"])
        result_df = result_df.drop(
            columns=["full_response, description_metadata"], errors="ignore"
        )
        for col in ["embedding_vector", "embedding_metadata"]:
            if col not in result_df.columns:
                result_df[col] = None
        descriptions_to_process = [
            (i, row["llm_description"])
            for i, (_, row) in enumerate(result_df.iterrows())
            if pd.notna(row["llm_description"])
            and not self._is_successful_embedding(row)
        ]

        logger.info(
            f"Found {len(descriptions_to_process)} descriptions to process for embeddings."
        )

        if not descriptions_to_process:
            logger.info("No descriptions to process. Returning original DataFrame.")
            return result_df

        for batch_id, batch_start in enumerate(
            range(0, len(descriptions_to_process), self.batch_size)
        ):
            batch_descriptions = descriptions_to_process[
                batch_start : batch_start + self.batch_size
            ]
            row_indices = [desc[0] for desc in batch_descriptions]
            texts = [desc[1] for desc in batch_descriptions]
            batch_embeddings, batch_success = self.embed_texts(texts, batch_id=batch_id)

            for row_id, embedding in zip(row_indices, batch_embeddings):
                embedding_json = json.dumps(embedding.tolist())
                metadata = {"success": batch_success, "model": self.model_name}
                result_df.at[row_id, "embedding_vector"] = embedding_json
                result_df.at[row_id, "embedding_metadata"] = json.dumps(metadata)

            if save_path:
                result_df.to_csv(save_path, index=False)
        if save_path:
            result_df.to_csv(save_path, index=False)
        return result_df

    def _is_successful_embedding(self, row):
        if not isinstance(row["embedding_vector"], str):
            return False
        embedding_metadata = (
            json.loads(row["embedding_metadata"]) if row["embedding_metadata"] else {}
        )
        return embedding_metadata.get("success", False)


def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Add embeddings to CSV file")
    parser.add_argument("--input_csv", type=str, help="Input CSV file with templates")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file with embeddings (optional)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regenerate embeddings even if they exist",
    )

    return parser.parse_args()


def add_embeddings_to_csv(
    input_csv: str,
    output_csv: str = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    force_regenerate: bool = False,
):
    """Add vector embeddings with custom model and force option"""

    if output_csv is None:
        # Include model name in filename
        model_suffix = model_name.split("/")[-1].replace("-", "_")
        output_csv = input_csv.replace(".csv", f"_{model_suffix}.csv")

    if os.path.exists(output_csv):
        logger.info(f"Output file {output_csv} already exists. Resume embedding")
        df = pd.read_csv(output_csv)
    elif input_csv.endswith(".csv.gz"):
        logger.info(f"Loading templates from {input_csv}")
        df = pd.read_csv(input_csv, compression="gzip", sep="\t")
    else:
        logger.info(f"Loading templates from {input_csv}")
        df = pd.read_csv(input_csv)

    # Filter templates with descriptions
    # valid_templates = df[df['llm_description'].notna()].copy()
    # logger.info(f"Found {len(valid_templates)} templates with descriptions")

    # # Check if embeddings already exist AND same model
    # if 'embedding_vector' in df.columns and not force_regenerate:
    #     existing_embeddings = df['embedding_vector'].notna().sum()
    #     existing_model = df['embedding_model'].iloc[0] if 'embedding_model' in df.columns else None

    #     if existing_embeddings > 0:
    #         if existing_model == model_name:
    #             logger.info(f"Found {existing_embeddings} existing embeddings with same model. Skipping generation.")
    #             if output_csv and output_csv != input_csv:
    #                 df.to_csv(output_csv, index=False)
    #                 logger.info(f"Copied to new file: {output_csv}")
    #                 return output_csv
    #             return input_csv
    #         else:
    #             logger.info(f"Found embeddings with different model ({existing_model} vs {model_name}). Regenerating...")

    # Initialize embedding model
    logger.info(f"Loading embedding model: {model_name}")
    embedding_model = EmbeddingModel(model_name=model_name, batch_size=batch_size)
    logger.info(
        f"Embedding model loaded: {model_name} with dimension {embedding_model.embedding_dim}"
    )

    embedding_model.embed_df(df, save_path=output_csv)
    # model = SentenceTransformer(model_name)

    # # Generate embeddings
    # descriptions = valid_templates['llm_description'].tolist()
    # logger.info(f"Generating embeddings for {len(descriptions)} descriptions...")

    # embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=32)
    # logger.info(f"Generated embeddings: {embeddings.shape}")

    # Add embeddings to df
    # embedding_strings = []
    # for embedding in embeddings:
    #     embedding_json = json.dumps(embedding.tolist())
    #     embedding_strings.append(embedding_json)

    # # Initialize embedding columns for all rows
    # df['embedding_vector'] = None
    # df['embedding_dimension'] = None
    # df['embedding_model'] = None
    # df['embedding_timestamp'] = None

    # # Add embeddings to valid templates
    # from datetime import datetime
    # current_time = datetime.now().isoformat()

    # valid_template_indices = valid_templates.index
    # for i, idx in enumerate(valid_template_indices):
    #     df.at[idx, 'embedding_vector'] = embedding_strings[i]
    #     df.at[idx, 'embedding_dimension'] = len(embeddings[0])
    #     df.at[idx, 'embedding_model'] = model_name
    #     df.at[idx, 'embedding_timestamp'] = current_time

    # Save updated CSV
    logger.info(f"Saving templates with embeddings to {output_csv}")
    # df.to_csv(output_csv, index=False)

    # logger.info(f"Templates with embeddings saved to {output_csv}")
    # logger.info(f"Total rows: {len(df)}, Rows with embeddings: {len(valid_templates)}")

    # return output_csv


if __name__ == "__main__":
    # input_file = "your_input_file.csv"  # Replace with your actual input file path
    args = parse_args()
    output_file = add_embeddings_to_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,  # Will generate based on model name
        model_name=args.model_name,
        batch_size=args.batch_size,
        force_regenerate=True,
    )
