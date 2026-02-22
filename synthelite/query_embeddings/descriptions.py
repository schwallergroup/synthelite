import pandas as pd
import litellm
import time
import json
import re
import os
import asyncio
import aiohttp
from typing import Dict, List, Optional
from pathlib import Path
import logging
import argparse

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# import weave
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateDescriptionGenerator:
    """Generate natural language descriptions for SMARTS reaction templates using LLM"""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
        retro: bool = True,
        log_frequency: float = 1.0,
        batch_submission: bool = False,
    ):
        """
        Initialize the description generator
        """
        self.model = model
        self.max_concurrent = max_concurrent

        if "claude" in model.lower():
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif not os.getenv("ANTHROPIC_API_KEY"):
                logger.warning("ANTHROPIC_API_KEY not found in environment")
        elif "gpt" in model.lower() or "openai" in model.lower():
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            elif not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not found in environment")

        self.last_request_time = 0
        if "claude" in model.lower():
            self.min_request_interval = 0.01
        else:
            self.min_request_interval = 0.2

        logger.info(f"Initialized with model: {self.model}")
        logger.info(f" Max concurrent requests: {self.max_concurrent}")
        logger.info(f" Rate limit: {1/self.min_request_interval:.1f} requests/second")

        self._template_col = "retro_template" if retro else "forward_template"
        self.retro = retro

        self.log_frequency = log_frequency
        self.batch_submission = batch_submission

        self.prompt_template = self.load_prompt_template()

    def load_prompt_template(self) -> str:
        """Load the prompt template from description_prompt.py"""
        try:
            from description_prompt import prompt_forward, prompt_retro

            logger.info("Loaded custom prompt template")
            if self.retro:
                return prompt_retro
            else:
                return prompt_forward
        except ImportError:
            logger.warning("Could not import prompt, using fallback")
            return """You are a chemistry expert. Analyze this retrosynthetic reaction template in SMARTS format and provide a concise 1-2 sentence description of the chemical transformation.

<smarts_reaction>
{{SMARTS_REACTION}}
</smarts_reaction>

Provide your analysis in <reaction_analysis> tags, then your final description in <description> tags."""

    def parse_llm_response(self, response: str) -> Dict[str, str]:
        result = {
            "description": None,
            "analysis": None,
            "raw_response": response.strip(),
        }
        try:
            desc_match = re.search(
                r"<description>\s*(.*?)\s*</description>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if desc_match:
                result["description"] = desc_match.group(1).strip()

            analysis_match = re.search(
                r"<reaction_breakdown>\s*(.*?)\s*</reaction_breakdown>",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if analysis_match:
                result["analysis"] = analysis_match.group(1).strip()

            if not result["description"]:
                sentences = response.split(".")
                result["description"] = (
                    sentences[0].strip() + "." if sentences else response.strip()
                )
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            result["description"] = response.strip()
        return result

    def resume_generation(self, output_path: str, batch_size: int = 50) -> pd.DataFrame:
        try:
            existing_df = pd.read_csv(output_path)
            # completed_count = existing_df['llm_description'].notna().sum()
            success_count = existing_df.apply(
                lambda row: self._is_success_row(row), axis=1
            ).sum()
            total_count = len(existing_df)
            if total_count - success_count == 0:
                logger.info("All descriptions already complete!")
                return existing_df
            return self.generate_batch_descriptions(
                existing_df, output_path, batch_size
            )
        except Exception as e:
            logger.error(f"Error resuming generation: {e}")
            raise e

    # @weave.op()
    async def generate_single_description_async(
        self, smarts_reaction: str, template_code: str = ""
    ) -> Dict:
        try:
            await asyncio.sleep(self.min_request_interval)
            prompt = self.prompt_template.replace(
                "{{SMARTS_REACTION}}", smarts_reaction
            )
            logger.info(f"Generating description for template {template_code}")
            response = await asyncio.to_thread(
                litellm.completion,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1,
            )
            full_response = response.choices[0].message.content
            parsed = self.parse_llm_response(full_response)
            token_usage = (
                getattr(response.usage, "total_tokens", None)
                if hasattr(response, "usage")
                else None
            )
            return {
                "success": True,
                "description": parsed["description"],
                "analysis": parsed["analysis"],
                "full_response": full_response,
                "template_code": template_code,
                "smarts_reaction": smarts_reaction,
                "model_used": self.model,
                "tokens_used": token_usage,
                "timestamp": pd.Timestamp.now(),
            }
        except Exception as e:

            logger.error(
                f"Error generating description for template {template_code}: {str(e)}"
            )
            return {
                "success": False,
                "description": f"Error: {str(e)}",
                "analysis": None,
                "full_response": None,
                "template_code": template_code,
                "smarts_reaction": smarts_reaction,
                "model_used": self.model,
                "tokens_used": None,
                "timestamp": pd.Timestamp.now(),
                "error": str(e),
            }

    def _get_forward_templates(self, retro_template: str):
        products, _, reactants = retro_template.split(">")
        return f"{reactants.strip()}>>{products.strip()}"

    # @weave.op()
    async def generate_batch_descriptions_async(
        self, batch_templates: List[tuple]
    ) -> List[Dict]:
        # template_col = 'retro_template' if self.retro else 'forward_template'
        # if template_col == 'forward_template' and template_col not in batch_templates[0][1]:
        #     template_col = 'retro_template'
        messages = []
        indices = []
        for index, row in batch_templates:
            smarts_reaction = str(row[self._template_col])
            messages.append(
                [
                    {
                        "role": "user",
                        "content": self.prompt_template.replace(
                            "{{SMARTS_REACTION}}", smarts_reaction
                        ),
                    }
                ]
            )
            indices.append(index)
        try:
            responses = await asyncio.to_thread(
                litellm.batch_completion,
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
                num_retries=3,
            )

            res = []
            for index, response in zip(indices, responses):
                full_response = response.choices[0].message.content
                parsed = self.parse_llm_response(full_response)
                token_usage = getattr(response, "usage", None)
                total_tokens, prompt_tokens, completion_tokens = None, None, None
                if token_usage is not None:
                    total_tokens = token_usage.get("total_tokens", None)
                    prompt_tokens = token_usage.get("prompt_tokens", None)
                    completion_tokens = token_usage.get("completion_tokens", None)
                res.append(
                    (
                        index,
                        {
                            "success": True,
                            "description": parsed["description"],
                            "analysis": parsed["analysis"],
                            "full_response": full_response,
                            "model_used": self.model,
                            "tokens_used": total_tokens,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "timestamp": pd.Timestamp.now(),
                        },
                    )
                )
            return res
        except Exception as e:
            logger.error(f"Error processing batch{batch_templates}: {e}")
            return [
                (
                    each[0],
                    {
                        "success": False,
                        "description": f"Batch error: {str(e)}",
                        "analysis": None,
                        "full_response": None,
                        "model_used": self.model,
                        "tokens_used": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "timestamp": pd.Timestamp.now(),
                        "error": str(e),
                    },
                )
                for each in batch_templates
            ]

    def get_description_process_fn(self, batch_submission: bool = False):
        def _get_rows_to_process(self, templates_df: pd.DataFrame, start_index: int):
            result_df = templates_df.copy()
            for col in [
                "llm_description",
                "llm_analysis",
                "full_response",
                "description_metadata",
            ]:
                if col not in result_df.columns:
                    result_df[col] = None

            if (
                self._template_col == "forward_template"
                and self._template_col not in templates_df.columns
            ):
                templates_df[self._template_col] = templates_df["retro_template"].apply(
                    self._get_forward_templates
                )
            # templates_to_process = [(i, row) for i, (index, row) in enumerate(templates_df.iterrows()) if i >= start_index and pd.isna(result_df.at[index, 'llm_description']) or not self._is_success_row(result_df, index)]
            templates_to_process = [
                (i, row)
                for i, (index, row) in enumerate(templates_df.iterrows())
                if i >= start_index and not self._is_success_row(row)
            ]
            logger.info(
                "Found {} templates to process".format(len(templates_to_process))
            )

            return result_df, templates_to_process

        # @weave.op(tracing_sample_rate=self.log_frequency)
        async def process_rows_in_batch_async(
            self,
            templates_df: pd.DataFrame,
            output_path: str = "templates_with_descriptions.csv",
            batch_size: int = 50,
            start_index: int = 0,
            save_progress: bool = True,
        ) -> pd.DataFrame:
            logger.info(f" Starting ASYNC batch description generation")
            result_df, templates_to_process = _get_rows_to_process(
                self, templates_df, start_index
            )

            # semaphore = asyncio.Semaphore(self.max_concurrent)
            # async def process_template_with_semaphore(index, row):
            #     async with semaphore:
            #         return index, await self.generate_single_description_async(str(row['retro_template']), str(row['template_code']))

            for batch_start in range(0, len(templates_to_process), batch_size):
                batch_templates = templates_to_process[
                    batch_start : batch_start + batch_size
                ]
                # tasks = [process_template_with_semaphore(index, row) for index, row in batch_templates]
                # batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                batch_results = await self.generate_batch_descriptions_async(
                    batch_templates
                )
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        continue
                    index, res = result
                    result_df.at[index, "llm_description"] = res["description"]
                    result_df.at[index, "llm_analysis"] = res["analysis"]
                    result_df.at[index, "full_response"] = res["full_response"]
                    result_df.at[index, "description_metadata"] = json.dumps(
                        {
                            "success": res["success"],
                            "model_used": res["model_used"],
                            "tokens_used": res["tokens_used"],
                            "prompt_tokens": res.get("prompt_tokens"),
                            "completion_tokens": res.get("completion_tokens"),
                            "timestamp": str(res["timestamp"]),
                            "error": res.get("error"),
                        }
                    )
                if save_progress:
                    result_df.to_csv(output_path, index=False)
            result_df.to_csv(output_path, index=False)
            return result_df

        def _retrieve_batch_results(
            batch_id: str, time_limit: int, client: Optional = None
        ):
            """Retrieve results for a batch submission from the anthropic API. time_limit is in seconds."""
            if client is None:
                client = anthropic.Anthropic()

            cur_time = 0
            while True:
                batch_status = client.messages.batches.retrieve(batch_id)
                if batch_status.processing_status == "ended":
                    print("Batch ended.", batch_status.processing_status)
                    break
                print(
                    f"Batch {batch_id} still processing. Status {batch_status.processing_status}. Waiting 60s..."
                )
                time.sleep(60)
                cur_time += 60
                if cur_time > time_limit:
                    raise TimeoutError(
                        f"Batch {batch_id} processing timed out after {time_limit} seconds."
                    )

            results = []
            for result in client.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    print("Succeeded for custom_id:", result.custom_id)
                else:
                    print("Error for custom_id:", result.custom_id)
                    print("Result type:", result.result.type)

                results.append(result)

            return results

        async def process_rows_batch_submission(
            self,
            templates_df: pd.DataFrame,
            output_path: str = "templates_with_descriptions.csv",
            batch_size: int = 50,
            start_index: int = 0,
            save_progress: bool = True,
        ) -> None:
            logger.info(f" Starting ASYNC batch description submission to anthropic")
            result_df, templates_to_process = _get_rows_to_process(
                self, templates_df, start_index
            )

            batch_id_file = os.path.join(os.path.dirname(output_path), "batch_id.txt")

            if not os.path.exists(batch_id_file):
                client = anthropic.Anthropic()

                requests = []

                for index, row in templates_to_process:
                    smarts_reaction = str(row[self._template_col])
                    prompt = self.prompt_template.replace(
                        "{{SMARTS_REACTION}}", smarts_reaction
                    )
                    requests.append(
                        Request(
                            custom_id=str(row["template_code"]),
                            params=MessageCreateParamsNonStreaming(
                                model=self.model,
                                max_tokens=2000,
                                temperature=0.1,
                                messages=[{"role": "user", "content": prompt}],
                            ),
                        )
                    )

                message_batch = client.messages.batches.create(
                    requests=requests,
                )
                message_id = message_batch.id
                logger.info(
                    f"Batch submitted with id {message_id}. Number of requests: {len(requests)}"
                )

                with open(batch_id_file, "w") as f:
                    f.write(message_id)

            else:
                with open(batch_id_file, "r") as f:
                    message_id = f.read().strip()
                logger.info(f"Resuming batch submission with id {message_id}")
                client = anthropic.Anthropic()

            results = _retrieve_batch_results(
                message_id, time_limit=86400, client=client
            )

            res = []
            for (index, row), result in zip(templates_to_process, results):
                if result.result.type != "succeeded":
                    continue
                full_response = result.result.message.content[0].text
                parsed = self.parse_llm_response(full_response)

                token_usage = result.result.message.usage

                res.append(
                    (
                        index,
                        {
                            "success": True,
                            "description": parsed["description"],
                            "analysis": parsed["analysis"],
                            "full_response": full_response,
                            "model_used": self.model,
                            "timestamp": pd.Timestamp.now(),
                        },
                    )
                )

                result_df.at[index, "llm_description"] = parsed["description"]
                result_df.at[index, "llm_analysis"] = parsed["analysis"]
                result_df.at[index, "full_response"] = full_response
                result_df.at[index, "description_metadata"] = json.dumps(
                    {
                        "success": True,
                        "model_used": self.model,
                        "batch_message_id": message_id,
                        "tokens_used": token_usage.input_tokens
                        + token_usage.output_tokens,
                        "prompt_tokens": token_usage.input_tokens,
                        "completion_tokens": token_usage.output_tokens,
                        "timestamp": str(pd.Timestamp.now()),
                    }
                )

            result_df.to_csv(output_path, index=False)
            return result_df

        if batch_submission:
            return process_rows_batch_submission
        else:
            return process_rows_in_batch_async

    # def _is_success_row(self, df, index):
    #     if pd.isna(df.at[index, 'llm_description']):
    #         return False
    #     if pd.isna(df.at[index, 'description_metadata']):
    #         return False
    #     description_metadata = json.loads(df.at[index, 'description_metadata'])
    #     return description_metadata['success']
    def _is_success_row(self, row):
        """Check if a row indicates a successful description generation."""
        if "llm_description" not in row or "description_metadata" not in row:
            return False
        if pd.isna(row["llm_description"]):
            return False
        if pd.isna(row["description_metadata"]):
            return False
        if row["llm_description"].startswith("Error:"):
            return False
        try:
            description_metadata = json.loads(row["description_metadata"])
            return description_metadata.get("success", False)
        except json.JSONDecodeError:
            return False

    def generate_batch_descriptions(
        self,
        templates_df: pd.DataFrame,
        output_path: str = "templates_with_descriptions.csv",
        batch_size: int = 50,
        start_index: int = 0,
        save_progress: bool = True,
    ) -> pd.DataFrame:
        return asyncio.run(
            self.get_description_process_fn(self.batch_submission)(
                self, templates_df, output_path, batch_size, start_index, save_progress
            )
        )


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Generate descriptions for retrosynthetic templates using LLM."
    )
    arg_parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with templates.",
    )
    arg_parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file for descriptions.",
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use for description generation.",
    )
    arg_parser.add_argument(
        "-retro",
        action="store_true",
        help="Use retrosynthetic templates (default is forward templates).",
    )
    arg_parser.add_argument(
        "--log_frequency",
        type=float,
        default=0.001,
        help="Frequency of logging progress.",
    )
    arg_parser.add_argument(
        "-batch_submission",
        action="store_true",
        help="Use batch submission for description generation.",
    )

    return arg_parser.parse_args()


def main(args):
    INPUT_CSV = args.input_csv
    OUTPUT_CSV = args.output_csv
    BATCH_SUBMISSION = args.batch_submission
    BATCH_SIZE = 25
    MODEL = "claude-3-5-sonnet-20241022"
    MAX_CONCURRENT = 5
    if not Path(INPUT_CSV).exists():
        logger.error(f"Input file not found: {INPUT_CSV}")
        return
    generator = TemplateDescriptionGenerator(
        model=MODEL,
        max_concurrent=MAX_CONCURRENT,
        retro=args.retro,
        log_frequency=args.log_frequency,
        batch_submission=BATCH_SUBMISSION,
    )

    if Path(OUTPUT_CSV).exists():
        result_df = generator.resume_generation(OUTPUT_CSV, batch_size=BATCH_SIZE)
    else:
        if INPUT_CSV.endswith(".csv.gz"):  # raw file from azf
            templates_df = pd.read_csv(INPUT_CSV, compression="gzip", sep="\t")
        else:
            templates_df = pd.read_csv(INPUT_CSV)
        result_df = generator.generate_batch_descriptions(
            templates_df, output_path=OUTPUT_CSV, batch_size=BATCH_SIZE
        )


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set!")
    else:
        args = parse_args()
        main(args)
