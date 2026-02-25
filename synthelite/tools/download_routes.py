"""Download precomputed synthesis routes from HuggingFace.

Routes are organized by experiment type and model:
    routes/strategic/{gemini2_5, claude4_5, gpt5}
    routes/starting_materials/{gemini2_5, claude4_5, gpt5}
    routes/uspto_190/{gemini2_5.zip, claude4_5.zip, gpt5.zip}
"""
import argparse
import os
import zipfile

from huggingface_hub import snapshot_download

HF_REPO_ID = "SchwallerGroup/synthelite"
ROUTE_PREFIX = "routes"

EXPERIMENTS = ["strategic", "starting_materials", "uspto_190"]
MODELS = ["gemini2_5", "claude4_5", "gpt5"]


def download_routes(
    output_dir: str = "data/routes",
    experiments: list[str] | None = None,
    models: list[str] | None = None,
    unzip: bool = True,
) -> None:
    """Download routes from HuggingFace and optionally unzip archives.

    Args:
        output_dir: Local directory to save routes into.
        experiments: Which experiment sets to download (default: all).
        models: Which model results to download (default: all).
        unzip: Whether to extract .zip files after downloading.
    """
    experiments = experiments or EXPERIMENTS
    models = models or MODELS

    allow_patterns = []
    for exp in experiments:
        for model in models:
            allow_patterns.append(f"{ROUTE_PREFIX}/{exp}/{model}/**")
            allow_patterns.append(f"{ROUTE_PREFIX}/{exp}/{model}.zip")

    print(f"Downloading routes to {os.path.abspath(output_dir)} ...")
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=output_dir,
    )

    if unzip:
        _unzip_all(os.path.join(local_dir, ROUTE_PREFIX))

    print(f"Done. Routes saved to {os.path.join(os.path.abspath(output_dir), ROUTE_PREFIX)}")


def _unzip_all(routes_dir: str) -> None:
    """Find and extract all .zip files under routes_dir."""
    for root, _, files in os.walk(routes_dir):
        for fname in files:
            if fname.endswith(".zip"):
                zip_path = os.path.join(root, fname)
                extract_to = os.path.join(root, os.path.splitext(fname)[0])
                if os.path.isdir(extract_to):
                    print(f"  Already extracted: {extract_to}, skipping.")
                    continue
                print(f"  Extracting {zip_path} -> {extract_to}")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_to)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download precomputed synthesis routes from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        default="data/routes",
        help="Directory to save routes (default: data/routes)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=None,
        help="Experiment sets to download (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=None,
        help="Model results to download (default: all)",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not extract .zip files after downloading",
    )
    args = parser.parse_args()
    download_routes(
        output_dir=args.output_dir,
        experiments=args.experiments,
        models=args.models,
        unzip=not args.no_unzip,
    )


if __name__ == "__main__":
    main()
