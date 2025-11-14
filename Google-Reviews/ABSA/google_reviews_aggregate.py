"""Aggregate Google Anytime Fitness reviews into a single dataset for ABSA."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def collect_review_files(base_dir: Path) -> List[Path]:
    """Return a sorted list of review CSV files under the All directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Review directory not found: {base_dir}")
    files = sorted(base_dir.glob("*_reviews.csv"))
    if not files:
        raise FileNotFoundError(f"No review CSV files found in {base_dir}")
    return files


def load_and_union(files: List[Path]) -> pd.DataFrame:
    """Load individual outlet review CSVs and concatenate into one DataFrame."""
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.copy()
        df["source_file"] = path.name
        if "outlet" not in df.columns:
            outlet_name = path.stem.replace("_reviews", "")
            df["outlet"] = outlet_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns and basic cleaning expected by ABSA pipeline."""
    expected_columns = [
        "outlet",
        "author",
        "rating",
        "text",
        "date_posted",
        "review_id",
        "source_file",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df["outlet"] = df["outlet"].fillna("Unknown Outlet").astype(str).str.strip()
    df["author"] = df["author"].fillna("Unknown Author").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)

    # Ensure rating numeric where possible
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Normalise date format (prefer normalized column when available)
    normalized_available = "date_posted_normalized" in df.columns
    df["date_posted_raw"] = df["date_posted_normalized"] if normalized_available else df["date_posted"]
    original_dates = pd.to_datetime(df["date_posted"], errors="coerce")
    if normalized_available:
        normalized_dates = pd.to_datetime(df["date_posted_normalized"], errors="coerce")
        df["date_posted"] = normalized_dates.fillna(original_dates)
    else:
        df["date_posted"] = original_dates

    # Remove exact duplicate reviews based on review_id or text fallback
    if df["review_id"].notna().any():
        df = df.sort_values("review_id").drop_duplicates(subset="review_id", keep="first")
    else:
        df = df.drop_duplicates(subset=["outlet", "author", "text"], keep="first")

    df = df.reset_index(drop=True)
    return df[expected_columns + ["date_posted_raw"]]


def aggregate_reviews(base_dir: Path, output_file: Path) -> Path:
    files = collect_review_files(base_dir)
    combined = load_and_union(files)
    cleaned = clean_reviews(combined)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_file, index=False, encoding="utf-8")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Google Anytime Fitness reviews")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("Google-Reviews") / "Reviews" / "All",
        help="Directory containing per-outlet review CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Google-Reviews") / "ABSA" / "google_reviews_all_outlets.csv",
        help="Path to save the aggregated CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = aggregate_reviews(args.base_dir, args.output)
    print(f"Aggregated reviews saved to: {output_path}")
    print(f"Total reviews: {sum(1 for _ in open(output_path, encoding='utf-8')) - 1}")


if __name__ == "__main__":
    main()
