"""Aspect-Based Sentiment Analysis for Google Anytime Fitness Reviews."""
from __future__ import annotations

import re
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


ASPECT_KEYWORDS: Dict[str, Sequence[str]] = {
    "Staff/Service": [
        "staff",
        "service",
        "trainer",
        "coach",
        "manager",
        "reception",
        "front desk",
        "friendly",
        "helpful",
        "support",
        "aid",
        "instructor",
        "team",
        "crew",
        "personnel",
        "customer service",
        "rude",
        "unhelpful",
    ],
    "Cleanliness": [
        "clean",
        "dirty",
        "hygiene",
        "sanitation",
        "sanitize",
        "towel",
        "toilet",
        "bathroom",
        "locker",
        "smell",
        "odor",
        "wipe",
        "disinfect",
        "tidy",
        "messy",
        "filthy",
        "spotless",
        "dusty",
        "pristine",
    ],
    "Equipment": [
        "equipment",
        "machine",
        "weights",
        "dumbbell",
        "barbell",
        "treadmill",
        "bench",
        "rack",
        "cable",
        "plate",
        "broken",
        "functional",
        "bike",
        "elliptical",
        "rower",
        "smith machine",
        "squat rack",
        "free weights",
        "cardio",
        "kettlebell",
    ],
    "Facilities": [
        "facility",
        "space",
        "room",
        "area",
        "floor",
        "parking",
        "locker room",
        "changing room",
        "layout",
        "aircon",
        "ventilation",
        "lighting",
        "ambience",
        "environment",
        "spacious",
        "cramped",
        "size",
    ],
    "Price/Fees": [
        "price",
        "cost",
        "fee",
        "expensive",
        "cheap",
        "membership",
        "subscription",
        "rate",
        "charge",
        "value",
        "worth",
        "money",
        "billing",
        "refund",
        "pricey",
        "discount",
        "offer",
    ],
    "Policies/Contract": [
        "policy",
        "contract",
        "terms",
        "condition",
        "rule",
        "agreement",
        "cancel",
        "freeze",
        "commitment",
        "clause",
        "terminate",
        "termination",
        "entry",
        "access",
        "membership terms",
        "protocol",
    ],
    "Crowdedness": [
        "crowd",
        "busy",
        "packed",
        "queue",
        "wait",
        "peak",
        "rush hour",
        "capacity",
        "overcrowded",
        "quiet",
        "empty",
        "crowded",
    ],
    "Location/Accessibility": [
        "location",
        "accessible",
        "convenient",
        "near",
        "far",
        "distance",
        "mrt",
        "bus",
        "central",
        "nearby",
        "close",
        "transport",
        "travel",
        "journey",
        "walk",
        "parking",
    ],
    "Classes/Programs": [
        "class",
        "programme",
        "program",
        "workshop",
        "muay thai",
        "yoga",
        "spin",
        "session",
        "trainer-led",
        "coaching",
        "pt",
        "personal training",
    ],
    "Overall Experience": [
        "experience",
        "overall",
        "impression",
        "feel",
        "vibe",
        "environment",
        "recommend",
        "review",
        "satisfaction",
        "happy",
        "unhappy",
        "love",
        "hate",
        "enjoy",
        "disappointed",
    ],
}


LOCATION_KEYWORDS = [
    "ang mo kio",
    "bedok",
    "bishan",
    "bugis",
    "bukit timah",
    "chinatown",
    "clementi",
    "dhoby ghaut",
    "harbourfront",
    "hougang",
    "jurong",
    "kovan",
    "macpherson",
    "novena",
    "orchard",
    "pasir ris",
    "paya lebar",
    "serangoon",
    "tampines",
    "toa payoh",
    "woodlands",
    "yishun",
]


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model en_core_web_sm...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()


def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    cleaned = re.sub(r"http\S+|www\S+", "", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def detect_aspects(sentences_lower: Sequence[str]) -> List[str]:
    aspects = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        pattern_cache = [re.compile(rf"\b{re.escape(keyword)}\b") for keyword in keywords]
        if any(pattern.search(sentence) for sentence in sentences_lower for pattern in pattern_cache):
            aspects.append(aspect)
    return aspects


def aspect_sentiment(sentences: Sequence[str], sentences_lower: Sequence[str], aspect: str) -> float | None:
    keywords = ASPECT_KEYWORDS[aspect]
    patterns = [re.compile(rf"\b{re.escape(keyword)}\b") for keyword in keywords]
    matched_sentences = [
        sentence
        for sentence, sentence_lower in zip(sentences, sentences_lower)
        if any(pattern.search(sentence_lower) for pattern in patterns)
    ]
    if not matched_sentences:
        return None
    scores = [vader.polarity_scores(sentence)["compound"] for sentence in matched_sentences]
    return float(np.mean(scores)) if scores else None


def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def sentiment_strength(score: float) -> str:
    if score >= 0.5:
        return "Very Positive"
    if score >= 0.05:
        return "Positive"
    if score > -0.05:
        return "Neutral"
    if score > -0.5:
        return "Negative"
    return "Very Negative"


def extract_locations(text: str, outlet_name: str) -> List[str]:
    text_lower = text.lower() + " " + outlet_name.lower()
    matches = {loc.title() for loc in LOCATION_KEYWORDS if loc in text_lower}
    doc = nlp(text)
    matches.update(ent.text for ent in doc.ents if ent.label_ == "GPE")
    return sorted(matches)


@dataclass
class ReviewResult:
    outlet: str
    review_id: str | int
    rating: float | None
    date_posted: str | pd.Timestamp | None
    text: str
    aspects: List[str]
    overall_compound: float
    overall_sentiment: str
    overall_strength: str
    locations: List[str]
    aspect_scores: Dict[str, float]


def analyze_reviews(df: pd.DataFrame) -> List[ReviewResult]:
    results: List[ReviewResult] = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        text_raw = preprocess_text(row.get("text", ""))
        if not text_raw or len(text_raw) < 5:
            continue
        sentences = tokenize_sentences(text_raw)
        if not sentences:
            continue
        sentences_lower = [s.lower() for s in sentences]
        aspects = detect_aspects(sentences_lower)
        if not aspects:
            continue

        overall = vader.polarity_scores(text_raw)["compound"]
        aspect_scores: Dict[str, float] = {}
        for aspect in aspects:
            score = aspect_sentiment(sentences, sentences_lower, aspect)
            if score is not None:
                aspect_scores[aspect] = score

        if not aspect_scores:
            continue

        locations = extract_locations(text_raw, row.get("outlet", ""))

        result = ReviewResult(
            outlet=row.get("outlet", "Unknown"),
            review_id=row.get("review_id") if not pd.isna(row.get("review_id")) else row.name,
            rating=float(row.get("rating")) if not pd.isna(row.get("rating")) else None,
            date_posted=row.get("date_posted_raw") or row.get("date_posted"),
            text=text_raw,
            aspects=list(aspect_scores.keys()),
            overall_compound=overall,
            overall_sentiment=sentiment_label(overall),
            overall_strength=sentiment_strength(overall),
            locations=locations,
            aspect_scores=aspect_scores,
        )
        results.append(result)
        if idx % 1000 == 0:
            print(f"Processed {idx}/{total} reviews ({idx/total:.1%})")
    return results


def build_detailed_dataframe(results: Iterable[ReviewResult]) -> pd.DataFrame:
    records = []
    for res in results:
        record = {
            "Review_ID": res.review_id,
            "Outlet": res.outlet,
            "Rating": res.rating,
            "Date_Posted": res.date_posted,
            "Overall_Compound": res.overall_compound,
            "Overall_Sentiment": res.overall_sentiment,
            "Overall_Strength": res.overall_strength,
            "Review_Text": res.text,
            "Aspects_Mentioned": ", ".join(res.aspects),
            "Locations_Mentioned": ", ".join(res.locations) if res.locations else "None",
            "Num_Aspects": len(res.aspects),
        }
        for aspect, score in res.aspect_scores.items():
            record[f"{aspect}_Score"] = round(score, 4)
            record[f"{aspect}_Sentiment"] = sentiment_label(score)
            record[f"{aspect}_Strength"] = sentiment_strength(score)
        records.append(record)
    return pd.DataFrame(records)


def summarize_aspects(results: Iterable[ReviewResult]) -> pd.DataFrame:
    aspect_data: Dict[str, List[float]] = defaultdict(list)
    for res in results:
        for aspect, score in res.aspect_scores.items():
            aspect_data[aspect].append(score)
    rows = []
    for aspect, scores in aspect_data.items():
        positives = sum(1 for s in scores if s >= 0.05)
        negatives = sum(1 for s in scores if s <= -0.05)
        neutrals = sum(1 for s in scores if -0.05 < s < 0.05)
        rows.append(
            {
                "Aspect": aspect,
                "Mention_Count": len(scores),
                "Avg_Sentiment": float(np.mean(scores)),
                "Std_Dev": float(np.std(scores)),
                "Min_Score": float(np.min(scores)),
                "Max_Score": float(np.max(scores)),
                "Median_Score": float(np.median(scores)),
                "Positive_Count": positives,
                "Negative_Count": negatives,
                "Neutral_Count": neutrals,
                "Positive_Pct": round(positives / len(scores) * 100, 2),
                "Negative_Pct": round(negatives / len(scores) * 100, 2),
                "Neutral_Pct": round(neutrals / len(scores) * 100, 2),
            }
        )
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("Mention_Count", ascending=False)
    return summary_df


def summarize_by_outlet(results: Iterable[ReviewResult]) -> pd.DataFrame:
    outlet_aspect: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for res in results:
        for aspect, score in res.aspect_scores.items():
            outlet_aspect[res.outlet][aspect].append(score)
    rows = []
    for outlet, aspects in outlet_aspect.items():
        for aspect, scores in aspects.items():
            rows.append(
                {
                    "Outlet": outlet,
                    "Aspect": aspect,
                    "Mention_Count": len(scores),
                    "Avg_Sentiment": float(np.mean(scores)),
                    "Positive_Pct": round(sum(1 for s in scores if s >= 0.05) / len(scores) * 100, 2),
                    "Negative_Pct": round(sum(1 for s in scores if s <= -0.05) / len(scores) * 100, 2),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_location(results: Iterable[ReviewResult]) -> pd.DataFrame:
    location_aspect: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for res in results:
        for location in res.locations:
            for aspect, score in res.aspect_scores.items():
                location_aspect[location][aspect].append(score)
    rows = []
    for location, aspects in location_aspect.items():
        for aspect, scores in aspects.items():
            if len(scores) < 3:
                continue
            rows.append(
                {
                    "Location": location,
                    "Aspect": aspect,
                    "Mention_Count": len(scores),
                    "Avg_Sentiment": float(np.mean(scores)),
                    "Positive_Pct": round(sum(1 for s in scores if s >= 0.05) / len(scores) * 100, 2),
                    "Negative_Pct": round(sum(1 for s in scores if s <= -0.05) / len(scores) * 100, 2),
                }
            )
    return pd.DataFrame(rows)


def create_visualizations(summary_df: pd.DataFrame, aspect_scores: Dict[str, List[float]], prefix: Path) -> None:
    if summary_df.empty:
        return
    top_aspects = summary_df.head(10)

    plt.figure(figsize=(12, 6))
    plt.barh(top_aspects["Aspect"], top_aspects["Mention_Count"], color="steelblue")
    plt.xlabel("Number of Mentions", fontsize=12, fontweight="bold")
    plt.ylabel("Aspect", fontsize=12, fontweight="bold")
    plt.title("Top 10 Most Discussed Aspects", fontsize=14, fontweight="bold", pad=20)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_mention_frequency.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    colors = [
        "#2ecc71" if score >= 0.05 else "#e74c3c" if score <= -0.05 else "#95a5a6"
        for score in top_aspects["Avg_Sentiment"]
    ]
    plt.barh(top_aspects["Aspect"], top_aspects["Avg_Sentiment"], color=colors)
    plt.xlabel("Average Sentiment (VADER)", fontsize=12, fontweight="bold")
    plt.ylabel("Aspect", fontsize=12, fontweight="bold")
    plt.axvline(x=0, color="black", linewidth=1)
    plt.axvline(x=0.05, color="green", linestyle="--", linewidth=0.8, alpha=0.4)
    plt.axvline(x=-0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
    plt.title("Average Sentiment Score by Aspect", fontsize=14, fontweight="bold", pad=20)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_avg_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 8))
    x = np.arange(len(top_aspects))
    width = 0.75
    plt.bar(x, top_aspects["Positive_Pct"], width, label="Positive", color="#2ecc71")
    plt.bar(x, top_aspects["Neutral_Pct"], width, bottom=top_aspects["Positive_Pct"], label="Neutral", color="#95a5a6")
    bottom_vals = top_aspects["Positive_Pct"] + top_aspects["Neutral_Pct"]
    plt.bar(x, top_aspects["Negative_Pct"], width, bottom=bottom_vals, label="Negative", color="#e74c3c")
    plt.xticks(x, top_aspects["Aspect"], rotation=45, ha="right")
    plt.xlabel("Aspect", fontsize=12, fontweight="bold")
    plt.ylabel("Sentiment Distribution (%)", fontsize=12, fontweight="bold")
    plt.title("Sentiment Distribution by Aspect", fontsize=14, fontweight="bold", pad=20)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_sentiment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 8))
    data_to_plot = [aspect_scores[aspect] for aspect in top_aspects["Aspect"] if aspect in aspect_scores]
    labels = [aspect for aspect in top_aspects["Aspect"] if aspect in aspect_scores]
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("darkblue")
            median.set_linewidth(2)
        for mean in bp["means"]:
            mean.set_color("red")
            mean.set_linewidth(2)
        ax.set_xlabel("Aspect", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sentiment Score", fontsize=12, fontweight="bold")
        ax.set_title("Sentiment Score Distribution by Aspect", fontsize=14, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.axhline(y=0, color="black", linewidth=1)
        plt.axhline(y=0.05, color="green", linestyle="--", linewidth=0.8, alpha=0.4)
        plt.axhline(y=-0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{prefix}_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()


def create_smoothed_temporal_visualizations(
    recent_df: pd.DataFrame, prefix: Path, recent_start_year: int, window: int = 3
) -> None:
    if recent_df.empty:
        return
    colors = {
        "Facilities": "#e74c3c",
        "Price/Fees": "#2ecc71",
        "Policies/Contract": "#f39c12",
        "Equipment": "#3498db",
        "Crowdedness": "#9b59b6",
        "Staff/Service": "#1abc9c",
        "Cleanliness": "#e67e22",
        "Location/Accessibility": "#34495e",
        "Overall Experience": "#95a5a6",
        "Classes/Programs": "#8e44ad",
    }
    temp = recent_df.copy()
    temp["Period"] = pd.PeriodIndex(temp["Period"], freq="M")
    temp = temp.sort_values(["Period", "Aspect"])
    pivot = temp.pivot(index="Period", columns="Aspect", values="Avg_Sentiment").sort_index()
    pivot_index = pivot.index.to_timestamp()
    smoothed = pivot.rolling(window=window, min_periods=1).mean()
    smoothed.index = pivot_index
    top_aspects = (
        recent_df.groupby("Aspect")["Mention_Count"].sum().sort_values(ascending=False).head(5).index
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    for aspect in top_aspects:
        if aspect not in smoothed.columns:
            continue
        series = smoothed[aspect].dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series.values,
            label=f"{aspect} ({window}-month avg)",
            linewidth=2.5,
            color=colors.get(aspect, "#555555"),
        )
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.4)
    ax.axhline(y=0.05, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.axhline(y=-0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Time Period (Month)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Sentiment", fontsize=12, fontweight="bold")
    ax.set_title(f"{window}-Month Rolling Sentiment Trends ({recent_start_year}+)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{prefix}_temporal_recent_smoothed.png", dpi=300, bbox_inches="tight")
    plt.close()

    smoothed_diff = smoothed.diff()
    fig, ax = plt.subplots(figsize=(14, 8))
    for aspect in top_aspects:
        if aspect not in smoothed_diff.columns:
            continue
        series = smoothed_diff[aspect].dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series.values,
            label=f"{aspect} ({window}-month avg)",
            linewidth=2.0,
            color=colors.get(aspect, "#555555"),
        )
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.4)
    ax.set_xlabel("Time Period (Month)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rolling MoM Sentiment Change", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{window}-Month Rolling Sentiment Change ({recent_start_year}+)", fontsize=14, fontweight="bold", pad=20
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{prefix}_temporal_recent_smoothed_change.png", dpi=300, bbox_inches="tight")
    plt.close()


def temporal_analysis(results: Iterable[ReviewResult], prefix: Path, recent_start_year: int = 2022) -> None:
    temporal_rows = []
    for res in results:
        date_val = pd.to_datetime(res.date_posted, errors="coerce")
        if pd.isna(date_val):
            continue
        period = date_val.to_period("M")
        for aspect, score in res.aspect_scores.items():
            temporal_rows.append(
                {
                    "Period": str(period),
                    "Aspect": aspect,
                    "Score": score,
                }
            )
    temporal_df = pd.DataFrame(temporal_rows)
    if temporal_df.empty:
        return
    grouped = (
        temporal_df.groupby(["Period", "Aspect"])["Score"].agg(
            Mention_Count="size",
            Avg_Sentiment="mean",
            Positive_Pct=lambda s: round(sum(val >= 0.05 for val in s) / len(s) * 100, 2),
            Negative_Pct=lambda s: round(sum(val <= -0.05 for val in s) / len(s) * 100, 2),
            Neutral_Pct=lambda s: round(sum(-0.05 < val < 0.05 for val in s) / len(s) * 100, 2),
        )
    ).reset_index()
    grouped.to_csv(f"{prefix}_temporal.csv", index=False)

    top_aspects = (
        grouped.groupby("Aspect")["Mention_Count"].sum().sort_values(ascending=False).head(5).index
    )
    plt.figure(figsize=(14, 8))
    for aspect in top_aspects:
        aspect_data = grouped[grouped["Aspect"] == aspect].sort_values("Period")
        plt.plot(
            aspect_data["Period"],
            aspect_data["Avg_Sentiment"],
            marker="o",
            linewidth=2,
            label=aspect,
        )
    plt.xlabel("Time Period", fontsize=12, fontweight="bold")
    plt.ylabel("Average Sentiment", fontsize=12, fontweight="bold")
    plt.title("Sentiment Trends Over Time by Aspect", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=0, color="black", linewidth=1)
    plt.axhline(y=0.05, color="green", linestyle="--", linewidth=0.8, alpha=0.4)
    plt.axhline(y=-0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_temporal.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Recent temporal analysis
    recent_df = grouped[grouped["Period"].apply(lambda p: int(p.split("-")[0]) >= recent_start_year)]
    if recent_df.empty:
        return
    recent_df.to_csv(f"{prefix}_temporal_recent.csv", index=False)
    plt.figure(figsize=(14, 8))
    for aspect in top_aspects:
        aspect_data = recent_df[recent_df["Aspect"] == aspect].sort_values("Period")
        if aspect_data.empty:
            continue
        plt.plot(
            aspect_data["Period"],
            aspect_data["Avg_Sentiment"],
            marker="o",
            linewidth=2,
            label=aspect,
        )
    plt.xlabel("Period", fontsize=12, fontweight="bold")
    plt.ylabel("Average Sentiment", fontsize=12, fontweight="bold")
    plt.title(f"Sentiment Trends Since {recent_start_year}", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=0, color="black", linewidth=1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_temporal_recent.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 8))
    for aspect in top_aspects:
        aspect_data = recent_df[recent_df["Aspect"] == aspect].sort_values("Period")
        if len(aspect_data) < 2:
            continue
        change = aspect_data["Avg_Sentiment"].diff()
        plt.plot(
            aspect_data["Period"].iloc[1:],
            change.iloc[1:],
            marker="o",
            linewidth=2,
            label=aspect,
        )
    plt.xlabel("Period", fontsize=12, fontweight="bold")
    plt.ylabel("MoM Sentiment Change", fontsize=12, fontweight="bold")
    plt.title("Month-over-Month Sentiment Change (Recent)", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=0, color="black", linewidth=1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_temporal_recent_change.png", dpi=300, bbox_inches="tight")
    plt.close()

    create_smoothed_temporal_visualizations(recent_df, prefix, recent_start_year)


def save_outputs(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outlet_df: pd.DataFrame,
    location_df: pd.DataFrame,
    base_path: Path,
    prefix: str,
) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(base_path / f"{prefix}_detailed.csv", index=False, encoding="utf-8")
    summary_df.to_csv(base_path / f"{prefix}_summary.csv", index=False, encoding="utf-8")
    outlet_df.to_csv(base_path / f"{prefix}_by_outlet.csv", index=False, encoding="utf-8")
    if not location_df.empty:
        location_df.to_csv(base_path / f"{prefix}_by_location.csv", index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aspect-Based Sentiment Analysis for Google reviews.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Google-Reviews/ABSA/google_reviews_all_outlets.csv"),
        help="Path to aggregated reviews CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Google-Reviews/ABSA"),
        help="Directory to store generated outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="google_af_absa",
        help="Prefix to use for output filenames.",
    )
    parser.add_argument(
        "--recent-start-year",
        type=int,
        default=2022,
        help="Starting year for recent temporal analysis.",
    )
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip generating temporal analysis outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir
    output_prefix = args.output_prefix

    if not input_path.exists():
        raise FileNotFoundError(
            f"Aggregated reviews not found at {input_path}. Run google_reviews_aggregate.py first."
        )

    print("Loading aggregated Google reviews...")
    reviews_df = pd.read_csv(input_path)
    print(f"Loaded {len(reviews_df)} reviews")

    results = analyze_reviews(reviews_df)
    if not results:
        print("No reviews with identifiable aspects were found. Exiting.")
        return

    detailed_df = build_detailed_dataframe(results)
    summary_df = summarize_aspects(results)
    outlet_df = summarize_by_outlet(results)
    location_df = summarize_by_location(results)

    save_outputs(detailed_df, summary_df, outlet_df, location_df, output_dir, output_prefix)

    aspect_scores = {
        aspect: [score for res in results for aspect_res, score in res.aspect_scores.items() if aspect_res == aspect]
        for aspect in summary_df["Aspect"].tolist()
    }

    prefix_path = output_dir / output_prefix
    create_visualizations(summary_df, aspect_scores, prefix_path)
    if not args.skip_temporal:
        temporal_analysis(results, prefix_path, recent_start_year=args.recent_start_year)

    print(f"\nAnalysis complete! Generated files saved in {output_dir}")


if __name__ == "__main__":
    main()
