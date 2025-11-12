"""
Aggregate all scraped review CSVs, run VADER sentiment analysis, and produce overview visualizations.
Saves outputs to Reviews/Overview/ as PNGs and prints a summary table.

Run:
    python analysis_sentiment_overview.py

"""
import os
import glob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

BASE_DIR = os.getcwd()
REVIEWS_DIR = os.path.join(BASE_DIR, 'Reviews', 'All')
OUT_DIR = os.path.join(BASE_DIR, 'Reviews', 'Overview')
os.makedirs(OUT_DIR, exist_ok=True)

# Find all review CSVs
csvs = glob.glob(os.path.join(REVIEWS_DIR, '*_reviews.csv'))
if not csvs:
    print('No review CSVs found in Reviews/All/. Make sure you ran the scraper and files are present.')
    raise SystemExit(1)

print(f'Found {len(csvs)} review CSV files. Loading...')

dfs = []
for p in csvs:
    try:
        df = pd.read_csv(p)
        df['source_file'] = os.path.basename(p)
        dfs.append(df)
    except Exception as e:
        print(f'Warning: failed to read {p}: {e}')

all_reviews = pd.concat(dfs, ignore_index=True)
print(f'Combined reviews: {len(all_reviews):,} rows')

# Basic cleaning
if 'text' not in all_reviews.columns:
    print("No 'text' column found in the CSVs. Available columns:", all_reviews.columns.tolist())
    raise SystemExit(1)

all_reviews['text'] = all_reviews['text'].fillna('').astype(str)

# Compute VADER sentiment
analyzer = SentimentIntensityAnalyzer()

def vader_category(text):
    s = analyzer.polarity_scores(text)
    compound = s['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

print('Scoring sentiments (VADER)...')
all_reviews['vader_compound'] = all_reviews['text'].apply(lambda t: analyzer.polarity_scores(t)['compound'])

def vader_label(c):
    if c >= 0.05:
        return 'positive'
    if c <= -0.05:
        return 'negative'
    return 'neutral'

all_reviews['vader_cat'] = all_reviews['vader_compound'].apply(vader_label)

# Quick stats
counts = all_reviews['vader_cat'].value_counts().reindex(['positive','neutral','negative']).fillna(0).astype(int)
perc = (counts / counts.sum() * 100).round(2)
summary = pd.DataFrame({'count': counts, 'percent': perc})
print('\nOverall sentiment summary:')
print(summary)

# Plot: bar chart
plt.figure(figsize=(7,5))
sns.barplot(x=summary.index, y='count', data=summary.reset_index().rename(columns={'index':'sentiment'}), palette=['green','grey','red'])
plt.title('Sentiment Overview (VADER)')
plt.ylabel('Number of reviews')
plt.xlabel('Sentiment')
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, 'sentiment_overview_bar.png')
plt.savefig(bar_path)
plt.close()

# Plot: pie chart
plt.figure(figsize=(6,6))
colors = ['#2ecc71', '#95a5a6', '#e74c3c']
plt.pie(summary['count'], labels=summary.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Sentiment Distribution (VADER)')
pie_path = os.path.join(OUT_DIR, 'sentiment_overview_pie.png')
plt.savefig(pie_path)
plt.close()

print(f"\nSaved bar chart to: {bar_path}")
print(f"Saved pie chart to: {pie_path}")

# Optional: top words per sentiment (very simple)
from collections import Counter
import re

print('\nExtracting top words per sentiment (simple tokenization)...')
all_reviews['words'] = all_reviews['text'].str.lower().apply(lambda t: re.findall(r"\b[a-z]{2,}\b", t))

top_words = {}
for cat in ['positive','neutral','negative']:
    words = all_reviews.loc[all_reviews['vader_cat']==cat, 'words'].explode().dropna()
    c = Counter(words)
    top_words[cat] = c.most_common(15)
    print(f"\nTop words for {cat}: {top_words[cat][:10]}")

print('\nDone.')

# -----------------------------
# Per-outlet sentiment breakdown
# -----------------------------
print('\nComputing per-outlet sentiment summaries...')
# ensure there's an outlet column; fall back to source_file if missing
if 'outlet' not in all_reviews.columns or all_reviews['outlet'].isna().all():
    all_reviews['outlet_name'] = all_reviews['source_file'].str.replace('_reviews.csv', '', regex=False)
else:
    all_reviews['outlet_name'] = all_reviews['outlet'].fillna(all_reviews['source_file'].str.replace('_reviews.csv', '', regex=False))

group = all_reviews.groupby('outlet_name')
summary_rows = []
for outlet, g in group:
    total = len(g)
    pos = (g['vader_cat'] == 'positive').sum()
    neu = (g['vader_cat'] == 'neutral').sum()
    neg = (g['vader_cat'] == 'negative').sum()
    no_text = (g['text'].fillna('') == '').sum()
    summary_rows.append({
        'outlet': outlet,
        'total_reviews': total,
        'positive': int(pos),
        'neutral': int(neu),
        'negative': int(neg),
        'no_text': int(no_text),
        'pct_positive': round(pos / total * 100, 2) if total else 0.0,
        'pct_neutral': round(neu / total * 100, 2) if total else 0.0,
        'pct_negative': round(neg / total * 100, 2) if total else 0.0,
    })

per_outlet_df = pd.DataFrame(summary_rows).sort_values('total_reviews', ascending=False)
per_outlet_csv = os.path.join(OUT_DIR, 'per_outlet_sentiment_summary.csv')
per_outlet_df.to_csv(per_outlet_csv, index=False)
print(f"Saved per-outlet sentiment summary to: {per_outlet_csv}")

# Plot top 20 outlets by total_reviews with stacked bars
top_n = 20
top_df = per_outlet_df.head(top_n).set_index('outlet')
plt.figure(figsize=(12,8))
top_df[['positive','neutral','negative']].plot(kind='bar', stacked=True, color=['#2ecc71', '#95a5a6', '#e74c3c'])
plt.title(f'Top {top_n} Outlets by Reviews - Sentiment Breakdown')
plt.ylabel('Number of reviews')
plt.tight_layout()
stacked_path = os.path.join(OUT_DIR, 'sentiment_by_outlet_top20.png')
plt.savefig(stacked_path)
plt.close()
print(f"Saved stacked sentiment-by-outlet chart to: {stacked_path}")
