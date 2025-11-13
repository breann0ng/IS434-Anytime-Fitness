"""
Improved Temporal Analysis - Recent Years Only (2022-2025)
This version filters to recent data for clearer trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# Import from main script
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load models
print("Loading models...")
nlp = spacy.load('en_core_web_sm')
vader_analyzer = SentimentIntensityAnalyzer()

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# ASPECT KEYWORDS (Same as main script)
# ============================================================================

ASPECT_KEYWORDS = {
    'Staff/Service': [
        'staff', 'service', 'employee', 'instructor', 'trainer', 'manager', 
        'reception', 'receptionist', 'worker', 'personnel', 'crew', 'team member',
        'customer service', 'front desk', 'representative', 'coach', 'help',
        'assistance', 'support', 'friendly', 'rude', 'helpful', 'unhelpful'
    ],
    'Cleanliness': [
        'clean', 'dirty', 'hygiene', 'sanitation', 'sanitize', 'towel', 
        'shower', 'toilet', 'bathroom', 'locker', 'smell', 'odor', 'stink',
        'wipe', 'disinfect', 'maintenance', 'tidy', 'messy', 'filthy',
        'spotless', 'dusty', 'grimy', 'pristine', 'disgusting', 'gross'
    ],
    'Equipment': [
        'equipment', 'machine', 'weights', 'dumbbell', 'barbell', 'treadmill',
        'bench', 'rack', 'cable', 'plate', 'broken', 'working', 'functional',
        'bike', 'elliptical', 'smith machine', 'squat rack', 'free weights',
        'cardio machine', 'rowing machine', 'leg press', 'cable machine',
        'kettlebell', 'resistance', 'apparatus', 'device'
    ],
    'Facilities': [
        'facility', 'space', 'gym', 'room', 'area', 'floor', 'locker room',
        'changing room', 'parking', 'location', 'building', 'outlet', 'branch',
        'venue', 'premise', 'aircon', 'air conditioning', 'ventilation', 'lighting',
        'interior', 'design', 'layout', 'spacious', 'cramped', 'size',
        'ambience', 'atmosphere', 'environment'
    ],
    'Price/Fees': [
        'price', 'cost', 'fee', 'expensive', 'cheap', 'afford', 'payment',
        'membership', 'subscription', 'rate', 'charge', 'fine', 'penalty',
        'value', 'worth', 'money', 'dollar', 'billing', 'refund',
        'pricing', 'budget', 'rebill', 'deduct', 'bank', 'credit card',
        'overpriced', 'reasonable', 'pricey'
    ],
    'Policies/Contract': [
        'policy', 'contract', 'terms', 'condition', 'rule', 'agreement',
        'cancellation', 'cancel', 'freeze', 'commitment', 'binding', 'clause',
        'terminate', 'termination', 'exit', 'legal', 'obligation', 'tailgate',
        'security', 'access', 'key fob', 'entry', 'membership terms',
        'violation', 'regulation', 'protocol', 'requirement'
    ],
    'Crowdedness': [
        'crowd', 'busy', 'packed', 'full', 'empty', 'queue', 'wait',
        'peak hour', 'rush hour', 'space', 'capacity', 'occupancy',
        'traffic', 'congestion', 'available', 'free', 'crowded',
        'overcrowded', 'quiet', 'peaceful', 'hectic', 'chaos'
    ],
    'Location/Accessibility': [
        'location', 'accessible', 'convenient', 'near', 'far', 'distance',
        'transport', 'mrt', 'bus', 'parking', 'central', 'ulu',
        'easy to reach', 'hard to reach', 'proximity', 'reachable',
        'commute', 'travel', 'journey', 'nearby', 'close by'
    ],
    'Overall Experience': [
        'experience', 'overall', 'general', 'impression', 'feel', 'atmosphere',
        'vibe', 'environment', 'recommend', 'suggestion', 'opinion', 'review',
        'satisfaction', 'disappointed', 'pleased', 'happy', 'unhappy',
        'satisfied', 'dissatisfied', 'enjoy', 'love', 'hate'
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentences_with_spacy(text):
    """Use spaCy to split text into sentences"""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def extract_aspects(text):
    """Extract aspects mentioned in the text"""
    text = preprocess_text(text)
    aspects_found = []
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                aspects_found.append(aspect)
                break
    
    return aspects_found

def extract_aspect_phrases_spacy(text, aspect):
    """Use spaCy to extract contextual phrases around aspect keywords"""
    text = preprocess_text(text)
    keywords = ASPECT_KEYWORDS[aspect]
    sentences = get_sentences_with_spacy(text)
    
    relevant_sentences = []
    for sentence in sentences:
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sentence.lower()):
                relevant_sentences.append(sentence)
                break
    
    return relevant_sentences

def get_aspect_sentiment_vader(text, aspect):
    """Get sentiment for a specific aspect using VADER"""
    relevant_sentences = extract_aspect_phrases_spacy(text, aspect)
    
    if not relevant_sentences:
        return None
    
    sentiments = []
    for sentence in relevant_sentences:
        vader_scores = vader_analyzer.polarity_scores(sentence)
        sentiments.append(vader_scores['compound'])
    
    return np.mean(sentiments) if sentiments else None

# ============================================================================
# IMPROVED TEMPORAL ANALYSIS - RECENT YEARS ONLY
# ============================================================================

def analyze_temporal_trends_recent(input_file, start_year=2022, output_prefix='temporal_recent'):
    """
    Analyze temporal trends for RECENT YEARS ONLY
    Default: 2022-2025 for cleaner visualization
    """
    print("\n" + "="*80)
    print(f"TEMPORAL ANALYSIS - RECENT YEARS ({start_year} onwards)")
    print("="*80 + "\n")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    # Convert to datetime
    print("Processing dates...")
    df['Date'] = pd.to_datetime(df['Created_UTC'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Filter to recent years only
    df = df[df['Date'].notna()]
    df = df[df['Date'].dt.year >= start_year]
    
    print(f"✓ Filtered to {len(df)} comments from {start_year} onwards")
    
    if len(df) == 0:
        print(f"⚠ No data found from {start_year} onwards")
        return
    
    # Create year-month periods
    df['Year_Month'] = df['Date'].dt.to_period('M')
    
    print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"✓ Unique months: {df['Year_Month'].nunique()}\n")
    
    # Analyze temporal trends
    temporal_results = []
    
    print("Analyzing sentiment by month and aspect...")
    periods = sorted(df['Year_Month'].unique())
    
    for i, period in enumerate(periods):
        if i % 5 == 0:
            print(f"Processing: {i}/{len(periods)} months...")
        
        period_df = df[df['Year_Month'] == period]
        
        for aspect in ASPECT_KEYWORDS.keys():
            aspect_sentiments = []
            
            for _, row in period_df.iterrows():
                comment = row.get('Comment', '')
                aspects = extract_aspects(comment)
                
                if aspect in aspects:
                    sentiment = get_aspect_sentiment_vader(comment, aspect)
                    if sentiment is not None:
                        aspect_sentiments.append(sentiment)
            
            if aspect_sentiments:
                temporal_results.append({
                    'Period': str(period),
                    'Aspect': aspect,
                    'Count': len(aspect_sentiments),
                    'Avg_Sentiment': round(np.mean(aspect_sentiments), 4),
                    'Positive_Pct': round(sum(1 for s in aspect_sentiments if s >= 0.05) / len(aspect_sentiments) * 100, 2),
                    'Negative_Pct': round(sum(1 for s in aspect_sentiments if s <= -0.05) / len(aspect_sentiments) * 100, 2)
                })
    
    print(f"✓ Processed all {len(periods)} months\n")
    
    temporal_df = pd.DataFrame(temporal_results)
    
    if temporal_df.empty:
        print("⚠ No temporal data generated")
        return
    
    # Save results
    csv_file = f'{output_prefix}.csv'
    temporal_df.to_csv(csv_file, index=False)
    print(f"✓ Saved temporal data: {csv_file}")
    
    # Create improved visualization
    create_improved_temporal_visualization(temporal_df, start_year, output_prefix)
    
    # Print summary
    print_temporal_summary(temporal_df, start_year)
    
    return temporal_df

# ============================================================================
# IMPROVED VISUALIZATION
# ============================================================================

def create_improved_temporal_visualization(temporal_df, start_year, output_prefix):
    """Create cleaner temporal visualization"""
    
    print("\nCreating improved temporal visualization...")
    
    # Get top 5 most discussed aspects
    top_aspects = temporal_df.groupby('Aspect')['Count'].sum().nlargest(5).index
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Define colors
    colors = {
        'Facilities': '#e74c3c',
        'Price/Fees': '#2ecc71', 
        'Policies/Contract': '#f39c12',
        'Equipment': '#3498db',
        'Crowdedness': '#9b59b6',
        'Staff/Service': '#1abc9c',
        'Cleanliness': '#e67e22',
        'Location/Accessibility': '#34495e',
        'Overall Experience': '#95a5a6'
    }
    
    # Plot each aspect
    for aspect in top_aspects:
        aspect_data = temporal_df[temporal_df['Aspect'] == aspect].sort_values('Period')
        
        if not aspect_data.empty:
            color = colors.get(aspect, '#555555')
            ax.plot(aspect_data['Period'], 
                   aspect_data['Avg_Sentiment'], 
                   marker='o', 
                   label=aspect, 
                   linewidth=2.5,
                   markersize=7,
                   color=color,
                   alpha=0.8)
    
    # Styling
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=0.05, color='green', linestyle='--', linewidth=0.8, alpha=0.3, label='Positive threshold')
    ax.axhline(y=-0.05, color='red', linestyle='--', linewidth=0.8, alpha=0.3, label='Negative threshold')
    
    ax.set_xlabel('Time Period (Month)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Sentiment Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Sentiment Trends Over Time by Aspect ({start_year} onwards)', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    png_file = f'{output_prefix}.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {png_file}")
    plt.close()
    
    # Create additional chart: Month-over-Month change
    create_change_chart(temporal_df, top_aspects, colors, output_prefix)

def create_change_chart(temporal_df, top_aspects, colors, output_prefix):
    """Create month-over-month sentiment change chart"""
    
    print("Creating sentiment change chart...")
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    for aspect in top_aspects:
        aspect_data = temporal_df[temporal_df['Aspect'] == aspect].sort_values('Period')
        
        if len(aspect_data) > 1:
            # Calculate month-over-month change
            aspect_data = aspect_data.copy()
            aspect_data['Sentiment_Change'] = aspect_data['Avg_Sentiment'].diff()
            
            color = colors.get(aspect, '#555555')
            ax.plot(aspect_data['Period'].iloc[1:], 
                   aspect_data['Sentiment_Change'].iloc[1:], 
                   marker='o', 
                   label=aspect, 
                   linewidth=2,
                   markersize=6,
                   color=color,
                   alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Time Period (Month)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Month-over-Month Sentiment Change', fontsize=13, fontweight='bold')
    ax.set_title('Month-over-Month Sentiment Change by Aspect', 
                fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    change_file = f'{output_prefix}_change.png'
    plt.savefig(change_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved change chart: {change_file}")
    plt.close()

# ============================================================================
# SUMMARY REPORTING
# ============================================================================

def print_temporal_summary(temporal_df, start_year):
    """Print summary of temporal analysis"""
    
    print("\n" + "="*80)
    print(f"TEMPORAL ANALYSIS SUMMARY ({start_year} onwards)")
    print("="*80 + "\n")
    
    # Overall statistics
    total_periods = temporal_df['Period'].nunique()
    total_mentions = temporal_df['Count'].sum()
    
    print(f"Total months analyzed: {total_periods}")
    print(f"Total aspect mentions: {total_mentions}")
    print(f"Average mentions per month: {total_mentions / total_periods:.1f}\n")
    
    # Aspect trends
    print("─" * 80)
    print("SENTIMENT TRENDS BY ASPECT:")
    print("─" * 80)
    
    for aspect in ASPECT_KEYWORDS.keys():
        aspect_data = temporal_df[temporal_df['Aspect'] == aspect].sort_values('Period')
        
        if len(aspect_data) > 1:
            first_sentiment = aspect_data['Avg_Sentiment'].iloc[0]
            last_sentiment = aspect_data['Avg_Sentiment'].iloc[-1]
            change = last_sentiment - first_sentiment
            
            if abs(change) < 0.02:
                trend = "➡️  Stable"
            elif change > 0:
                trend = "📈 Improving"
            else:
                trend = "📉 Declining"
            
            avg_sentiment = aspect_data['Avg_Sentiment'].mean()
            
            print(f"\n{aspect}:")
            print(f"  Start: {first_sentiment:+.3f} → End: {last_sentiment:+.3f}")
            print(f"  Change: {change:+.3f} ({trend})")
            print(f"  Average: {avg_sentiment:.3f}")
            print(f"  Total mentions: {aspect_data['Count'].sum()}")
    
    print("\n" + "="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run improved temporal analysis"""
    
    input_file = 'reddit_singapore_anytimefitness_comments.csv'
    start_year = 2022  # Change this to adjust the starting year
    
    temporal_df = analyze_temporal_trends_recent(
        input_file, 
        start_year=start_year,
        output_prefix='anytime_fitness_temporal_recent'
    )
    
    print("\n" + "="*80)
    print("✓ TEMPORAL ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("1. anytime_fitness_temporal_recent.csv - Monthly sentiment data")
    print("2. anytime_fitness_temporal_recent.png - Clean temporal trends")
    print("3. anytime_fitness_temporal_recent_change.png - Month-over-month changes")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
