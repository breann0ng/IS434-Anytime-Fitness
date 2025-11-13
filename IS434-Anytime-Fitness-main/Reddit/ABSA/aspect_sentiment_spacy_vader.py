"""
Aspect-Based Sentiment Analysis for Anytime Fitness Reddit Comments
Using spaCy for NLP and VADER for sentiment analysis

Author: Potato
Course: IS434 Social Analytics & Applications
Description: Advanced ABSA implementation for analyzing customer sentiment
             across different gym aspects (facilities, price, staff, etc.)
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import spaCy and VADER
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("Loading spaCy model...")
# Load spaCy model for better text processing
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Downloading spaCy model... This may take a minute.")
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# ASPECT DEFINITIONS
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
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentences_with_spacy(text):
    """
    Use spaCy to split text into sentences (more accurate than regex)
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# ============================================================================
# ASPECT EXTRACTION FUNCTIONS
# ============================================================================

def extract_aspects(text):
    """
    Extract aspects mentioned in the text using keyword matching
    """
    text = preprocess_text(text)
    aspects_found = []
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundaries to match whole words
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                aspects_found.append(aspect)
                break  # Found this aspect, move to next
    
    return aspects_found

def extract_aspect_phrases_spacy(text, aspect):
    """
    Use spaCy to extract more contextual phrases around aspect keywords
    Returns sentences and their surrounding context
    """
    text = preprocess_text(text)
    keywords = ASPECT_KEYWORDS[aspect]
    
    # Get sentences using spaCy
    sentences = get_sentences_with_spacy(text)
    
    relevant_sentences = []
    for sentence in sentences:
        # Check if any aspect keyword is in the sentence
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sentence.lower()):
                relevant_sentences.append(sentence)
                break
    
    return relevant_sentences

# ============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================================================

def get_aspect_sentiment_vader(text, aspect):
    """
    Get sentiment for a specific aspect using VADER
    Analyzes sentences containing aspect keywords
    """
    relevant_sentences = extract_aspect_phrases_spacy(text, aspect)
    
    if not relevant_sentences:
        return None
    
    # Analyze sentiment of relevant sentences using VADER
    sentiments = []
    for sentence in relevant_sentences:
        vader_scores = vader_analyzer.polarity_scores(sentence)
        # Use compound score (normalized from -1 to 1)
        sentiments.append(vader_scores['compound'])
    
    # Return average sentiment
    return np.mean(sentiments) if sentiments else None

def categorize_sentiment(compound_score):
    """
    Categorize sentiment based on VADER compound score
    VADER threshold: positive >= 0.05, negative <= -0.05
    """
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_sentiment_details(compound_score):
    """
    Get detailed sentiment breakdown
    """
    category = categorize_sentiment(compound_score)
    
    if compound_score >= 0.5:
        strength = "Very Positive"
    elif compound_score >= 0.05:
        strength = "Positive"
    elif compound_score > -0.05:
        strength = "Neutral"
    elif compound_score > -0.5:
        strength = "Negative"
    else:
        strength = "Very Negative"
    
    return category, strength

# ============================================================================
# LOCATION EXTRACTION
# ============================================================================

def extract_locations(text):
    """
    Extract Singapore location mentions using spaCy NER and keyword matching
    """
    # Common Singapore locations
    sg_locations = [
        'bugis', 'orchard', 'jurong', 'tampines', 'bedok', 'clementi', 
        'toa payoh', 'bishan', 'novena', 'marina', 'raffles', 'newton',
        'serangoon', 'punggol', 'sengkang', 'yishun', 'woodlands', 'changi',
        'pasir ris', 'city hall', 'dhoby ghaut', 'chinatown', 'clarke quay',
        'harbourfront', 'tiong bahru', 'queenstown', 'dover', 'commonwealth',
        'buona vista', 'holland village', 'botanic gardens', 'farrer road',
        'downtown', 'telok ayer', 'tanjong pagar', 'outram', 'kallang',
        'lavender', 'rochor', 'hougang', 'kovan', 'ang mo kio', 'yio chu kang'
    ]
    
    text_lower = text.lower()
    locations_found = []
    
    for location in sg_locations:
        if location in text_lower:
            locations_found.append(location.title())
    
    # Also use spaCy NER to find locations
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'GPE':  # Geo-political entity
            locations_found.append(ent.text)
    
    return list(set(locations_found))

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def perform_aspect_based_sentiment_analysis(input_file, output_prefix='absa_results'):
    """
    Main function to perform comprehensive aspect-based sentiment analysis
    using spaCy for NLP and VADER for sentiment
    """
    print(f"\n{'='*80}")
    print("ASPECT-BASED SENTIMENT ANALYSIS")
    print("Using spaCy + VADER")
    print(f"{'='*80}\n")
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"✓ Loaded {len(df)} comments")
    print(f"✓ Columns: {df.columns.tolist()}\n")
    
    # Prepare results storage
    aspect_sentiments = defaultdict(list)
    comment_aspects = []
    location_aspect_map = defaultdict(lambda: defaultdict(list))
    
    print("Analyzing aspects and sentiments using spaCy + VADER...")
    print("This may take a few minutes for large datasets...\n")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing: {idx}/{len(df)} comments ({idx/len(df)*100:.1f}%)")
        
        comment = row.get('Comment', '')
        if pd.isna(comment) or len(str(comment).strip()) < 10:
            continue
        
        # Extract aspects mentioned
        aspects = extract_aspects(comment)
        
        if not aspects:
            continue
        
        # Extract locations mentioned
        locations = extract_locations(str(row.get('Title', '')) + ' ' + comment)
        
        # Get sentiment for each aspect
        row_result = {
            'Comment_ID': idx,
            'Post_ID': row.get('ID', ''),
            'Comment_Text': comment[:150] + '...' if len(str(comment)) > 150 else comment,
            'Full_Comment': comment,
            'Date': row.get('Created_UTC', ''),
            'Subreddit': row.get('Subreddit', ''),
            'Post_Title': row.get('Title', ''),
            'Overall_Sentiment': row.get('Comment_Sentiment_Label', 'Unknown'),
            'Overall_Compound': row.get('Comment_Compound_Score', 0),
            'Aspects_Mentioned': ', '.join(aspects) if aspects else 'None',
            'Locations_Mentioned': ', '.join(locations) if locations else 'None',
            'Num_Aspects': len(aspects)
        }
        
        # Analyze sentiment for each aspect
        for aspect in aspects:
            sentiment_score = get_aspect_sentiment_vader(comment, aspect)
            if sentiment_score is not None:
                aspect_sentiments[aspect].append(sentiment_score)
                category, strength = get_sentiment_details(sentiment_score)
                
                row_result[f'{aspect}_Sentiment'] = category
                row_result[f'{aspect}_Score'] = round(sentiment_score, 4)
                row_result[f'{aspect}_Strength'] = strength
                
                # Track location-aspect relationships
                for location in locations:
                    location_aspect_map[location][aspect].append(sentiment_score)
        
        comment_aspects.append(row_result)
    
    print(f"\n✓ Analysis complete! Processed {len(comment_aspects)} comments with aspects\n")
    
    # Create detailed results dataframe
    results_df = pd.DataFrame(comment_aspects)
    
    # Calculate aspect statistics
    print("Calculating statistics...")
    aspect_stats = []
    for aspect, scores in aspect_sentiments.items():
        positive = sum(1 for s in scores if s >= 0.05)
        negative = sum(1 for s in scores if s <= -0.05)
        neutral = sum(1 for s in scores if -0.05 < s < 0.05)
        
        aspect_stats.append({
            'Aspect': aspect,
            'Mention_Count': len(scores),
            'Avg_Sentiment': round(np.mean(scores), 4),
            'Std_Dev': round(np.std(scores), 4),
            'Min_Score': round(np.min(scores), 4),
            'Max_Score': round(np.max(scores), 4),
            'Median_Score': round(np.median(scores), 4),
            'Positive_Count': positive,
            'Negative_Count': negative,
            'Neutral_Count': neutral,
            'Positive_Pct': round(positive / len(scores) * 100, 2),
            'Negative_Pct': round(negative / len(scores) * 100, 2),
            'Neutral_Pct': round(neutral / len(scores) * 100, 2)
        })
    
    stats_df = pd.DataFrame(aspect_stats)
    stats_df = stats_df.sort_values('Mention_Count', ascending=False)
    
    # Calculate location-aspect statistics
    location_stats = []
    for location, aspects in location_aspect_map.items():
        for aspect, scores in aspects.items():
            if len(scores) >= 3:  # Only include if at least 3 mentions
                location_stats.append({
                    'Location': location,
                    'Aspect': aspect,
                    'Mention_Count': len(scores),
                    'Avg_Sentiment': round(np.mean(scores), 4),
                    'Positive_Pct': round(sum(1 for s in scores if s >= 0.05) / len(scores) * 100, 2),
                    'Negative_Pct': round(sum(1 for s in scores if s <= -0.05) / len(scores) * 100, 2)
                })
    
    location_df = pd.DataFrame(location_stats)
    if not location_df.empty:
        location_df = location_df.sort_values(['Location', 'Mention_Count'], ascending=[True, False])
    
    # Save results
    print("Saving results...")
    results_file = f'{output_prefix}_detailed.csv'
    stats_file = f'{output_prefix}_summary.csv'
    location_file = f'{output_prefix}_by_location.csv'
    
    results_df.to_csv(results_file, index=False)
    stats_df.to_csv(stats_file, index=False)
    if not location_df.empty:
        location_df.to_csv(location_file, index=False)
    
    print(f"✓ Detailed results saved: {results_file}")
    print(f"✓ Summary statistics saved: {stats_file}")
    if not location_df.empty:
        print(f"✓ Location analysis saved: {location_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(stats_df, aspect_sentiments, output_prefix)
    
    # Print summary
    print_summary(df, results_df, stats_df)
    
    return results_df, stats_df, location_df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(stats_df, aspect_sentiments, output_prefix):
    """Create comprehensive visualizations"""
    
    # 1. Aspect Mention Frequency
    plt.figure(figsize=(12, 6))
    top_aspects = stats_df.head(10)
    plt.barh(top_aspects['Aspect'], top_aspects['Mention_Count'], color='steelblue')
    plt.xlabel('Number of Mentions', fontsize=12, fontweight='bold')
    plt.ylabel('Aspect', fontsize=12, fontweight='bold')
    plt.title('Top 10 Most Discussed Aspects', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_mention_frequency.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_mention_frequency.png")
    plt.close()
    
    # 2. Average Sentiment by Aspect
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if x >= 0.05 else '#e74c3c' if x <= -0.05 else '#95a5a6' 
              for x in top_aspects['Avg_Sentiment']]
    plt.barh(top_aspects['Aspect'], top_aspects['Avg_Sentiment'], color=colors)
    plt.xlabel('Average Sentiment Score', fontsize=12, fontweight='bold')
    plt.ylabel('Aspect', fontsize=12, fontweight='bold')
    plt.title('Average Sentiment Score by Aspect (VADER)', fontsize=14, fontweight='bold', pad=20)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=0.05, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Positive threshold')
    plt.axvline(x=-0.05, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Negative threshold')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_avg_sentiment.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_avg_sentiment.png")
    plt.close()
    
    # 3. Stacked Bar Chart - Sentiment Distribution
    plt.figure(figsize=(14, 8))
    aspects = top_aspects['Aspect'].values
    positive = top_aspects['Positive_Pct'].values
    neutral = top_aspects['Neutral_Pct'].values
    negative = top_aspects['Negative_Pct'].values
    
    x = np.arange(len(aspects))
    width = 0.8
    
    p1 = plt.bar(x, positive, width, label='Positive', color='#2ecc71')
    p2 = plt.bar(x, neutral, width, bottom=positive, label='Neutral', color='#95a5a6')
    p3 = plt.bar(x, negative, width, bottom=positive+neutral, label='Negative', color='#e74c3c')
    
    plt.xlabel('Aspect', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('Sentiment Distribution by Aspect (%)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, aspects, rotation=45, ha='right')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_sentiment_distribution.png")
    plt.close()
    
    # 4. Box Plot - Sentiment Score Distribution
    fig, ax = plt.subplots(figsize=(14, 8))
    
    data_to_plot = []
    labels = []
    for aspect in top_aspects['Aspect'].values:
        if aspect in aspect_sentiments:
            data_to_plot.append(aspect_sentiments[aspect])
            labels.append(aspect)
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     notch=True, showmeans=True, meanline=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Color the median lines
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(2)
    
    # Color the mean lines
    for mean in bp['means']:
        mean.set_color('red')
        mean.set_linewidth(2)
    
    plt.xlabel('Aspect', fontsize=12, fontweight='bold')
    plt.ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    plt.title('Sentiment Score Distribution by Aspect (Box Plot)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axhline(y=0.05, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axhline(y=-0.05, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_boxplot.png")
    plt.close()
    
    print("✓ All visualizations created successfully!")

# ============================================================================
# SUMMARY PRINTING
# ============================================================================

def print_summary(df, results_df, stats_df):
    """Print analysis summary"""
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total comments in dataset: {len(df)}")
    print(f"Comments with identified aspects: {len(results_df)} ({len(results_df)/len(df)*100:.1f}%)")
    print(f"Total aspect mentions: {stats_df['Mention_Count'].sum()}")
    
    print(f"\n{'─'*80}")
    print("TOP 5 MOST DISCUSSED ASPECTS:")
    print(f"{'─'*80}")
    for _, row in stats_df.head(5).iterrows():
        print(f"\n{row['Aspect']}:")
        print(f"  • Mentions: {row['Mention_Count']}")
        print(f"  • Avg Sentiment: {row['Avg_Sentiment']:.4f}")
        print(f"  • Distribution: {row['Positive_Pct']:.1f}% Pos | {row['Neutral_Pct']:.1f}% Neu | {row['Negative_Pct']:.1f}% Neg")
    
    print(f"\n{'─'*80}")
    print("SENTIMENT RANKING (Best to Worst):")
    print(f"{'─'*80}")
    ranked = stats_df.sort_values('Avg_Sentiment', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        emoji = "🟢" if row['Avg_Sentiment'] >= 0.05 else "🔴" if row['Avg_Sentiment'] <= -0.05 else "🟡"
        print(f"{i}. {emoji} {row['Aspect']}: {row['Avg_Sentiment']:.4f}")
    
    print(f"\n{'='*80}\n")

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

def analyze_temporal_trends(df, output_prefix='temporal_analysis'):
    """Analyze sentiment trends over time"""
    print("\nAnalyzing temporal trends...")
    
    if 'Created_UTC' not in df.columns:
        print("⚠ No date column found. Skipping temporal analysis.")
        return
    
    # Convert to datetime
    df['Date'] = pd.to_datetime(df['Created_UTC'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['Year_Month'] = df['Date'].dt.to_period('M')
    
    # Filter out invalid dates
    df_with_dates = df[df['Date'].notna()]
    
    if len(df_with_dates) == 0:
        print("⚠ No valid dates found. Skipping temporal analysis.")
        return
    
    temporal_results = []
    
    for period in df_with_dates['Year_Month'].unique():
        period_df = df_with_dates[df_with_dates['Year_Month'] == period]
        
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
    
    temporal_df = pd.DataFrame(temporal_results)
    
    if not temporal_df.empty:
        temporal_df.to_csv(f'{output_prefix}.csv', index=False)
        print(f"✓ Temporal analysis saved: {output_prefix}.csv")
        
        # Create temporal visualization
        top_aspects = temporal_df.groupby('Aspect')['Count'].sum().nlargest(5).index
        
        plt.figure(figsize=(14, 8))
        for aspect in top_aspects:
            aspect_data = temporal_df[temporal_df['Aspect'] == aspect].sort_values('Period')
            if not aspect_data.empty:
                plt.plot(aspect_data['Period'], aspect_data['Avg_Sentiment'], 
                        marker='o', label=aspect, linewidth=2, markersize=6)
        
        plt.xlabel('Time Period', fontsize=12, fontweight='bold')
        plt.ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
        plt.title('Sentiment Trends Over Time by Aspect', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.axhline(y=0.05, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.axhline(y=-0.05, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_prefix}.png")
        plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("ASPECT-BASED SENTIMENT ANALYSIS")
    print("Using spaCy for NLP + VADER for Sentiment Analysis")
    print("="*80)
    
    # Configuration
    input_file = 'reddit_singapore_anytimefitness_comments.csv'
    output_prefix = 'anytime_fitness_absa'
    
    # Perform main analysis
    results_df, stats_df, location_df = perform_aspect_based_sentiment_analysis(
        input_file, 
        output_prefix=output_prefix
    )
    
    # Perform temporal analysis
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    analyze_temporal_trends(df, output_prefix=f'{output_prefix}_temporal')
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"1. {output_prefix}_detailed.csv - Detailed analysis per comment")
    print(f"2. {output_prefix}_summary.csv - Summary statistics by aspect")
    print(f"3. {output_prefix}_by_location.csv - Analysis by location")
    print(f"4. {output_prefix}_mention_frequency.png")
    print(f"5. {output_prefix}_avg_sentiment.png")
    print(f"6. {output_prefix}_sentiment_distribution.png")
    print(f"7. {output_prefix}_boxplot.png")
    print(f"8. {output_prefix}_temporal.csv")
    print(f"9. {output_prefix}_temporal.png")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
