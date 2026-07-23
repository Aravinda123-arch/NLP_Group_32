import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

def generate_insights(df):
    """Method 5: Generates automated statistical insights to the console."""
    print("\n" + "="*40)
    print("🧠 AUTOMATED EDA INSIGHTS")
    print("="*40)
    
    total_articles = len(df)
    true_count = len(df[df['label_name'] == 'True News'])
    fake_count = len(df[df['label_name'] == 'Fake News'])
    
    # Calculate how many articles fit into BERT's 512 token limit
    under_512 = len(df[df['word_count'] <= 512])
    percent_under_512 = (under_512 / total_articles) * 100
    
    print(f"Total Articles Analyzed: {total_articles}")
    print(f"Class Balance: {true_count} True | {fake_count} Fake")
    print(f"Average Word Count: {int(df['word_count'].mean())} words")
    print(f"Max Word Count: {df['word_count'].max()} words")
    print(f"BERT Suitability: {percent_under_512:.2f}% of articles are under 512 words.")
    if percent_under_512 > 90:
        print("-> Insight: Safe to use max_length=512 or 256 for BERT without losing much context.")
    else:
        print("-> Insight: Consider truncation strategies, as many articles exceed 512 words.")
    print("="*40 + "\n")

def get_top_ngrams(corpus, n=15, ngram_range=(1, 1)):
    """Helper function to extract top N-Grams and remove stop words."""
    # We remove english stop words (like 'the', 'is', 'and') so the charts show meaningful words
    vec = CountVectorizer(stop_words='english', ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    return pd.DataFrame(words_freq[:n], columns=['N-Gram', 'Frequency'])

def run_master_eda():
    # 1. Load the preprocessed dataset
    # (Note: Pointing to master_preprocessed_dataset since it has the 'Reuters' fix)
    file_path = "data/advanced_preprocessed_dataset.csv" 
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Drop any accidental missing values just to be safe
    df = df.dropna(subset=['fused_text', 'label'])
    
    # Map labels to readable names
    if df['label'].dtype in ['int64', 'int32']:
        df['label_name'] = df['label'].map({0: 'True News', 1: 'Fake News'})
    else:
        df['label_name'] = df['label']

    # Pre-calculate word count for distributions
    df['word_count'] = df['fused_text'].apply(lambda x: len(str(x).split()))
    
    # Set standard visualization style
    sns.set_theme(style="whitegrid")
    
    # Separate the texts by class for text analysis
    true_texts = df[df['label_name'] == 'True News']['fused_text'].astype(str)
    fake_texts = df[df['label_name'] == 'Fake News']['fused_text'].astype(str)

    # ==========================================
    # METHOD 1: CLASS DISTRIBUTION
    # ==========================================
    print("1. Generating Class Distribution plot...")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='label_name',hue='label_name' ,palette='Set2',legend=False)
    plt.title('Class Distribution: Fake vs True News', fontsize=14, fontweight='bold')
    plt.xlabel('News Type', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.show()

    # ==========================================
    # METHOD 2: DATA DISTRIBUTION (Text Length Histogram for BERT)
    # ==========================================
    print("2. Generating Data Distribution (Word Count) plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='word_count', hue='label_name', bins=60, palette='Set2')
    plt.title('Data Distribution: Word Count per Article', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 2000) 
    
    # Add the critical BERT max_length cutoff line
    plt.axvline(x=512, color='red', linestyle='--', label='BERT 512 Max Token Limit')
    plt.legend()
    plt.show()

    # ==========================================
    # METHOD 3: WORD FREQUENCY ANALYSIS (Unigrams)
    # ==========================================
    print("3. Generating Single Word Frequency Analysis...")
    top_true_words = get_top_ngrams(true_texts, n=15, ngram_range=(1, 1))
    top_fake_words = get_top_ngrams(fake_texts, n=15, ngram_range=(1, 1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(ax=axes[0], data=top_true_words, x='Frequency', y='N-Gram', hue='N-Gram', palette='Blues_r', legend=False)
    axes[0].set_title('Top 15 Single Words in True News', fontsize=14, fontweight='bold')
    
    sns.barplot(ax=axes[1], data=top_fake_words, x='Frequency', y='N-Gram',hue='N-Gram', palette='Reds_r',legend=False)
    axes[1].set_title('Top 15 Single Words in Fake News', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ==========================================
    # METHOD 4: N-GRAM FREQUENCY ANALYSIS (Bigrams)
    # ==========================================
    print("4. Generating N-Gram (Bigram) Frequency Charts...")
    top_true_bigrams = get_top_ngrams(true_texts, n=15, ngram_range=(2, 2))
    top_fake_bigrams = get_top_ngrams(fake_texts, n=15, ngram_range=(2, 2))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(ax=axes[0], data=top_true_bigrams, x='Frequency', y='N-Gram',hue='N-Gram', palette='Greens_r',legend=False)
    axes[0].set_title('Top 15 Bigrams (2-Words) in True News', fontsize=14, fontweight='bold')
    
    sns.barplot(ax=axes[1], data=top_fake_bigrams, x='Frequency', y='N-Gram',hue='N-Gram' , palette='Oranges_r',legend=False)
    axes[1].set_title('Top 15 Bigrams (2-Words) in Fake News', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ==========================================
    # METHOD 5: INSIGHTS GENERATION
    # ==========================================
    generate_insights(df)

if __name__ == "__main__":
    run_master_eda()