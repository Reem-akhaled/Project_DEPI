import pandas as pd
from collections import Counter
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- NLTK RESOURCE CHECK & DOWNLOAD ---
def download_nltk_resources():
    """Ensure all required NLTK resources are available."""
    resources = [
        'punkt', 'stopwords', 'averaged_perceptron_tagger',
        'wordnet', 'omw-1.4',
    ]
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            try:
                nltk.data.find(f'taggers/{res}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{res}')
                except LookupError:
                    logging.info(f"Downloading NLTK resource: {res}")
                    nltk.download(res, quiet=True)

download_nltk_resources()


# ====================================================================
# CONFIGURATION VARIABLES (Edit these)
# ====================================================================

# 1. FILE PATH
FILE_PATH = "Bank_Reviews_Dataset_Expanded_3000.csv"
OUTPUT_FILE = "Bank_Word_Analysis.xlsx"

# 2. COLUMN NAMES
RATING_COLUMN = "rating"
# Used for Sheet 1 & 3 (Short text)
TEXT_COLUMN_SHORT = "rating_title_by_user"
# Used for Sheet 2 (Long text)
TEXT_COLUMN_LONG = "review"

# 3. WORD COUNT SETTINGS
# Settings for Sheet 1 (Standard frequency analysis)
BIGRAM_MERGE_THRESHOLD = 3
# Settings for Sheet 2 (Word Cloud Data)
N_TOP_WORDS_WC = 50
# Settings for Sheet 3 (POS Tagging analysis)
N_TOP_WORDS_CATEGORY = 10

# 4. STOP WORD / EXCLUSION LISTS

# A. STOP WORDS FOR SHEET 1 (Simple Unigram/Bigram Frequency)
STOP_WORDS_SHEET1 = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'could', 'would',

    # Banking/Review-specific words for Sheet 1
    'bank', 'service', 'customer', 'account', 'card', 'branch', 'online', 'app',
    'money', 'time', 'issue', 'problem', 'get', 'like', 'call', 'told',
    'really', 'much', 'back', 'go', 'use', 'know', 'say', 'one',
    'staff', 'since', 'good', 'great', 'always', 'even', 'also', 'still',
    'said', 'make', 'want', 'need', 'new', 'people', 'well', 'sure', 'never',
    'thing', 'wait', 'look', 'first', 'last', 'next', 'title', 'rating', 'review',
    'years', 'days', 'hours', 'month', 'per', 'etc', 'loan', 'credit', 'business',
    'away', 'blown'
])


# B. EXCLUSION LISTS FOR SHEET 2 & 3 (Word Cloud / POS Tagging)
NLTK_STOPS = set(stopwords.words('english'))
BANK_NOISE_SHEET2 = set([
    "sbi", "axis", "hdfc", "unknown", "idbi", "kotak", "indusind",
    "canara", "citibank", "punjab", "national", "icici", "pnb", "baroda",
    "bank", "banks", "banking", "review", "reviews", "customer", "service",
    "account", "card"
])
# Consolidated blocklist for POS tagging (Sheet 3)
BLOCKLIST_SHEET3 = NLTK_STOPS.union(set([
    "bank", "sbi", "axis", "hdfc", "icici", "kotak", "pnb", "baroda",
    "indusind", "canara", "citibank", "idbi", "union", "india", "punjab",
    "national", "branch", "atm", "account", "app", "application", "money",
    "transaction", "service", "customer", "staff", "manager", "experience",
    "issue", "problem", "review", "sms", "email", "mail", "call", "phone",
    "number", "mobile", "online", "digital", "server", "link", "site",
    "system", "network", "card", "credit", "debit", "fund", "salary",
    "charge", "charges", "balance", "minimum", "maximum", "interest"
]))
VAGUE_ADJECTIVES_SHEET3 = set([
    "other", "another", "such", "same", "different", "various", "certain",
    "more", "much", "most", "less", "least", "many", "few", "some", "any",
    "previous", "last", "next", "first", "second", "third", "past", "recent",
    "present", "future", "daily", "monthly", "weekly", "yearly", "annual",
    "due", "available", "applicable", "unable", "able", "ready", "total",
    "net", "gross", "major", "minor", "real", "actual", "usual", "normal",
    "particular", "specific", "general", "hidden", "extra", "full", "whole",
    "proper", "high", "low", "non", "top", "old", "new", "big", "small",
    "own", "private", "public", "main", "financial", "indian"
])
WHITELIST_SHORT_SHEET3 = {"bad", "sad", "hot", "fun", "wow"}


# ====================================================================
# FUNCTIONS FOR SHEET 1: 'Words Summary By Rating' (Unigram/Bigram Method on TEXT_COLUMN_SHORT)
# ====================================================================

def clean_text_sheet1(text):
    """ Cleans and tokenizes text using the SHEET 1 stop word list. """
    if pd.isna(text): return []
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS_SHEET1 and len(word) > 2
    ]
    return tokens

def get_ngrams(tokens, n=2):
    """ Generates n-grams (bigrams if n=2) from a list of tokens. """
    if not tokens or len(tokens) < n: return []
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def temp_clean_text_sheet1(text, temp_stop_words):
    """ Temporary cleaning function for bigram generation. """
    if pd.isna(text): return []
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return [word for word in text.split() if word not in temp_stop_words and len(word) > 2]

def analyze_subset_sheet1(df_subset, subset_name):
    """
    Analyzes a DataFrame subset (TEXT_COLUMN_SHORT) for word frequency and returns the most/least frequent.
    """
    logging.info(f"Analyzing subset (Sheet 1): {subset_name}...")

    # --- 1. Generate Unigrams and Bigrams ---
    list_of_unigram_lists = df_subset[TEXT_COLUMN_SHORT].apply(clean_text_sheet1)
    all_unigrams = [token for sublist in list_of_unigram_lists for token in sublist]
    unigram_counts = Counter(all_unigrams)

    bigram_specific_words = {'blown', 'away'}
    temp_stop_words = STOP_WORDS_SHEET1 - bigram_specific_words

    list_of_temp_unigram_lists = df_subset[TEXT_COLUMN_SHORT].apply(lambda x: temp_clean_text_sheet1(x, temp_stop_words))
    all_bigrams = [bigram for unigram_list in list_of_temp_unigram_lists for bigram in get_ngrams(unigram_list, n=2)]
    bigram_counts = Counter(all_bigrams)

    # --- 2. Combine Unigrams and Bigrams ---
    final_ranking = unigram_counts.copy()

    for bigram, count in bigram_counts.most_common():
        if count >= BIGRAM_MERGE_THRESHOLD:
            final_ranking[bigram] = count
            words = bigram.split(' ')
            for word in words:
                if word not in STOP_WORDS_SHEET1:
                    final_ranking[word] = max(0, final_ranking.get(word, 0) - count)
                    if final_ranking[word] <= 0:
                        del final_ranking[word]

    if not final_ranking:
        return "N/A", 0, "N/A", 0, []

    # --- 3. Extract Most and Least Frequent ---
    sorted_ranking = final_ranking.most_common()
    most_frequent_word, most_frequent_count = sorted_ranking[0]
    rare_items = [item for item in final_ranking.items() if item[1] > 1]
    if rare_items:
        sorted_rare_items = sorted(rare_items, key=lambda item: item[1])
        least_frequent_word, least_frequent_count = sorted_rare_items[0]
    else:
        least_frequent_word, least_frequent_count = sorted_ranking[-1]

    return most_frequent_word, most_frequent_count, least_frequent_word, least_frequent_count, None


# ====================================================================
# FUNCTIONS FOR SHEET 2: 'Word Cloud Data' (Simple Unigram Method on TEXT_COLUMN_LONG)
# ====================================================================

def clean_and_count_words_sheet2(df, column_name):
    """
    Processes the text data based on the Word Cloud script's logic (NLTK stopwords + exclusions).
    Uses TEXT_COLUMN_LONG.
    """
    logging.info(f"Starting word frequency analysis on column: '{column_name}' for Sheet 2...")

    df = df.dropna(subset=[column_name])
    stop_words_wc = NLTK_STOPS
    all_words = []
    punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')

    for review in df[column_name]:
        text = review.lower()
        text = punctuation_pattern.sub('', text)
        words = text.split()

        filtered_words = [word for word in words
                          if word not in stop_words_wc and
                             len(word) > 2 and
                             word.isalpha() and
                             word not in BANK_NOISE_SHEET2]

        all_words.extend(filtered_words)

    word_counts = Counter(all_words)
    top_words = word_counts.most_common(N_TOP_WORDS_WC)

    return top_words


# ====================================================================
# FUNCTIONS FOR SHEET 3: 'Top Words per Category' (POS Tagging Method on TEXT_COLUMN_SHORT)
# ====================================================================

def get_sentiment_category(rating):
    """Maps numerical rating to sentiment category."""
    try:
        r = float(rating)
        if 0 <= r <= 2: return 'Negative'
        elif r == 3: return 'Neutral'
        elif 4 <= r <= 5: return 'Positive'
        return 'Unknown'
    except: return 'Unknown'

def get_meaningful_terms_sheet3(text_series):
    """
    Extracts terms using NLTK POS tagging, lemmatization, and specific filtering rules.
    Uses TEXT_COLUMN_SHORT.
    """
    lemmatizer = WordNetLemmatizer()
    extracted_terms = []
    blocklist = BLOCKLIST_SHEET3.union(VAGUE_ADJECTIVES_SHEET3)

    for review in text_series:
        if not isinstance(review, str): continue

        clean_review = re.sub(r'[^a-z\s]', ' ', review.lower())
        clean_review = re.sub(r'\s+', ' ', clean_review)

        words = word_tokenize(clean_review)
        if not words: continue

        try:
            tagged = nltk.pos_tag(words)
        except Exception as e:
            logging.warning(f"POS tagging failed: {e}")
            continue

        i = 0
        while i < len(tagged):
            word, tag = tagged[i]
            root_word = lemmatizer.lemmatize(word)

            consumed = False

            # --- PHRASE CHECK (Priority) ---
            if i < len(tagged) - 1:
                next_word, next_tag = tagged[i+1]
                is_valid_phrase = False

                # Check specific POS patterns
                if (tag.startswith('RB') and next_tag.startswith('JJ')) or \
                   (tag.startswith('JJ') and next_tag.startswith('JJ')):
                    is_valid_phrase = True

                # Check for Adjective + specific noun patterns
                elif tag.startswith('JJ') and lemmatizer.lemmatize(next_word) in ["experience", "service", "staff", "behavior", "response"]:
                    is_valid_phrase = True

                # Check for Negation + Verb
                elif word == "not" and next_tag.startswith('VB'):
                    is_valid_phrase = True

                # Check for "blown away"
                elif word == "blown" and next_word == "away":
                    is_valid_phrase = True

                if is_valid_phrase:
                    # Final filter against vague words in the phrase
                    if word in VAGUE_ADJECTIVES_SHEET3 or next_word in VAGUE_ADJECTIVES_SHEET3:
                         is_valid_phrase = False

                if is_valid_phrase:
                    extracted_terms.append(f"{word} {next_word}")
                    consumed = True
                    i += 1 # Skip next word

            # --- SINGLE WORD CHECK ---
            if not consumed:
                # Target Adjectives only
                if tag.startswith('JJ'):
                    if root_word not in blocklist:
                        # Apply length filter or whitelist short words
                        if len(word) >= 4 or word in WHITELIST_SHORT_SHEET3:
                            extracted_terms.append(word)

            i += 1

    return extracted_terms


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def analyze_reviews():
    """
    Main function to read data and generate the Excel file with 3 sheets.
    """
    # 1. Load the CSV file
    try:
        logging.info(f"Loading data from: {FILE_PATH}")
        df = pd.read_csv(FILE_PATH)
        logging.info(f"Successfully loaded '{FILE_PATH}'. Total rows: {len(df)}")
    except FileNotFoundError:
        logging.error(f"Error: The file '{FILE_PATH}' was not found. Please check the path.")
        return
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV: {e}")
        return

    # 2. Input column checks
    missing_cols = []
    if TEXT_COLUMN_SHORT not in df.columns: missing_cols.append(TEXT_COLUMN_SHORT)
    if RATING_COLUMN not in df.columns: missing_cols.append(RATING_COLUMN)
    if TEXT_COLUMN_LONG not in df.columns: missing_cols.append(TEXT_COLUMN_LONG)

    if missing_cols:
        logging.error(f"Error: Column(s) '{', '.join(missing_cols)}' not found in the dataset columns: {list(df.columns)}")
        return

    # --- SETUP: Sentiment Grouping for Sheet 1 & 3 ---
    df['Rating Category'] = df[RATING_COLUMN].apply(get_sentiment_category)
    df_filtered = df[df['Rating Category'] != 'Unknown'].copy()

    df_positive = df_filtered[df_filtered['Rating Category'] == 'Positive'].copy()
    df_negative = df_filtered[df_filtered['Rating Category'] == 'Negative'].copy()
    df_neutral = df_filtered[df_filtered['Rating Category'] == 'Neutral'].copy()

    # --------------------------------------------------------------------
    # SECTION A: Data for SHEET 1: 'Words Summary By Rating' (Short Text + Bigram)
    # --------------------------------------------------------------------
    # Analyze ALL DATA first (required for bigram deduction logic)
    analyze_subset_sheet1(df_filtered, "ALL DATA (Required Setup)") # Result is discarded

    most_pos, count_most_pos, least_pos, count_least_pos, _ = analyze_subset_sheet1(df_positive, "Positive Ratings")
    most_neg, count_most_neg, least_neg, count_least_neg, _ = analyze_subset_sheet1(df_negative, "Negative Ratings")
    most_neut, count_most_neut, least_neut, count_least_neut, _ = analyze_subset_sheet1(df_neutral, "Neutral Ratings")

    # Create DataFrame for Sheet 1
    summary_rating_data = {
        'Rating Category': ['Positive', 'Negative', 'Neutral'],
        'Most Frequent Word': [most_pos, most_neg, most_neut],
        'Most Frequent Word Count': [count_most_pos, count_most_neg, count_most_neut],
        'Least Frequent Word': [least_pos, least_neg, least_neut],
        'Least Frequent Word Count': [count_least_pos, count_least_neg, count_least_neut],
    }
    df_summary_by_rating = pd.DataFrame(summary_rating_data)


    # --------------------------------------------------------------------
    # SECTION B: Data for SHEET 2: 'Word Cloud Data' (Long Text + Simple Unigram)
    # --------------------------------------------------------------------
    word_cloud_ranking = clean_and_count_words_sheet2(df, TEXT_COLUMN_LONG)

    # Create DataFrame for Sheet 2
    df_word_cloud = pd.DataFrame(
        word_cloud_ranking,
        columns=['Word/Phrase', 'Occurrence Count']
    )

    # --------------------------------------------------------------------
    # SECTION C: Data for SHEET 3: 'Top Words per Category' (Short Text + POS Tagging)
    # --------------------------------------------------------------------
    final_results_sheet3 = []

    for category in ['Negative', 'Neutral', 'Positive']:
        subset = df_filtered[df_filtered['Rating Category'] == category]
        # Use TEXT_COLUMN_SHORT as requested
        all_terms = get_meaningful_terms_sheet3(subset[TEXT_COLUMN_SHORT])

        if not all_terms: continue

        # Get TOP 10 ONLY
        counts = Counter(all_terms).most_common(N_TOP_WORDS_CATEGORY)

        for term, count in counts:
            final_results_sheet3.append({
                'Word': term,
                'Count': count,
                'Rating Category': category
            })

    # Create DataFrame for Sheet 3
    df_top_words_category = pd.DataFrame(final_results_sheet3)


    # ----------------------------------------------------
    # WRITE TO EXCEL FILE (3 Sheets)
    # ----------------------------------------------------
    try:
        logging.info(f"Writing results to Excel file: {OUTPUT_FILE} (3 sheets)")
        # Using xlsxwriter for consistent formatting and cell column adjustments
        with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:

            # Sheet 1: Words Summary By Rating (Standard Excel Write)
            df_summary_by_rating.to_excel(writer, sheet_name='Words Summary By Rating', index=False)
            worksheet1 = writer.sheets['Words Summary By Rating']
            worksheet1.set_column('A:A', 20)
            worksheet1.set_column('B:B', 20)
            worksheet1.set_column('C:C', 25)
            worksheet1.set_column('D:D', 20)
            worksheet1.set_column('E:E', 25)

            # Sheet 2: Word Cloud Data (Standard Excel Write)
            df_word_cloud.to_excel(writer, sheet_name='Word Cloud Data', index=False)
            worksheet2 = writer.sheets['Word Cloud Data']
            worksheet2.set_column('A:A', 30)
            worksheet2.set_column('B:B', 20)

            # Sheet 3: Top Words per Category (Custom Header Write as requested)

            # Prepare custom header rows exactly as requested by the user's logic
            header_row = pd.DataFrame({'Word': ['Word'], 'Count': ['Count'], 'Rating Category': ['Rating Category']})
            export_df_sheet3 = pd.concat([header_row, df_top_words_category], ignore_index=True)
            # Temporary column names to avoid pandas using them as a header
            export_df_sheet3.columns = ['A', 'B', 'C']

            export_df_sheet3.to_excel(
                writer,
                sheet_name='Top Words per Category',
                index=False,
                header=False, # Crucial: prevents pandas from writing its own header
                startrow=0    # Start writing at the very first row
            )
            worksheet3 = writer.sheets['Top Words per Category']
            worksheet3.set_column('A:A', 30)
            worksheet3.set_column('B:B', 15)
            worksheet3.set_column('C:C', 20)


        logging.info("Successfully created the Excel file.")
    except Exception as e:
        logging.error(f"An error occurred while writing the Excel file: {e}")


if __name__ == '__main__':
    analyze_reviews()
