from textblob import TextBlob
import streamlit as st
import pandas as pd
from collections import Counter
import re
import subprocess
import nltk
from nltk.corpus import stopwords

#title/subheader set here
st.title("Customer Sentiment Data Entry")
st.subheader("Enter weekly ZD CSAT, App Reviews, and Jira Information in the appropiate fields below")
#input from users is set here
app_reviews = st.text_area("Enter App Reviews here, leave a line break between reviews")
facebook_comments = st.text_area("Enter Facebook posts here, leave a line break between posts")
zd_csat_value = st.number_input("Enter a ZD CSAT number:", min_value=0.0, step=0.01, format="%.2f")
jira_value = st.number_input("Enter Jira score here:", min_value=0.0, step=0.01, format ="%.2f")

# Custom lexicon for sentiment analysis
custom_lexicon = {
    # Positive words
    'love': 2.0, 'fantastic': 2.0, 'helpful': 1.5, 'amazing': 2.0, 'great': 1.5, 'awesome': 1.5,
    'appreciate': 1.5, 'enjoy': 1.5, 'easy': 1.5, 'intuitive': 1.5, 'user-friendly': 1.5,
    'efficient': 1.5, 'streamlined': 1.5, 'well-designed': 1.5, 'smooth': 1.5, 'seamless': 1.5,
    'responsive': 1.5, 'accurate': 1.5, 'clear': 1.5, 'collaborative': 1.5, 'supportive': 1.5,
    'connected': 1.5, 'helpful feature': 1.5, 'successful': 1.5, 'benefit': 1.5, 'reliable': 1.5,
    'improving': 1.5, 'motivating': 1.5, 'insightful': 1.5, 'productive': 1.5, 'consistent': 1.5,
    'solid': 1.5, 'well-organized': 1.5, 'excellent': 2.0, 'top-notch': 2.0, 'highly recommend': 2.0,
    'well-executed': 1.5, 'collaborative environment': 1.5, 'positive outcome': 1.5, 'empowering': 1.5,
    'excited': 1.5, 'motivating': 1.5, 'flexibility': 1.5, 'powerful': 1.5, 'clarity': 1.5, 'potential': 1.0,
    'eager': 1.0, 'looking forward': 1.5, 'curious': 1.0, 'could you help': 0.5, 'would it be possible': 0.5,
    'suggestion': 0.5,

    # Neutral words
    'data': 0.0, 'session': 0.0, 'training': 0.0, 'workout': 0.0, 'feedback': 0.0, 'effort': 0.0,
    'schedule': 0.0, 'goal': 0.0, 'performance': 0.0, 'tracking': 0.0, 'program': 0.0, 'function': 0.0,
    'feature': 0.0, 'metrics': 0.0, 'pace': 0.0,

    # Negative words
    'frustrating': -1.5, 'disappointing': -1.5, 'confusing': -1.5, 'clunky': -1.5, 'slow': -1.0,
    'unreliable': -1.5, 'inaccurate': -1.5, 'outdated': -1.5, 'broken': -2.0, 'difficult': -1.5,
    'complicated': -1.5, 'problem': -1.5, 'issue': -1.0, 'annoying': -1.5, 'not working': -1.5,
    'error': -1.5, 'glitchy': -1.5, 'needs improvement': -1.5, 'basic feature': -1.0, 'limited': -1.0,
    'unintuitive': -1.5, 'inconsistent': -1.5, 'lacks flexibility': -1.0, 'tedious': -1.0,
    'slow response': -1.0, 'poor design': -2.0, 'time-consuming': -1.5, 'unresponsive': -1.5,
    'difficult to navigate': -1.5, 'overly complex': -1.5, 'bugs': -1.5, 'unreliable sync': -1.5,
    'crashes': -2.0, 'not user-friendly': -1.5, 'poorly implemented': -1.5, 'overwhelming': -1.0,
    'needs work': -1.0
}

def analyze_sentiment(reviews, lexicon):
    total_polarity, total_subjectivity, total_custom_score = 0, 0, 0
    if not reviews:
        return 0, 0, 0

    for review in reviews:
        analysis = TextBlob(review)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        total_polarity += polarity
        total_subjectivity += subjectivity

        # Custom lexicon analysis
        words = review.lower().split()
        score = 0
        custom_score = sum(lexicon.get(word, 0) for word in words)
        total_custom_score += custom_score

    avg_polarity = total_polarity / len(reviews)
    avg_subjectivity = total_subjectivity / len(reviews)
    avg_custom_score = total_custom_score / len(reviews)
    
    for i, word in enumerate(words):
        
        if i < len(words) - 1 and words[i] == 'do' and words[i + 1] == 'not':
            if i + 2 < len(words) and words[i + 2] in custom_lexicon:
                # Negate the sentiment of the word after "do not"
                score -= custom_lexicon[words[i + 2]]
                skip = True  # Skip the next word since it is part of the negation
            continue
            # Handle cases like "not good"
        elif word == 'not' and i + 1 < len(words) and words[i + 1] in custom_lexicon:
            # Reverse the sentiment of the word following "not"
            score -= custom_lexicon[words[i + 1]]
            skip = True  # Skip the next word since it was already processed
        elif word in custom_lexicon:
            score += custom_lexicon[word]
    
    return avg_polarity, avg_subjectivity, avg_custom_score

# Split the text input by lines
app_reviews_list = app_reviews.split('\n') if app_reviews else []
facebook_comments_list = facebook_comments.split('\n') if facebook_comments else []

# Get average sentiment scores for App reviews and Facebook group comments
app_avg_polarity, app_avg_subjectivity, app_avg_custom_score = analyze_sentiment(app_reviews_list, custom_lexicon)
fb_avg_polarity, fb_avg_subjectivity, fb_avg_custom_score = analyze_sentiment(facebook_comments_list, custom_lexicon)

# Calculate combined average scores
def calculate_combined_score(avg_polarity, avg_custom_score):
    return (avg_polarity + avg_custom_score) / 2

combined_app_score = calculate_combined_score(app_avg_polarity, app_avg_custom_score)
combined_fb_score = calculate_combined_score(fb_avg_polarity, fb_avg_custom_score)

# Final calculation with weighted average function
def weighted_average(scores, weights, scales, target_scale):
    scaled_scores = [(score - scale[0]) / (scale[1] - scale[0]) * target_scale for score, scale in zip(scores, scales)]
    weighted_avg = sum(weight * score for weight, score in zip(weights, scaled_scores))
    return weighted_avg
    
final_app_score = (combined_app_score + 2) * 2.5
final_fb_score = (combined_fb_score + 2) * 2.5

# Use the combined sentiment scores to calculate the final weighted score
scores = [final_app_score, final_fb_score, zd_csat_value, jira_value]  # Combined App reviews, Combined FB comments, ZD CSAT

weights = [0.20, 0.05, 0.55, 0.20]  # Weight distribution
scales = [(3, 7), (3, 7), (8, 10), (0, 10)]  # Scales for the three scores
target_scale = 10  # Target scale

# Calculate final score
final_score = weighted_average(scores, weights, scales, target_scale)

if st.button("Submit"):
    result = ( round(final_app_score, 2),
               round(final_fb_score, 2),
               round(jira_value, 2),
               round(zd_csat_value, 2),
               round(final_score, 2)
               )
 
 # Create a formatted output message
    output_message = f"""
    Combined App Score: {result[0]}\n
    Combined FB Score: {result[1]}\n
    Jira Value: {result[2]}\n
    ZD CSAT Value: {result[3]}\n
    Final Score: {result[4]}\n
    """
    
    # Display the output message
    st.write(output_message)

#Begin ZD most used words script
    
nltk.download('stopwords')

# Load stop words from NLTK
nltk_stop_words = set(stopwords.words('english'))

# Custom list of syncategorematic words
custom_syncategorematic_words = custom_syncategorematic_words = { "and", "or", "but", "if", "while", "because", "although", "the", "a", "an", "in", "on", "with",
    "by", "to", "of", "for", "from", "at", "about", "between", "before", "after", "since", "until",
    "does", "not", "really", "due", "it", "appears", "yet", "possible", "can", "transfer", "from",
    "in", "addition", "to", "the", "opposite", "stroke", "works", "well", "is", "only", "bit",
    "my", "opinion", "perfect", "platform", "this", "though", "leaves", "a", "lot",
    "to", "be", "desired", "specially", "around", "you", "can't", "even", "move",
    "a", "workout", "to", "a", "different", "day", "may", "be", "more", "data", "than", "i", "need",
    "personally", "but", "overall", "it's", "a", "training", "app", "the", "sync", "with", "the",
    "is", "a", "touch", "confusing", "the", "says", "i", "did", "it", "training", "peaks", "says",
    "i", "did", "something", "but", "it", "doesn't", "seem", "to", "know", "what", "unfortunately",
    "the", "preview", "images", "in", "are", "misleading", "the", "app", "is", "of",
    "little", "use", "without", "a", "good", "knowledge", "of", "it's", "a", "shame",
    "i'm", "convinced", "that", "it", "would", "otherwise", "be", "helpful", "trainingpeaks", "me", "p",
    "com", "https", "www", "your", "please", "have", "1", "account", "hi", "email", "coach", "de", "thanks",
    "thank", "help", "2024", "like", "tp", "im", "get", "support", "submitted", "workouts", "one", "request", "athlete",
    "hello", "la", "time", "see", "attached", "el", "want", "ive", "dont", "new", "llc", "information", "could", "regards",
    "cant", "add", "2", "way", "athletes", "change", "back", "wrote", "view", "using", "que", "using", "make", "received",
    "un", "set", "date", "mi", "louisville", "co", "sep", "us", "didnt", "285", "century", "question", "able", "en", "happy"
}

st.title("Populate most used words in Zendesk")

# Combine NLTK stop words and custom syncategorematic words into one set
STOP_WORDS = nltk_stop_words.union(custom_syncategorematic_words)

# Function to clean and count words while ignoring stop words and unwanted characters
def count_words_ignore_stopwords(text_series, stop_words):
    words = []
    for text in text_series:
        # Remove unwanted characters like spaces, hyphens, and HTML tags using regex
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove non-word characters except spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)  # Remove HTML tags like <p>
        
        # Split the cleaned text into words, lowercase them, and filter out stop words
        words.extend([word for word in cleaned_text.lower().split() if word not in stop_words])
    
    return Counter(words)

# Build the file path through GitHub
df = pd.read_csv('zd_tickets.csv')
# Count word occurrences in the 'Description' column while ignoring stop words and unwanted characters
occurrences = count_words_ignore_stopwords(df['Description'], STOP_WORDS)


most_used_words = occurrences.most_common(50)

# Print the top 10 most used words with their counts
#for word, count in top_10_most_used_words:
    #print(f"{word}: {count}")
if st.button("See ZD Words"):
    result = (occurrences.most_common(50))
    output_message = f"""
    Top used words: {most_used_words}
    """
    st.write(output_message)

