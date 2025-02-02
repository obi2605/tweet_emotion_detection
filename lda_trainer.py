import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import nltk
import joblib
from tqdm import tqdm
import re

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Encompassing functions
def clean_tweet(tweet):
    tweet = re.sub(r"http|Sr|www|Sr", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'|@|w+||#', '', tweet)
    tweet = tweet.lower().strip()
    return tweet

def preprocess_for_lda(tweets):
    vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    tweet_term_matrix = vectorizer.fit_transform(tweets)
    return tweet_term_matrix, vectorizer

def extract_topics_as_words(lda_model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_top_words - 1:-1]]
        topics.append(top_words)
    return topics

def find_optimal_topics(corpus, dictionary, tweet_term_matrix, vectorizer, tweets_tokenized, min_topics=10, max_topics=30, n_top_words=10):
    best_coherence = -1
    best_n_topics = min_topics

    for n_topics in tqdm(range(min_topics, max_topics + 1), desc="Finding Optimal Topics"):
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_model.fit(tweet_term_matrix)
        topics_as_words = extract_topics_as_words(lda_model, vectorizer, n_top_words)
        coherence_model = CoherenceModel(topics=topics_as_words, dictionary=dictionary, texts=tweets_tokenized, coherence='c_v')
        coherence = coherence_model.get_coherence()
        if coherence > best_coherence:
            best_coherence = coherence
            best_n_topics = n_topics

    return best_n_topics

# Main function
if __name__ == '__main__':
    # Load datasets
    kaggle_dataset = pd.read_csv('sentiment140.csv', encoding='ISO-8859-1')
    crowdflower_dataset = pd.read_csv('text_emotion.csv')

    # Rename Kaggle dataset columns for clarity
    kaggle_dataset.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']

    # Combine the tweets from both datasets
    tweets = pd.concat([kaggle_dataset['text'], crowdflower_dataset['content']], ignore_index=True)

    # Preprocess tweets: Clean and remove stopwords
    tweets = tweets.apply(clean_tweet)
    stop_words = set(stopwords.words('english'))
    tweets = tweets.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Preprocess tweets and train LDA
    tweet_term_matrix, vectorizer = preprocess_for_lda(tweets)

    # Convert tweet-term_matrix into a format Gensim can use
    tweets_tokenized = [tweet.split() for tweet in tweets]
    dictionary = corpora.Dictionary(tweets_tokenized)
    corpus = [dictionary.doc2bow(text) for text in tweets_tokenized]

    # Find optimal number of topics with progress bar
    optimal_n_topics = find_optimal_topics(corpus, dictionary, tweet_term_matrix, vectorizer, tweets_tokenized)

    # Train the final LDA model with optimal number of topics
    lda = LatentDirichletAllocation(n_components=optimal_n_topics, random_state=42)

    # Adding indm progress bar for LDA fitting
    with tqdm(total=optimal_n_topics, desc="Training LDA") as pbar:
        for i in range(optimal_n_topics):
            lda.partial_fit(tweet_term_matrix)
            pbar.update(1)

    # Display topics
    def display_topics(model, vectorizer, top_n_words):
        terms = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(model.components_):
            print(f"Topic {idx}: {', '.join([terms[i] for i in topic.argsort()[-top_n_words:]])}")

    display_topics(lda, vectorizer, 10)

    # Save models
    joblib.dump(lda, 'lda_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(dictionary, 'dictionary.pkl')

    print("Model trained and saved!")