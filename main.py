from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import joblib

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_model')
tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')

# Load the pre-trained LDA model (trained with 14 topics) and vectorizer
lda = joblib.load('lda_model.pkl')  # Updated model filename
vectorizer = joblib.load('vectorizer.pkl')  # Updated vectorizer filename

# Define a class for BERT-based nuanced sentiment analysis
class BertNuancedSentimentAnalyzer:
    def __init__(self, model, tokenizer, emotion_labels):
        self.model = model
        self.tokenizer = tokenizer
        self.emotion_labels = emotion_labels

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)
        probabilities = probabilities.detach().numpy().flatten()  # Convert to a flat numpy array

        # Combine emotions with their corresponding probabilities
        emotion_percentages = [(self.emotion_labels[i], prob * 100) for i, prob in enumerate(probabilities)]

        # Sort emotions by probabilities in descending order and take the top 3
        top_3_emotions = sorted(emotion_percentages, key=lambda x: x[1], reverse=True)[:3]

        return top_3_emotions

# Emotion labels for BERT sentiment analysis
unique_emotions = [
    'empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
    'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger'
]

# Define a list of topic labels for the 14-topic LDA model
topic_labels = [
    'Daily Life & Fun Activities', 'Food, Reading & Leisure', 'Parties & Social Events',
    'Social Media & Followers', 'Optimism & Night Plans', 'Friendship & Daily Feelings',
    'Work, Home & General Thoughts', 'Entertainment & Social Updates', 'Love, Sleep, & Emotions',
    'Dislikes & Summer Activities', 'Help, Plans, & Watching', 'New Beginnings & Reflection',
    'Cars, Coffee, & Leisure', 'Weekend Plans & Tomorrow'
]

# Instantiate the BERT sentiment analyzer
bert_analyzer = BertNuancedSentimentAnalyzer(model, tokenizer, unique_emotions)

# Function to predict the topics using LDA and return multiple topics
def predict_topics(text, vectorizer, lda_model, threshold=0.1, top_n=5):
    text_vectorized = vectorizer.transform([text])
    topic_distribution = lda_model.transform(text_vectorized)[0]  # Get the topic distribution for the tweet

    # Find the indices of the top topics
    top_topic_indices = topic_distribution.argsort()[-top_n:][::-1]  # Top N topics
    selected_topics = [(i, topic_distribution[i]) for i in top_topic_indices if topic_distribution[i] > threshold]

    return selected_topics

# Function to analyze a tweet
def analyze_tweet(tweet):
    # Get nuanced sentiment from BERT model
    top_3_emotions = bert_analyzer.predict_sentiment(tweet)

    # Get topics from LDA model
    topic_probs = predict_topics(tweet, vectorizer, lda)

    return top_3_emotions, topic_probs

# Continuously take tweet input from the user
while True:
    tweet = input("Enter a tweet (or type 'exit' to stop): ")

    if tweet.lower() == 'exit':
        print("Exiting tweet analysis...")
        break

    # Perform analysis
    top_3_emotions, topic_probs = analyze_tweet(tweet)

    # Display top 3 nuanced sentiments
    print("Top 3 Emotions:")
    for emotion, percent in top_3_emotions:
        print(f"  {emotion}: {percent:.2f}%")

    # Display multiple topics and their probabilities
    print("Topics:")
    for idx, prob in topic_probs:
        percent = prob * 100
        print(f"  Topic {idx}: {topic_labels[idx]} ({percent:.2f}%)")

    print("\n")
