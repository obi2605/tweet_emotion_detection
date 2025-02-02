import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import re
import nltk
from torch.utils.data import Dataset

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the Crowdflower dataset
crowdflower_dataset = pd.read_csv('text_emotion.csv')

# Use only the content and sentiment columns from the Crowdflower dataset
tweets = crowdflower_dataset['content']
emotions = crowdflower_dataset['sentiment']  # Use the sentiment column from the Crowdflower dataset

# Preprocess tweets: Remove special characters, URLs, and lowercase all text
def clean_tweet(tweet):
    tweet = re.sub(r"http|$+|www|$+|https|$+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'|@|w+||#', '', tweet)
    tweet = tweet.lower().strip()
    return tweet

tweets = tweets.apply(clean_tweet)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tweets = tweets.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Prepare the emotion labels from the Crowdflower dataset
unique_emotions = emotions.unique()

# Create a PyTorch dataset for tokenized tweets and labels
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# BERT Model Training (Multi-class classification for nuanced sentiment)
class BertNuancedSentimentAnalyzer:
    def __init__(self, emotion_labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(emotion_labels))
        self.emotion_labels = emotion_labels

    def tokenize_and_encode(self, tweets):
        return self.tokenizer(list(tweets), padding=True, truncation=True, max_length=128, return_tensors="pt")

    def train_model(self, tokenized_data, labels):
        # Create a PyTorch dataset
        dataset = TweetDataset(tokenized_data, labels)

        # Training setup
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,
            per_device_train_batch_size=8,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            compute_metrics=None
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained('./bert_model')
        self.tokenizer.save_pretrained('./bert_tokenizer')

# Initialize BERT model
bert_analyzer = BertNuancedSentimentAnalyzer(unique_emotions)

# Tokenize and train BERT
tokenized_tweets = bert_analyzer.tokenize_and_encode(crowdflower_dataset['content'])
labels = torch.tensor([list(unique_emotions).index(emotion) for emotion in emotions])

bert_analyzer.train_model(tokenized_tweets, labels)
bert_analyzer.save_model()