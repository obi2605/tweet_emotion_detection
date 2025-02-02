import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
#from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
#import joblib
from tqdm import tqdm

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_model')
tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')

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
        sentiment = torch.argmax(logits, dim=1).item()
        return sentiment

# Emotion labels for BERT sentiment analysis
unique_emotions = [
    'empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise',
    'love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger'
]

# Instantiate the BERT sentiment analyzer
bert_analyzer = BertNuancedSentimentAnalyzer(model, tokenizer, unique_emotions)

# Load the new dataset for evaluation
new_dataset_path = 'further_amplified_text_emotion_10k_4'  # Path to the new dataset
df = pd.read_csv(new_dataset_path)

# Map the sentiment labels to correct indices for BERT classification
emotion_to_index = {label: i for i, label in enumerate(unique_emotions)}
df['label'] = df['sentiment'].map(emotion_to_index)

# Remove rows with NaN labels
df = df.dropna(subset=['label'])

# Function to evaluate the model
def evaluate_model(df):
    true_labels = df['label'].tolist()
    pred_labels = []

    # Predict sentiments for each tweet in the dataset
    for tweet in tqdm(df['content']):
        pred_label = bert_analyzer.predict_sentiment(tweet)
        pred_labels.append(pred_label)

    # Calculate F1 score
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Calculate precision, recall, and accuracy
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Generate classification report
    class_report = classification_report(true_labels, pred_labels, target_names=unique_emotions, labels=range(len(unique_emotions)), zero_division=0)

    return f1, precision, recall, accuracy, conf_matrix, class_report

# Run evaluation on the complete dataset
f1, precision, recall, accuracy, conf_matrix, class_report = evaluate_model(df)

# Print performance metrics
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
