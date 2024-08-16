#Comprehensive Sentiment Analysis: From Emotion Detection to Stock Market Prediction

This project presents a unified approach to sentiment analysis, beginning with emotion detection in textual data using neural networks and extending to predicting stock market trends based on sentiment. By leveraging advanced machine learning techniques, this project showcases the evolution from understanding emotions in sentences to analyzing market sentiments that influence stock trends.

## Table of Contents

- [Project Overview](#project-overview)
- [Evolution of the Project](#evolution-of-the-project)
- [Datasets](#datasets)
- [Tech Stack and Requirements](#tech-stack-and-requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Upload Dataset](#1-upload-dataset)
  - [2. Extract Features](#2-extract-features)
  - [3. Train the Models](#3-train-the-models)
  - [4. Evaluate the Models](#4-evaluate-the-models)
- [Stock Market Sentiment Analysis](#stock-market-sentiment-analysis)
  - [Sentiment Analysis Application](#sentiment-analysis-application)
  - [Supporting Code](#supporting-code)
- [Results and Insights](#results-and-insights)
- [Model Saving and Loading](#model-saving-and-loading)
- [Next Steps](#next-steps)
- [References](#references)

## Project Overview

This project initially focused on detecting emotions in sentences using neural networks, a crucial task in understanding and interpreting human communication. Building on this foundation, the project expanded to apply sentiment analysis in predicting stock market trends by analyzing financial news, social media, and other textual data sources.

## Evolution of the Project

The project began with the goal of understanding emotions in textual data through a neural network-based model. This phase involved:

1. **Emotion Detection**: A Multi-Layer Perceptron (MLP) model was trained to classify text into different emotion categories.
2. **Sentiment Analysis**: The techniques and insights gained from emotion detection were then adapted to sentiment analysis in the financial domain.
3. **Stock Market Prediction**: By analyzing the sentiment of news articles and social media posts, the project evolved to predict bullish or bearish trends in the stock market.

This progression highlights the adaptability of sentiment analysis techniques and their applicability across different domains.

## Datasets

- **Emotion Detection Dataset**: Textual data labeled with emotions such as positive, negative, or neutral.
- **Stock Market Sentiment Dataset**: Financial news articles and social media posts labeled as bullish or bearish.

## Tech Stack and Requirements

- **Programming Language**: Python 3
- **Libraries**:
  - `librosa`
  - `soundfile`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `nltk`
  - `pandas`
  - `TextBlob`
  - `pickle`

## Setup

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/sentiment-analysis-project.git
    cd sentiment-analysis-project
    ```

2. **Set Up a Virtual Environment**:
    ```sh
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install Requirements**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Upload Dataset

Ensure your datasets are uploaded to the appropriate directories.

### 2. Extract Features

Run the feature extraction functions to derive relevant features from the datasets. This includes preprocessing the text data, such as removing URLs, stock symbols, and stop words.

### 3. Train the Models

- **Emotion Detection**: Train the neural network model using the extracted features from the emotion detection dataset.
- **Stock Sentiment Analysis**: Train the Logistic Regression model on preprocessed sentiment data to classify stock market sentiment as bullish or bearish.

### 4. Evaluate the Models

Evaluate the performance of both models using accuracy, precision, recall, F1 score, and other relevant metrics.

## Stock Market Sentiment Analysis

### Sentiment Analysis Application

The sentiment analysis techniques developed for emotion detection were extended to analyze stock market trends. By applying these methods to financial data, the project aims to predict whether a stock is likely to perform well (bullish) or poorly (bearish) based on the sentiment expressed in news articles and other textual data.

### Supporting Code

Below is an example of applying sentiment analysis to stock market data:

```python
import pandas as pd
from textblob import TextBlob

# Load stock market news data
news_data = pd.read_csv('stock_news.csv')

# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    return 'positive' if analysis.sentiment.polarity > 0 else 'negative'

# Apply sentiment analysis
news_data['Sentiment'] = news_data['News'].apply(get_sentiment)

# Display sentiment analysis results
print(news_data.head())
```

## Results and Insights

- **Emotion Detection**: The MLP model effectively classified emotions with a high accuracy, demonstrating the potential of neural networks in understanding textual data.
- **Stock Sentiment Analysis**: The Logistic Regression model achieved a baseline accuracy of 53.03%, indicating that while the model can predict stock sentiment slightly better than random guessing, there is room for improvement.

## Model Saving and Loading

To save the trained models:

```python
import pickle

# Save the emotion detection model
pickle.dump(emotion_model, open('emotion_model.pkl', 'wb'))

# Save the stock sentiment model
pickle.dump(stock_model, open('stock_model.pkl', 'wb'))
```

To load the trained models:

```python
# Load the emotion detection model
emotion_model = pickle.load(open('emotion_model.pkl', 'rb'))

# Load the stock sentiment model
stock_model = pickle.load(open('stock_model.pkl', 'rb'))
```

## Next Steps

1. **Data Augmentation**: Increase the dataset size by incorporating more diverse textual data sources.
2. **Model Tuning**: Experiment with advanced models and hyperparameters to improve prediction accuracy.
3. **Feature Engineering**: Explore additional features such as TF-IDF, word embeddings, or sentiment-specific keywords.
4. **Real-time Sentiment Analysis**: Implement real-time sentiment analysis for live-stock market predictions.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Benzinga Scraping Tools](https://github.com/miguelaenlle/Scraping-Tools-Benzinga)
