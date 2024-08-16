
# Stock Sentiment Analysis

Based on sentiment data, this project predicts whether a stock is bullish or bearish. The data is sourced from [Benzinga. com](https://www.benzinga.com/), a financial news source. The data was scraped using tools available [here](https://github.com/miguelaenlle/Scraping-Tools-Benzinga).

## Table of Contents
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview

This project uses sentiment analysis to determine the market sentiment (bullish or bearish) of stocks based on textual data. The analysis is carried out using a Logistic Regression model trained on preprocessed sentiment data.

## Data Source

The data is sourced from Benzinga.com, which provides financial news and insights. The scraping tool used to retrieve the data can be found [here](https://github.com/miguelaenlle/Scraping-Tools-Benzinga).

## Installation

To run this project, you'll need to install the following Python packages:

- `nltk`
- `pandas`
- `numpy`
- `scikit-learn`

You can install these packages using pip:

```bash
pip install nltk pandas numpy scikit-learn
```

## Usage

1. **Upload Data**: Upload the CSV file (`Sentiment_Stock_data.csv`) containing the sentiment data to the Colab environment.

2. **Run the Notebook**: Execute each cell sequentially in the notebook to preprocess the data, train the model, and make predictions.

3. **Model Training**: The Logistic Regression model is trained using the preprocessed sentiment data. The model learns to classify sentences as indicative of either a bullish or bearish sentiment.

4. **Prediction**: The trained model can predict the sentiment of new sentences, determining whether they indicate a bullish or bearish market sentiment.

## Model

The project uses a Logistic Regression model to classify sentiment data. The model is trained on features extracted from the text data using a frequency dictionary of words associated with positive and negative sentiments.

### Feature Engineering

- **process_stock_sentence**: Preprocesses the text data, including removing URLs, stock symbols, and stop words.
- **build_stock_freqs**: Builds a frequency dictionary of words labeled by sentiment.
- **extract_features**: Converts preprocessed text into feature vectors for model training.

### Training

The model is trained using the preprocessed features and sentiment labels. The Logistic Regression model is used for its simplicity and interpretability in binary classification tasks.

## Results

The model's performance is evaluated using accuracy. The model achieved an accuracy score of **53.03%** on the test data. This score indicates the proportion of correct predictions made by the model.

### Example Prediction

For the example sentence "The company reported a significant increase in revenue.", the model predicted the sentiment as "Bearish". This prediction might be counterintuitive, given the positive nature of the sentence. It suggests that the model may require further refinement and additional data to accurately distinguish between bullish and bearish sentiments.

### Interpretation

1. **Model Accuracy**: An accuracy of 53.03% suggests that the model correctly classified the sentiment of the test sentences slightly better than random guessing (which would be around 50% for a binary classification). This indicates that the model may need further tuning or additional features to improve its performance.

2. **Prediction for Example Sentence**: The model predicted "Bearish" for the example sentence. This could indicate that the model's training data may contain biases or that the model has not learned to distinguish contextually positive financial statements accurately.

### Next Steps

To improve the model's performance, consider the following steps:

- **Data Quality**: Ensure the data is accurately labeled and covers a diverse range of sentiments.
- **Feature Engineering**: Experiment with additional features, such as TF-IDF, word embeddings, or sentiment-specific keywords.
- **Model Tuning**: Experiment with different models, hyperparameters, and regularization techniques.
- **Data Augmentation**: Increase the size of the dataset by including more data points or using data augmentation techniques.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

