"""
Module for retrieving and analyzing news using the NewsAPI.
This module fetches financial and forex news and performs sentiment analysis.
"""

import logging
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import pandas as pd
import numpy as np
from config import NEWS_API_KEY

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def get_forex_news(currency_pair, days=7, language='en'):
    """
    Fetch news articles related to a specific forex currency pair.
    
    Args:
        currency_pair (str): Currency pair in format like 'EURUSD'
        days (int, optional): Number of days to look back for news. Default is 7.
        language (str, optional): Language of news articles. Default is 'en'.
        
    Returns:
        list: List of news articles or empty list if API call fails
    """
    try:
        # Split the currency pair into individual currencies (e.g., EURUSD -> EUR, USD)
        if '=' in currency_pair:
            # Handle YFinance format (EURUSD=X)
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:6]
        else:
            # Handle standard format (EURUSD)
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[3:]
        
        # Prepare search queries for the currency pair
        search_query = f"({base_currency} AND {quote_currency}) OR forex OR 'foreign exchange'"
        
        # Calculate the date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Format dates for the API
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        # Fetch news from the API
        all_articles = newsapi.get_everything(
            q=search_query,
            from_param=from_date_str,
            to=to_date_str,
            language=language,
            sort_by='publishedAt'
        )
        
        logger.info(f"Retrieved {len(all_articles.get('articles', []))} news articles for {currency_pair}")
        return all_articles.get('articles', [])
    
    except Exception as e:
        logger.error(f"Error fetching news for {currency_pair}: {str(e)}")
        return []

def simple_sentiment_analysis(text):
    """
    Perform a simple sentiment analysis on a text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Sentiment score (-1 to 1, where -1 is very negative and 1 is very positive)
    """
    # This is a simple keyword-based approach
    # In a production environment, you would use a more sophisticated model
    
    positive_words = [
        'gain', 'gains', 'up', 'rise', 'rises', 'rising', 'risen', 'rose', 'bullish', 
        'strong', 'stronger', 'strength', 'strengthen', 'positive', 'optimistic', 
        'rally', 'rallies', 'rallied', 'recovery', 'recoveries', 'recovered', 
        'growth', 'grow', 'growing', 'grew', 'grown', 'high', 'higher', 'highest', 
        'surge', 'surged', 'surging', 'jump', 'jumped', 'jumping', 'climb', 'climbed', 
        'climbing', 'boom', 'booming', 'boomed', 'success', 'successful'
    ]
    
    negative_words = [
        'loss', 'losses', 'down', 'fall', 'falls', 'falling', 'fell', 'fallen', 
        'bearish', 'weak', 'weaker', 'weakness', 'weakening', 'negative', 'pessimistic', 
        'drop', 'drops', 'dropped', 'dropping', 'slump', 'slumps', 'slumped', 'slumping', 
        'decline', 'declines', 'declined', 'declining', 'decrease', 'decreases', 'decreased', 
        'decreasing', 'low', 'lower', 'lowest', 'plunge', 'plunges', 'plunged', 'plunging', 
        'crash', 'crashes', 'crashed', 'crashing', 'tumble', 'tumbles', 'tumbled', 'tumbling', 
        'slide', 'slides', 'slid', 'sliding', 'sink', 'sinks', 'sank', 'sunk', 'sinking'
    ]
    
    # Convert text to lowercase
    text = text.lower()
    
    # Count occurrences of positive and negative words
    positive_count = sum(1 for word in positive_words if f" {word} " in f" {text} ")
    negative_count = sum(1 for word in negative_words if f" {word} " in f" {text} ")
    
    # Calculate sentiment score
    total_count = positive_count + negative_count
    if total_count == 0:
        return 0.0  # Neutral sentiment if no positive or negative words found
    
    sentiment_score = (positive_count - negative_count) / total_count
    return sentiment_score

def analyze_news_sentiment(articles):
    """
    Analyze the sentiment of a list of news articles.
    
    Args:
        articles (list): List of news articles from NewsAPI
        
    Returns:
        dict: Dictionary with sentiment analysis results
    """
    if not articles:
        return {
            'overall_sentiment': 0,  # Neutral
            'sentiment_scores': [],
            'articles_df': pd.DataFrame(),
            'article_count': 0
        }
    
    # Create lists to store data
    titles = []
    sources = []
    published_dates = []
    sentiment_scores = []
    
    # Analyze each article
    for article in articles:
        title = article.get('title', '')
        source = article.get('source', {}).get('name', 'Unknown')
        published_at = article.get('publishedAt', '')
        description = article.get('description', '')
        
        # Combine title and description for better sentiment analysis
        content = f"{title} {description}"
        
        # Calculate sentiment score
        sentiment = simple_sentiment_analysis(content)
        
        # Store data
        titles.append(title)
        sources.append(source)
        published_dates.append(published_at)
        sentiment_scores.append(sentiment)
    
    # Create DataFrame with the news data
    news_df = pd.DataFrame({
        'title': titles,
        'source': sources,
        'published_at': published_dates,
        'sentiment': sentiment_scores
    })
    
    # Calculate overall sentiment
    overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    
    return {
        'overall_sentiment': overall_sentiment,
        'sentiment_scores': sentiment_scores,
        'articles_df': news_df,
        'article_count': len(articles)
    }

def get_news_sentiment_for_pair(currency_pair, days=7):
    """
    Complete workflow to get news and analyze sentiment for a currency pair.
    
    Args:
        currency_pair (str): Currency pair in format like 'EURUSD=X'
        days (int, optional): Number of days to look back for news. Default is 7.
        
    Returns:
        dict: Dictionary with sentiment analysis results
    """
    # Get news articles
    articles = get_forex_news(currency_pair, days)
    
    # Analyze sentiment
    sentiment_results = analyze_news_sentiment(articles)
    
    return sentiment_results