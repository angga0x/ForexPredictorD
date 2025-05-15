"""
Module for retrieving economic calendar data via web scraping.
This module fetches economic event data that can impact forex markets.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import trafilatura
import re
from bs4 import BeautifulSoup
import requests
from io import StringIO

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_investing_calendar(days=7):
    """
    Fetch economic calendar data from Investing.com.
    
    Args:
        days (int, optional): Number of days to look ahead. Default is 7.
        
    Returns:
        pd.DataFrame: DataFrame with economic events or empty DataFrame if scraping fails
    """
    try:
        # Calculate date range
        today = datetime.now()
        end_date = today + timedelta(days=days)
        
        # Format dates for the URL
        today_str = today.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Investing.com economic calendar URL
        url = f"https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
        
        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://www.investing.com/economic-calendar/'
        }
        
        # POST request data
        data = {
            'dateFrom': today_str,
            'dateTo': end_date_str,
            'timeZone': '0',
            'timeFilter': 'timeRemain',
            'currentTab': 'custom',
            'limit_from': '0'
        }
        
        # Send the request
        response = requests.post(url, headers=headers, data=data)
        
        # Alternative: If the POST request doesn't work, try a simpler GET request
        if response.status_code != 200:
            logger.warning("POST request failed, trying backup method...")
            fallback_url = "https://www.investing.com/economic-calendar/"
            response = requests.get(fallback_url, headers={'User-Agent': headers['User-Agent']})
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the table data
            events = []
            
            # Find all event rows
            rows = soup.select('tr.js-event-item')
            
            for row in rows:
                try:
                    # Extract event details
                    time_element = row.select_one('td.time')
                    country_element = row.select_one('td.flagCur span')
                    event_element = row.select_one('td.event')
                    impact_element = row.select_one('td.sentiment')
                    actual_element = row.select_one('td.act')
                    forecast_element = row.select_one('td.fore')
                    previous_element = row.select_one('td.prev')
                    
                    # Get text from elements if they exist
                    time = time_element.text.strip() if time_element else ""
                    country = country_element.get('title', "") if country_element else ""
                    event = event_element.text.strip() if event_element else ""
                    impact = len(impact_element.select('i.grayFullBullishIcon')) if impact_element else 0
                    actual = actual_element.text.strip() if actual_element else ""
                    forecast = forecast_element.text.strip() if forecast_element else ""
                    previous = previous_element.text.strip() if previous_element else ""
                    
                    # Create event dictionary
                    event_dict = {
                        'time': time,
                        'country': country,
                        'event': event,
                        'impact': impact,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous
                    }
                    
                    events.append(event_dict)
                except Exception as e:
                    logger.error(f"Error parsing row: {str(e)}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(events)
            
            logger.info(f"Successfully retrieved {len(df)} economic events")
            return df
        else:
            logger.error(f"Failed to retrieve data: {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {str(e)}")
        return pd.DataFrame()

def fetch_forexfactory_calendar(days=7):
    """
    Fetch economic calendar data from ForexFactory.
    
    Args:
        days (int, optional): Number of days to look ahead. Default is 7.
        
    Returns:
        pd.DataFrame: DataFrame with economic events or empty DataFrame if scraping fails
    """
    try:
        # ForexFactory calendar URL
        url = "https://www.forexfactory.com/calendar"
        
        # Send GET request
        response = requests.get(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        
        if response.status_code == 200:
            # Get the main content
            content = response.text
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract events from the calendar table
            calendar_table = soup.select_one('table.calendar__table')
            
            if not calendar_table:
                logger.error("Calendar table not found in the HTML")
                return pd.DataFrame()
            
            # Extract rows
            rows = calendar_table.select('tr.calendar__row')
            
            events = []
            current_date = None
            
            for row in rows:
                try:
                    # Check if this is a date row
                    date_cell = row.select_one('td.calendar__date')
                    if date_cell and date_cell.text.strip():
                        date_text = date_cell.text.strip()
                        try:
                            # Parse the date (format may vary)
                            current_date = date_text
                        except:
                            pass
                        continue
                    
                    # Extract event details
                    time_cell = row.select_one('td.calendar__time')
                    currency_cell = row.select_one('td.calendar__currency')
                    impact_cell = row.select_one('td.calendar__impact')
                    event_cell = row.select_one('td.calendar__event')
                    actual_cell = row.select_one('td.calendar__actual')
                    forecast_cell = row.select_one('td.calendar__forecast')
                    previous_cell = row.select_one('td.calendar__previous')
                    
                    # Get text if elements exist
                    time = time_cell.text.strip() if time_cell else ""
                    currency = currency_cell.text.strip() if currency_cell else ""
                    event_name = event_cell.text.strip() if event_cell else ""
                    
                    # Determine impact level from the impact icon class
                    impact = 0
                    if impact_cell:
                        impact_span = impact_cell.select_one('span')
                        if impact_span:
                            try:
                                impact_class = impact_span.get('class')
                                if impact_class is None:
                                    impact_class_str = ""
                                elif isinstance(impact_class, list):
                                    impact_class_str = ' '.join(impact_class)
                                else:
                                    impact_class_str = str(impact_class)
                            except:
                                impact_class_str = ""
                                
                            if 'high' in impact_class_str:
                                impact = 3
                            elif 'medium' in impact_class_str:
                                impact = 2
                            elif 'low' in impact_class_str:
                                impact = 1
                    
                    actual = actual_cell.text.strip() if actual_cell else ""
                    forecast = forecast_cell.text.strip() if forecast_cell else ""
                    previous = previous_cell.text.strip() if previous_cell else ""
                    
                    # Create event dictionary
                    event_dict = {
                        'date': current_date,
                        'time': time,
                        'currency': currency,
                        'event': event_name,
                        'impact': impact,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous
                    }
                    
                    events.append(event_dict)
                except Exception as e:
                    logger.error(f"Error parsing row: {str(e)}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(events)
            
            # Filter to keep only future events within specified days
            if not df.empty and 'date' in df.columns:
                # Keep only relevant days if we can parse the dates
                today = datetime.now().date()
                end_date = today + timedelta(days=days)
                
                # Filter based on the date if possible
                try:
                    # Attempt to convert date strings to datetime objects
                    # This is a simplified approach; might need adjustments based on actual format
                    df = df[df['date'].apply(lambda x: today <= pd.to_datetime(x, errors='coerce').date() <= end_date)]
                except:
                    # If date parsing fails, keep all rows
                    pass
            
            logger.info(f"Successfully retrieved {len(df)} economic events")
            return df
        else:
            logger.error(f"Failed to retrieve data: {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error fetching ForexFactory calendar: {str(e)}")
        return pd.DataFrame()

def get_economic_calendar(days=7, sources=None):
    """
    Get economic calendar data from multiple sources.
    
    Args:
        days (int, optional): Number of days to look ahead. Default is 7.
        sources (list, optional): List of sources to use. Default is ['forexfactory', 'investing'].
        
    Returns:
        dict: Dictionary with calendar data from each source
    """
    if sources is None:
        sources = ['forexfactory', 'investing']
    
    results = {}
    
    # Try to fetch from each source
    if 'investing' in sources:
        investing_df = fetch_investing_calendar(days)
        if not investing_df.empty:
            results['investing'] = investing_df
    
    if 'forexfactory' in sources:
        forexfactory_df = fetch_forexfactory_calendar(days)
        if not forexfactory_df.empty:
            results['forexfactory'] = forexfactory_df
    
    # Add a fallback if all sources fail
    if not results:
        logger.warning("All sources failed, using trafilatura as fallback")
        
        # Use trafilatura to extract text content from ForexFactory
        try:
            url = "https://www.forexfactory.com/calendar"
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            
            if text:
                logger.info("Successfully retrieved text content as fallback")
                
                # Create a simple DataFrame with the raw text
                results['text_content'] = pd.DataFrame({
                    'raw_content': [text],
                    'source': ['ForexFactory'],
                    'extraction_date': [datetime.now().strftime('%Y-%m-%d')]
                })
        except Exception as e:
            logger.error(f"Fallback extraction failed: {str(e)}")
    
    return results