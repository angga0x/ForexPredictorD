"""
Module for retrieving economic calendar data via web scraping.
This module fetches economic event data that can impact forex markets.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
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
        
        # More comprehensive headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://www.investing.com/economic-calendar/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'Cookie': 'CONSENT=YES+',  # Add minimal cookie consent
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
        
        # Use a more realistic browser user agent and add more headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        
        # Send GET request with expanded headers
        response = requests.get(url, headers=headers)
        
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

def generate_sample_calendar_data(days=7, currency=None):
    """
    Generate a sample economic calendar with common financial events when web scraping fails.
    This ensures the app shows something useful even when external data sources are unavailable.
    
    Args:
        days (int): Number of days to create events for
        currency (str, optional): Currency code to filter events for (e.g., 'EUR', 'USD')
        
    Returns:
        pd.DataFrame: Sample economic calendar data
    """
    # Get the current date
    today = datetime.now().date()
    
    # Common economic events by currency
    event_types = {
        'USD': [
            'Interest Rate Decision', 'Non-Farm Payrolls', 'Unemployment Rate', 
            'GDP Growth Rate', 'CPI', 'Retail Sales', 'Industrial Production',
            'Consumer Confidence', 'Trade Balance', 'ISM Manufacturing PMI'
        ],
        'EUR': [
            'Interest Rate Decision', 'Unemployment Rate', 'GDP Growth Rate', 
            'CPI', 'Retail Sales', 'Industrial Production', 'Trade Balance',
            'Manufacturing PMI', 'Consumer Confidence', 'Economic Sentiment'
        ],
        'GBP': [
            'Interest Rate Decision', 'Unemployment Rate', 'GDP Growth Rate', 
            'CPI', 'Retail Sales', 'Industrial Production', 'Trade Balance',
            'Manufacturing PMI', 'Construction PMI', 'Services PMI'
        ],
        'JPY': [
            'Interest Rate Decision', 'Unemployment Rate', 'GDP Growth Rate', 
            'CPI', 'Retail Sales', 'Industrial Production', 'Trade Balance',
            'Tankan Manufacturing Index', 'Machine Orders', 'Foreign Bond Investment'
        ],
        'AUD': [
            'Interest Rate Decision', 'Unemployment Rate', 'GDP Growth Rate', 
            'CPI', 'Retail Sales', 'Trade Balance', 'Building Approvals',
            'Business Confidence', 'Consumer Sentiment', 'Home Loans'
        ],
    }
    
    # All currencies if none specified
    all_currencies = list(event_types.keys())
    currencies_to_use = [currency] if currency in event_types else all_currencies
    
    # Create empty lists for data
    dates = []
    times = []
    currencies = []
    events = []
    impacts = []
    
    # Generate events for each day
    for day_offset in range(days):
        event_date = today + timedelta(days=day_offset)
        date_str = event_date.strftime('%Y-%m-%d')
        
        # Generate 1-3 events per currency per day
        for curr in currencies_to_use:
            num_events = min(len(event_types[curr]), 3)
            selected_events = np.random.choice(event_types[curr], num_events, replace=False)
            
            for event in selected_events:
                # Random hour between 8 AM and 6 PM
                hour = np.random.randint(8, 19)
                minute = np.random.choice([0, 15, 30, 45])
                time_str = f"{hour:02d}:{minute:02d}"
                
                # Random impact level (1-3)
                impact = np.random.randint(1, 4)
                
                # Add to lists
                dates.append(date_str)
                times.append(time_str)
                currencies.append(curr)
                events.append(event)
                impacts.append(impact)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'time': times,
        'currency': currencies,
        'event': events,
        'impact': impacts,
    })
    
    # Sort by date and time
    df = df.sort_values(by=['date', 'time'])
    
    return df

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
    scraping_success = False
    
    # Try to fetch from each source
    if 'investing' in sources:
        investing_df = fetch_investing_calendar(days)
        if not investing_df.empty:
            results['investing'] = investing_df
            scraping_success = True
    
    if 'forexfactory' in sources:
        forexfactory_df = fetch_forexfactory_calendar(days)
        if not forexfactory_df.empty:
            results['forexfactory'] = forexfactory_df
            scraping_success = True
    
    # Try fallback text extraction if scraping fails
    if not scraping_success:
        logger.warning("Primary scraping methods failed, trying text extraction fallback")
        
        try:
            # Try with a different URL
            url = "https://www.forexfactory.com/calendar"
            
            # Use requests instead with custom headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Get the page content with requests
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            # Then pass to trafilatura
            downloaded = response.text
            text = trafilatura.extract(downloaded)
            
            if text and len(text) > 100:  # Ensure we got meaningful content
                logger.info("Successfully retrieved text content via trafilatura")
                
                # Create a simple DataFrame with the raw text
                results['text_content'] = pd.DataFrame({
                    'raw_content': [text],
                    'source': ['ForexFactory'],
                    'extraction_date': [datetime.now().strftime('%Y-%m-%d')]
                })
                scraping_success = True
        except Exception as e:
            logger.error(f"Text extraction fallback failed: {str(e)}")
    
    # If all web scraping methods fail, use generated sample data as final fallback
    if not scraping_success:
        logger.warning("All web scraping methods failed, using generated calendar data")
        sample_df = generate_sample_calendar_data(days)
        results['sample_data'] = sample_df
        
        # Add a note that this is sample data
        logger.info(f"Generated {len(sample_df)} sample economic events as fallback")
    
    return results