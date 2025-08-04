import asyncio
import aiohttp
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
import redis
import threading
import time
from queue import Queue
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from logging_config import log_error, performance_monitor

class MarketDataFeed:
    """Real-time market data feed with multiple sources"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.data_cache = {}
        self.subscribers = []
        self.is_running = False
        self.data_queue = Queue()
        
        # API keys (should be in environment variables)
        self.alpha_vantage_key = None  # Set your Alpha Vantage API key
        self.polygon_key = None        # Set your Polygon API key
        
        # Data sources
        self.sources = {
            'yfinance': self._fetch_yfinance_data,
            'alpha_vantage': self._fetch_alpha_vantage_data,
            'polygon': self._fetch_polygon_data
        }
    
    def add_subscriber(self, callback: Callable):
        """Add a subscriber for real-time data updates"""
        self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable):
        """Remove a subscriber"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, data: Dict):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                log_error("subscriber_error", f"Error in subscriber callback: {str(e)}")
    
    async def start_feed(self, symbols: List[str], interval: int = 60):
        """Start the real-time data feed"""
        self.is_running = True
        self.symbols = symbols
        
        # Start data collection threads
        threads = []
        for symbol in symbols:
            thread = threading.Thread(
                target=self._collect_data_loop,
                args=(symbol, interval)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Start data processing thread
        processing_thread = threading.Thread(target=self._process_data_loop)
        processing_thread.daemon = True
        processing_thread.start()
        
        logging.info(f"Started market data feed for {len(symbols)} symbols")
        
        return threads
    
    def stop_feed(self):
        """Stop the real-time data feed"""
        self.is_running = False
        logging.info("Stopped market data feed")
    
    def _collect_data_loop(self, symbol: str, interval: int):
        """Collect data for a specific symbol"""
        while self.is_running:
            try:
                performance_monitor.start_timer(f'data_collection_{symbol}')
                
                # Fetch data from multiple sources
                data = self._fetch_market_data(symbol)
                
                if data:
                    # Add to queue for processing
                    self.data_queue.put({
                        'symbol': symbol,
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    })
                
                performance_monitor.end_timer(f'data_collection_{symbol}')
                
                # Wait for next interval
                time.sleep(interval)
                
            except Exception as e:
                log_error("data_collection_error", f"Error collecting data for {symbol}: {str(e)}")
                time.sleep(interval)
    
    def _process_data_loop(self):
        """Process data from the queue"""
        while self.is_running:
            try:
                if not self.data_queue.empty():
                    item = self.data_queue.get()
                    
                    # Process and cache data
                    processed_data = self._process_market_data(item)
                    
                    # Cache in Redis
                    self._cache_data(item['symbol'], processed_data)
                    
                    # Notify subscribers
                    self._notify_subscribers(processed_data)
                    
                    self.data_queue.task_done()
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                log_error("data_processing_error", f"Error processing data: {str(e)}")
    
    def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data from multiple sources"""
        data = {}
        
        # Try different data sources
        for source_name, source_func in self.sources.items():
            try:
                source_data = source_func(symbol)
                if source_data:
                    data[source_name] = source_data
            except Exception as e:
                log_error("data_source_error", f"Error fetching from {source_name}: {str(e)}")
        
        return data if data else None
    
    def _fetch_yfinance_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get real-time quote
            quote = ticker.history(period="1d", interval="1m")
            
            if not quote.empty:
                latest = quote.iloc[-1]
                return {
                    'price': float(latest['Close']),
                    'volume': int(latest['Volume']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'open': float(latest['Open']),
                    'timestamp': latest.name.isoformat(),
                    'source': 'yfinance'
                }
        except Exception as e:
            log_error("yfinance_error", f"Error fetching Yahoo Finance data: {str(e)}")
        
        return None
    
    def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return None
        
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            data, meta_data = ts.get_quote_endpoint(symbol)
            
            if not data.empty:
                return {
                    'price': float(data['05. price']),
                    'volume': int(data['06. volume']),
                    'high': float(data['03. high']),
                    'low': float(data['04. low']),
                    'open': float(data['02. open']),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'alpha_vantage'
                }
        except Exception as e:
            log_error("alpha_vantage_error", f"Error fetching Alpha Vantage data: {str(e)}")
        
        return None
    
    def _fetch_polygon_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Polygon.io"""
        if not self.polygon_key:
            return None
        
        try:
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            params = {'apikey': self.polygon_key}
            
            # This would need to be implemented with proper async HTTP client
            # For now, return None
            return None
            
        except Exception as e:
            log_error("polygon_error", f"Error fetching Polygon data: {str(e)}")
        
        return None
    
    def _process_market_data(self, item: Dict) -> Dict:
        """Process and aggregate market data"""
        symbol = item['symbol']
        raw_data = item['data']
        
        # Aggregate data from multiple sources
        processed_data = {
            'symbol': symbol,
            'timestamp': item['timestamp'],
            'price': None,
            'volume': None,
            'high': None,
            'low': None,
            'open': None,
            'sources': list(raw_data.keys()),
            'confidence': len(raw_data) / len(self.sources)
        }
        
        # Calculate weighted average prices from multiple sources
        prices = []
        volumes = []
        highs = []
        lows = []
        opens = []
        
        for source_data in raw_data.values():
            if source_data.get('price'):
                prices.append(source_data['price'])
            if source_data.get('volume'):
                volumes.append(source_data['volume'])
            if source_data.get('high'):
                highs.append(source_data['high'])
            if source_data.get('low'):
                lows.append(source_data['low'])
            if source_data.get('open'):
                opens.append(source_data['open'])
        
        # Set aggregated values
        if prices:
            processed_data['price'] = np.mean(prices)
        if volumes:
            processed_data['volume'] = int(np.mean(volumes))
        if highs:
            processed_data['high'] = np.mean(highs)
        if lows:
            processed_data['low'] = np.mean(lows)
        if opens:
            processed_data['open'] = np.mean(opens)
        
        return processed_data
    
    def _cache_data(self, symbol: str, data: Dict):
        """Cache data in Redis"""
        try:
            # Cache latest data
            self.redis_client.setex(
                f"market_data:{symbol}:latest",
                300,  # 5 minutes TTL
                json.dumps(data)
            )
            
            # Add to time series
            self.redis_client.zadd(
                f"market_data:{symbol}:history",
                {json.dumps(data): time.time()}
            )
            
            # Keep only last 1000 data points
            self.redis_client.zremrangebyrank(
                f"market_data:{symbol}:history",
                0, -1001
            )
            
        except Exception as e:
            log_error("cache_error", f"Error caching data for {symbol}: {str(e)}")
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get latest data for a symbol"""
        try:
            cached_data = self.redis_client.get(f"market_data:{symbol}:latest")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            log_error("cache_read_error", f"Error reading cached data for {symbol}: {str(e)}")
        
        return None
    
    def get_historical_data(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get historical data for a symbol"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            data_points = self.redis_client.zrangebyscore(
                f"market_data:{symbol}:history",
                cutoff_time,
                '+inf'
            )
            
            return [json.loads(point) for point in data_points]
            
        except Exception as e:
            log_error("historical_data_error", f"Error fetching historical data for {symbol}: {str(e)}")
            return []

class WebSocketFeed:
    """WebSocket-based real-time data feed"""
    
    def __init__(self, url: str, api_key: str = None):
        self.url = url
        self.api_key = api_key
        self.ws = None
        self.is_connected = False
        self.subscribers = []
    
    async def connect(self):
        """Connect to WebSocket feed"""
        try:
            self.ws = await websockets.connect(self.url)
            self.is_connected = True
            logging.info(f"Connected to WebSocket feed: {self.url}")
            
            # Start listening for messages
            asyncio.create_task(self._listen())
            
        except Exception as e:
            log_error("websocket_connect_error", f"Error connecting to WebSocket: {str(e)}")
    
    async def disconnect(self):
        """Disconnect from WebSocket feed"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logging.info("Disconnected from WebSocket feed")
    
    async def _listen(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    self._notify_subscribers(data)
                except json.JSONDecodeError:
                    log_error("websocket_parse_error", "Invalid JSON received from WebSocket")
                except Exception as e:
                    log_error("websocket_message_error", f"Error processing WebSocket message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logging.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            log_error("websocket_listen_error", f"Error in WebSocket listener: {str(e)}")
            self.is_connected = False
    
    def add_subscriber(self, callback: Callable):
        """Add a subscriber for WebSocket data"""
        self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable):
        """Remove a subscriber"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, data: Dict):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                log_error("websocket_subscriber_error", f"Error in WebSocket subscriber: {str(e)}")

class DataAggregator:
    """Aggregate data from multiple sources"""
    
    def __init__(self):
        self.feeds = {}
        self.aggregated_data = {}
    
    def add_feed(self, name: str, feed: MarketDataFeed):
        """Add a data feed"""
        self.feeds[name] = feed
    
    def add_websocket_feed(self, name: str, feed: WebSocketFeed):
        """Add a WebSocket feed"""
        self.feeds[name] = feed
    
    def get_aggregated_data(self, symbol: str) -> Dict:
        """Get aggregated data for a symbol"""
        aggregated = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'consensus': {}
        }
        
        # Collect data from all feeds
        for feed_name, feed in self.feeds.items():
            if hasattr(feed, 'get_latest_data'):
                data = feed.get_latest_data(symbol)
                if data:
                    aggregated['sources'][feed_name] = data
        
        # Calculate consensus values
        if aggregated['sources']:
            prices = [data['price'] for data in aggregated['sources'].values() if data.get('price')]
            volumes = [data['volume'] for data in aggregated['sources'].values() if data.get('volume')]
            
            if prices:
                aggregated['consensus']['price'] = np.mean(prices)
                aggregated['consensus']['price_std'] = np.std(prices)
            if volumes:
                aggregated['consensus']['volume'] = int(np.mean(volumes))
        
        return aggregated
    
    def start_all_feeds(self, symbols: List[str]):
        """Start all data feeds"""
        for feed_name, feed in self.feeds.items():
            if hasattr(feed, 'start_feed'):
                asyncio.create_task(feed.start_feed(symbols))
            elif hasattr(feed, 'connect'):
                asyncio.create_task(feed.connect())
    
    def stop_all_feeds(self):
        """Stop all data feeds"""
        for feed_name, feed in self.feeds.items():
            if hasattr(feed, 'stop_feed'):
                feed.stop_feed()
            elif hasattr(feed, 'disconnect'):
                asyncio.create_task(feed.disconnect())

# Global data aggregator instance
data_aggregator = DataAggregator()

# Example usage
async def setup_market_data():
    """Setup market data feeds"""
    # Create market data feed
    market_feed = MarketDataFeed()
    data_aggregator.add_feed('market', market_feed)
    
    # Start feeds
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    data_aggregator.start_all_feeds(symbols)
    
    return data_aggregator 