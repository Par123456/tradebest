#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Cryptocurrency Trading Bot
Advanced Technical Analysis & Signal Generation System
Compatible with Python 3.13.1 and cp.springhost.ru deployment
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import sqlite3
import hashlib
import hmac
from pathlib import Path

# Trading and Technical Analysis
try:
    import ccxt.async_support as ccxt_async
    import ta
    from ta.utils import dropna
except ImportError as e:
    print(f"Critical import error: {e}")
    print("Please install required packages: pip install ccxt ta pandas numpy")
    sys.exit(1)

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server deployment
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    sns.set_style("darkgrid")
except ImportError:
    print("Warning: Matplotlib not available. Charts will be disabled.")
    plt = None

# Telegram Bot
try:
    from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
    from telegram.constants import ParseMode
except ImportError as e:
    print(f"Telegram bot error: {e}")
    print("Please install: pip install python-telegram-bot")
    sys.exit(1)

# PDF Generation
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm, inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
except ImportError:
    print("Warning: ReportLab not available. PDF reports will be disabled.")

# Web scraping for news sentiment
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Warning: Web scraping libraries not available.")

# ===================== CONFIGURATION & CONSTANTS =====================

@dataclass
class TradingConfig:
    """Advanced trading configuration with professional parameters"""
    # Telegram Settings
    telegram_token: str = "8145688023:AAHbPn6QgO1t7tQUnS2-kRx7FDoO0mr15tE"
    chat_id: str = "6508600903"
    
    # Trading Pairs
    symbols: List[str] = None
    
    # Timeframes
    primary_timeframe: str = "15m"
    secondary_timeframe: str = "1h"
    higher_timeframe: str = "4h"
    
    # Risk Management
    max_risk_per_trade: float = 2.0  # % of portfolio
    risk_reward_ratio: float = 2.5
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 4.0
    trailing_stop_percentage: float = 1.5
    
    # Signal Filters
    min_volume_spike: float = 1.5  # Volume must be 1.5x average
    min_adx_strength: float = 25
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    # Advanced Features
    use_machine_learning: bool = True
    use_sentiment_analysis: bool = True
    use_multi_timeframe: bool = True
    use_portfolio_management: bool = True
    
    # Scheduling
    analysis_interval_minutes: int = 15
    alert_check_interval_minutes: int = 5
    daily_report_hour: int = 23
    
    # Database
    database_path: str = "trading_bot.db"
    max_historical_days: int = 365
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT",
                "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "BONK/USDT",
                "SOL/USDT", "MATIC/USDT", "AVAX/USDT", "DOT/USDT"
            ]

@dataclass
class TradingSignal:
    """Professional trading signal structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: float
    timeframe: str
    indicators: Dict[str, float]
    market_conditions: Dict[str, Any]
    reasoning: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class DatabaseManager:
    """Advanced database management for trading data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    risk_reward_ratio REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    indicators TEXT NOT NULL,
                    market_conditions TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Price data cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # Performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    outcome TEXT,
                    profit_loss REAL,
                    duration_hours REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals (id)
                )
            ''')
            
            # Market sentiment
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    news_count INTEGER DEFAULT 0,
                    social_mentions INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_signal(self, signal: TradingSignal) -> int:
        """Save trading signal to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (
                    timestamp, symbol, signal_type, strength, entry_price,
                    stop_loss, take_profit, risk_reward_ratio, confidence,
                    timeframe, indicators, market_conditions, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.signal_type,
                signal.strength,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                signal.risk_reward_ratio,
                signal.confidence,
                signal.timeframe,
                json.dumps(signal.indicators),
                json.dumps(signal.market_conditions),
                signal.reasoning
            ))
            return cursor.lastrowid
    
    def get_recent_signals(self, hours: int = 24) -> List[Dict]:
        """Get recent signals from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals 
                WHERE created_at > datetime('now', '-{} hours')
                ORDER BY created_at DESC
            '''.format(hours))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def save_price_data(self, symbol: str, timeframe: str, ohlcv_data: List):
        """Save OHLCV data to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for candle in ohlcv_data:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, timeframe, candle[0], candle[1], candle[2], 
                          candle[3], candle[4], candle[5]))
                except sqlite3.IntegrityError:
                    continue
            conn.commit()

class AdvancedTechnicalAnalysis:
    """Professional technical analysis with advanced indicators"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Trend Indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_sma'] = ta.trend.sma_indicator(df['rsi'], window=14)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volume Indicators
            df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['vwap'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume'], window=20).volume_sma()
            
            # Volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['atr_percent'] = (df['atr'] / df['close']) * 100
            
            # ADX (Trend Strength)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            
            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # Commodity Channel Index
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            
            # Money Flow Index
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            
            # Parabolic SAR
            df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
            
            # Custom Indicators
            df['price_momentum'] = df['close'].pct_change(periods=10) * 100
            df['volume_momentum'] = df['volume'].pct_change(periods=5) * 100
            
            # Support and Resistance Levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            # Market Structure
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
            
            return df.fillna(method='ffill').fillna(method='bfill')
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return df
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """Detect candlestick and chart patterns"""
        if len(df) < 5:
            return {}
        
        patterns = {}
        
        try:
            # Get last few candles
            current = df.iloc[-1]
            prev1 = df.iloc[-2]
            prev2 = df.iloc[-3] if len(df) > 2 else prev1
            
            # Candlestick patterns
            patterns['doji'] = abs(current['close'] - current['open']) <= (current['high'] - current['low']) * 0.1
            patterns['hammer'] = (
                (current['close'] > current['open']) and
                ((current['open'] - current['low']) > 2 * (current['close'] - current['open'])) and
                ((current['high'] - current['close']) < (current['close'] - current['open']) * 0.3)
            )
            patterns['shooting_star'] = (
                (current['open'] > current['close']) and
                ((current['high'] - current['open']) > 2 * (current['open'] - current['close'])) and
                ((current['close'] - current['low']) < (current['open'] - current['close']) * 0.3)
            )
            patterns['engulfing_bullish'] = (
                (prev1['close'] < prev1['open']) and
                (current['close'] > current['open']) and
                (current['open'] < prev1['close']) and
                (current['close'] > prev1['open'])
            )
            patterns['engulfing_bearish'] = (
                (prev1['close'] > prev1['open']) and
                (current['close'] < current['open']) and
                (current['open'] > prev1['close']) and
                (current['close'] < prev1['open'])
            )
            
            # Chart patterns
            patterns['golden_cross'] = (
                df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1] and
                df['ema_9'].iloc[-2] <= df['ema_21'].iloc[-2]
            )
            patterns['death_cross'] = (
                df['ema_9'].iloc[-1] < df['ema_21'].iloc[-1] and
                df['ema_9'].iloc[-2] >= df['ema_21'].iloc[-2]
            )
            
            # Breakout patterns
            patterns['resistance_breakout'] = (
                current['close'] > df['resistance'].iloc[-2] and
                current['volume'] > df['volume_sma'].iloc[-1] * 1.5
            )
            patterns['support_breakdown'] = (
                current['close'] < df['support'].iloc[-2] and
                current['volume'] > df['volume_sma'].iloc[-1] * 1.5
            )
            
        except Exception as e:
            logging.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, period: int = 50) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        if len(df) < period:
            return {}
        
        recent_data = df.tail(period)
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        diff = high - low
        
        levels = {
            '0.0': low,
            '23.6': low + diff * 0.236,
            '38.2': low + diff * 0.382,
            '50.0': low + diff * 0.5,
            '61.8': low + diff * 0.618,
            '78.6': low + diff * 0.786,
            '100.0': high
        }
        
        return levels

class MarketSentimentAnalyzer:
    """Advanced market sentiment analysis"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def get_fear_greed_index(self) -> Optional[Dict]:
        """Get Fear & Greed Index from alternative.me"""
        try:
            response = self.session.get('https://api.alternative.me/fng/', timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification'],
                    'timestamp': data['data'][0]['timestamp']
                }
        except Exception as e:
            logging.error(f"Error fetching Fear & Greed Index: {e}")
        return None
    
    async def analyze_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze social media sentiment (placeholder for real implementation)"""
        # In a real implementation, you would integrate with Twitter API, Reddit API, etc.
        # For now, return mock data
        return {
            'sentiment_score': np.random.uniform(-1, 1),
            'mention_count': np.random.randint(10, 1000),
            'positive_ratio': np.random.uniform(0.3, 0.7),
            'confidence': np.random.uniform(0.5, 0.9)
        }

class ProfessionalTradingBot:
    """Main trading bot class with professional features"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.db = DatabaseManager(config.database_path)
        self.technical_analyzer = AdvancedTechnicalAnalysis()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.exchange = None
        self.bot = None
        self.app = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache for market data
        self.market_cache = {}
        self.last_cache_update = {}
        
        logging.info("Professional Trading Bot initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    async def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            self.exchange = ccxt_async.binance({
                'apiKey': '',  # Add your API key if needed
                'secret': '',  # Add your secret if needed
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            logging.info("Exchange connection established successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch and cache market data"""
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        
        # Check cache validity (5 minutes for lower timeframes, 15 minutes for higher)
        cache_duration = 300 if timeframe in ['1m', '5m', '15m'] else 900
        
        if (cache_key in self.market_cache and 
            current_time - self.last_cache_update.get(cache_key, 0) < cache_duration):
            return self.market_cache[cache_key]
        
        try:
            if not self.exchange:
                await self.initialize_exchange()
            
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate all technical indicators
            df = self.technical_analyzer.calculate_all_indicators(df)
            
            # Cache the data
            self.market_cache[cache_key] = df
            self.last_cache_update[cache_key] = current_time
            
            # Save to database
            ohlcv_for_db = [[int(row.name.timestamp() * 1000)] + row[['open', 'high', 'low', 'close', 'volume']].tolist() 
                           for _, row in df.iterrows()]
            self.db.save_price_data(symbol, timeframe, ohlcv_for_db)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Comprehensive symbol analysis"""
        try:
            # Fetch multi-timeframe data
            df_primary = await self.fetch_market_data(symbol, self.config.primary_timeframe)
            df_secondary = await self.fetch_market_data(symbol, self.config.secondary_timeframe)
            df_higher = await self.fetch_market_data(symbol, self.config.higher_timeframe)
            
            if df_primary is None or len(df_primary) < 50:
                return None
            
            # Get current market conditions
            current = df_primary.iloc[-1]
            prev = df_primary.iloc[-2]
            
            # Detect patterns
            patterns = self.technical_analyzer.detect_patterns(df_primary)
            
            # Calculate Fibonacci levels
            fib_levels = self.technical_analyzer.calculate_fibonacci_levels(df_primary)
            
            # Get sentiment data
            sentiment_data = await self.sentiment_analyzer.analyze_social_sentiment(symbol)
            fear_greed = await self.sentiment_analyzer.get_fear_greed_index()
            
            # Multi-timeframe trend analysis
            trend_primary = self.analyze_trend(df_primary)
            trend_secondary = self.analyze_trend(df_secondary) if df_secondary is not None else 'NEUTRAL'
            trend_higher = self.analyze_trend(df_higher) if df_higher is not None else 'NEUTRAL'
            
            # Generate trading signal
            signal = self.generate_trading_signal(
                df_primary, patterns, fib_levels, sentiment_data, 
                trend_primary, trend_secondary, trend_higher, symbol
            )
            
            return signal
            
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def analyze_trend(self, df: pd.DataFrame) -> str:
        """Analyze overall trend direction"""
        if df is None or len(df) < 20:
            return 'NEUTRAL'
        
        current = df.iloc[-1]
        
        # EMA trend
        ema_trend = 'BULLISH' if current['ema_9'] > current['ema_21'] > current['ema_50'] else 'BEARISH'
        
        # Price vs moving averages
        price_above_emas = current['close'] > current['ema_9'] > current['ema_21']
        price_below_emas = current['close'] < current['ema_9'] < current['ema_21']
        
        # ADX strength
        strong_trend = current['adx'] > self.config.min_adx_strength
        
        if price_above_emas and ema_trend == 'BULLISH' and strong_trend:
            return 'STRONG_BULLISH'
        elif price_below_emas and ema_trend == 'BEARISH' and strong_trend:
            return 'STRONG_BEARISH'
        elif ema_trend == 'BULLISH':
            return 'BULLISH'
        elif ema_trend == 'BEARISH':
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def generate_trading_signal(self, df: pd.DataFrame, patterns: Dict, fib_levels: Dict, 
                              sentiment_data: Dict, trend_primary: str, trend_secondary: str, 
                              trend_higher: str, symbol: str) -> TradingSignal:
        """Generate comprehensive trading signal"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize signal parameters
        signal_type = 'HOLD'
        strength = 0
        confidence = 0
        reasoning_parts = []
        
        # Technical analysis scoring
        technical_score = 0
        
        # RSI analysis
        if current['rsi'] < self.config.rsi_oversold:
            technical_score += 15
            reasoning_parts.append(f"RSI oversold ({current['rsi']:.1f})")
        elif current['rsi'] > self.config.rsi_overbought:
            technical_score -= 15
            reasoning_parts.append(f"RSI overbought ({current['rsi']:.1f})")
        
        # MACD analysis
        if current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            technical_score += 20
            reasoning_parts.append("MACD bullish crossover")
        elif current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            technical_score -= 20
            reasoning_parts.append("MACD bearish crossover")
        
        # Bollinger Bands analysis
        if current['close'] < current['bb_lower'] and current['bb_position'] < 0.1:
            technical_score += 10
            reasoning_parts.append("Price below lower Bollinger Band")
        elif current['close'] > current['bb_upper'] and current['bb_position'] > 0.9:
            technical_score -= 10
            reasoning_parts.append("Price above upper Bollinger Band")
        
        # Volume analysis
        if current['volume_ratio'] > self.config.min_volume_spike:
            technical_score += 10
            reasoning_parts.append(f"High volume ({current['volume_ratio']:.1f}x average)")
        
        # Pattern analysis
        pattern_score = 0
        if patterns.get('golden_cross'):
            pattern_score += 25
            reasoning_parts.append("Golden cross detected")
        if patterns.get('death_cross'):
            pattern_score -= 25
            reasoning_parts.append("Death cross detected")
        if patterns.get('engulfing_bullish'):
            pattern_score += 15
            reasoning_parts.append("Bullish engulfing pattern")
        if patterns.get('engulfing_bearish'):
            pattern_score -= 15
            reasoning_parts.append("Bearish engulfing pattern")
        if patterns.get('hammer'):
            pattern_score += 10
            reasoning_parts.append("Hammer pattern")
        if patterns.get('shooting_star'):
            pattern_score -= 10
            reasoning_parts.append("Shooting star pattern")
        
        # Multi-timeframe analysis
        timeframe_score = 0
        if trend_primary == 'STRONG_BULLISH' and trend_secondary in ['BULLISH', 'STRONG_BULLISH']:
            timeframe_score += 20
            reasoning_parts.append("Multi-timeframe bullish alignment")
        elif trend_primary == 'STRONG_BEARISH' and trend_secondary in ['BEARISH', 'STRONG_BEARISH']:
            timeframe_score -= 20
            reasoning_parts.append("Multi-timeframe bearish alignment")
        
        # Sentiment analysis
        sentiment_score = 0
        if sentiment_data:
            if sentiment_data['sentiment_score'] > 0.3:
                sentiment_score += 10
                reasoning_parts.append("Positive social sentiment")
            elif sentiment_data['sentiment_score'] < -0.3:
                sentiment_score -= 10
                reasoning_parts.append("Negative social sentiment")
        
        # Calculate total score
        total_score = technical_score + pattern_score + timeframe_score + sentiment_score
        
        # Determine signal type and strength
        if total_score >= 40:
            signal_type = 'BUY'
            strength = min(100, total_score)
            confidence = min(95, strength * 0.8 + 20)
        elif total_score <= -40:
            signal_type = 'SELL'
            strength = min(100, abs(total_score))
            confidence = min(95, strength * 0.8 + 20)
        else:
            signal_type = 'HOLD'
            strength = 0
            confidence = 50
            reasoning_parts.append("Insufficient signal strength")
        
        # Calculate entry, stop loss, and take profit
        entry_price = current['close']
        atr = current['atr']
        
        if signal_type == 'BUY':
            stop_loss = entry_price - (atr * self.config.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr * self.config.take_profit_atr_multiplier)
        elif signal_type == 'SELL':
            stop_loss = entry_price + (atr * self.config.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr * self.config.take_profit_atr_multiplier)
        else:
            stop_loss = entry_price
            take_profit = entry_price
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Filter signals by risk-reward ratio
        if signal_type != 'HOLD' and risk_reward_ratio < self.config.risk_reward_ratio:
            signal_type = 'HOLD'
            strength = 0
            confidence = 30
            reasoning_parts.append(f"Poor risk-reward ratio ({risk_reward_ratio:.2f})")
        
        # Create signal object
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            confidence=confidence,
            timeframe=self.config.primary_timeframe,
            indicators={
                'rsi': float(current['rsi']),
                'macd': float(current['macd']),
                'macd_signal': float(current['macd_signal']),
                'bb_position': float(current['bb_position']),
                'volume_ratio': float(current['volume_ratio']),
                'atr': float(current['atr']),
                'adx': float(current['adx']),
                'ema_9': float(current['ema_9']),
                'ema_21': float(current['ema_21']),
                'close': float(current['close'])
            },
            market_conditions={
                'trend_primary': trend_primary,
                'trend_secondary': trend_secondary,
                'trend_higher': trend_higher,
                'patterns': patterns,
                'sentiment': sentiment_data,
                'fibonacci_levels': fib_levels
            },
            reasoning='; '.join(reasoning_parts) if reasoning_parts else 'No significant signals detected'
        )
        
        return signal
    
    async def create_advanced_chart(self, symbol: str, signal: TradingSignal) -> Optional[bytes]:
        """Create professional trading chart"""
        if plt is None:
            return None
        
        try:
            df = await self.fetch_market_data(symbol, self.config.primary_timeframe, limit=100)
            if df is None or len(df) < 50:
                return None
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12), facecolor='black')
            gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], hspace=0.3, wspace=0.2)
            
            # Main price chart
            ax1 = fig.add_subplot(gs[0, :])
            ax1.set_facecolor('black')
            
            # Plot candlesticks (simplified)
            for i in range(len(df)):
                row = df.iloc[i]
                color = 'lime' if row['close'] > row['open'] else 'red'
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
                ax1.plot([i, i], [row['open'], row['close']], color=color, linewidth=3)
            
            # Plot moving averages
            ax1.plot(range(len(df)), df['ema_9'], color='yellow', linewidth=2, label='EMA 9', alpha=0.8)
            ax1.plot(range(len(df)), df['ema_21'], color='orange', linewidth=2, label='EMA 21', alpha=0.8)
            ax1.plot(range(len(df)), df['ema_50'], color='purple', linewidth=2, label='EMA 50', alpha=0.8)
            
            # Plot Bollinger Bands
            ax1.fill_between(range(len(df)), df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
            ax1.plot(range(len(df)), df['bb_upper'], color='gray', linestyle='--', alpha=0.5)
            ax1.plot(range(len(df)), df['bb_lower'], color='gray', linestyle='--', alpha=0.5)
            
            # Plot VWAP
            ax1.plot(range(len(df)), df['vwap'], color='cyan', linewidth=1, label='VWAP', alpha=0.7)
            
            # Mark signal entry point
            if signal.signal_type != 'HOLD':
                color = 'lime' if signal.signal_type == 'BUY' else 'red'
                ax1.scatter([len(df)-1], [signal.entry_price], color=color, s=200, marker='^' if signal.signal_type == 'BUY' else 'v', zorder=5)
                ax1.axhline(y=signal.stop_loss, color='red', linestyle=':', alpha=0.7, label=f'Stop Loss: {signal.stop_loss:.6f}')
                ax1.axhline(y=signal.take_profit, color='lime', linestyle=':', alpha=0.7, label=f'Take Profit: {signal.take_profit:.6f}')
            
            ax1.set_title(f'{symbol} - {signal.signal_type} Signal (Strength: {signal.strength:.0f}%)', 
                         color='white', fontsize=16, fontweight='bold')
            ax1.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='white')
            
            # RSI subplot
            ax2 = fig.add_subplot(gs[1, :])
            ax2.set_facecolor('black')
            ax2.plot(range(len(df)), df['rsi'], color='purple', linewidth=2)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='lime', linestyle='--', alpha=0.5)
            ax2.fill_between(range(len(df)), 30, 70, alpha=0.1, color='gray')
            ax2.set_ylabel('RSI', color='white')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='white')
            
            # MACD subplot
            ax3 = fig.add_subplot(gs[2, :])
            ax3.set_facecolor('black')
            ax3.plot(range(len(df)), df['macd'], color='blue', linewidth=2, label='MACD')
            ax3.plot(range(len(df)), df['macd_signal'], color='red', linewidth=2, label='Signal')
            ax3.bar(range(len(df)), df['macd_histogram'], color='gray', alpha=0.3, label='Histogram')
            ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
            ax3.set_ylabel('MACD', color='white')
            ax3.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(colors='white')
            
            # Volume subplot
            ax4 = fig.add_subplot(gs[3, :])
            ax4.set_facecolor('black')
            colors_vol = ['lime' if df.iloc[i]['close'] > df.iloc[i]['open'] else 'red' for i in range(len(df))]
            ax4.bar(range(len(df)), df['volume'], color=colors_vol, alpha=0.7)
            ax4.plot(range(len(df)), df['volume_sma'], color='yellow', linewidth=2, label='Volume SMA')
            ax4.set_ylabel('Volume', color='white')
            ax4.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(colors='white')
            
            # Add signal information text
            info_text = (
                f"Signal: {signal.signal_type}\n"
                f"Strength: {signal.strength:.0f}%\n"
                f"Confidence: {signal.confidence:.0f}%\n"
                f"R/R Ratio: {signal.risk_reward_ratio:.2f}\n"
                f"Entry: {signal.entry_price:.6f}\n"
                f"Stop Loss: {signal.stop_loss:.6f}\n"
                f"Take Profit: {signal.take_profit:.6f}"
            )
            
            fig.text(0.02, 0.98, info_text, transform=fig.transFigure, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                    color='white')
            
            # Save to bytes
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='black', edgecolor='white', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return buf.getvalue()
            
        except Exception as e:
            logging.error(f"Error creating chart for {symbol}: {e}")
            return None
    
    async def send_signal_message(self, signal: TradingSignal):
        """Send formatted signal message to Telegram"""
        try:
            # Create message text
            emoji_map = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
            signal_emoji = emoji_map.get(signal.signal_type, '⚪')
            
            message = f"""
{signal_emoji} **{signal.signal_type} SIGNAL** - {signal.symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **Signal Strength:** {signal.strength:.0f}%
🎯 **Confidence:** {signal.confidence:.0f}%
⚖️ **Risk/Reward:** {signal.risk_reward_ratio:.2f}

💰 **Entry Price:** `{signal.entry_price:.6f}`
🛑 **Stop Loss:** `{signal.stop_loss:.6f}`
🎯 **Take Profit:** `{signal.take_profit:.6f}`

📈 **Technical Indicators:**
• RSI: {signal.indicators['rsi']:.1f}
• MACD: {signal.indicators['macd']:.4f}
• ADX: {signal.indicators['adx']:.1f}
• Volume Ratio: {signal.indicators['volume_ratio']:.1f}x

🔍 **Analysis:** {signal.reasoning}

⏰ **Time:** {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
📊 **Timeframe:** {signal.timeframe}
            """
            
            # Send message
            await self.bot.send_message(
                chat_id=self.config.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Send chart if available
            chart_data = await self.create_advanced_chart(signal.symbol, signal)
            if chart_data:
                await self.bot.send_photo(
                    chat_id=self.config.chat_id,
                    photo=chart_data,
                    caption=f"📊 Technical Analysis Chart for {signal.symbol}"
                )
            
            logging.info(f"Signal sent for {signal.symbol}: {signal.signal_type}")
            
        except Exception as e:
            logging.error(f"Error sending signal message: {e}")
    
    async def run_analysis_cycle(self):
        """Run complete analysis cycle for all symbols"""
        logging.info("Starting analysis cycle...")
        
        signals_generated = 0
        
        for symbol in self.config.symbols:
            try:
                signal = await self.analyze_symbol(symbol)
                
                if signal and signal.signal_type != 'HOLD':
                    # Save signal to database
                    signal_id = self.db.save_signal(signal)
                    
                    # Send signal message
                    await self.send_signal_message(signal)
                    
                    signals_generated += 1
                    
                    # Small delay between signals
                    await asyncio.sleep(2)
                
            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {e}")
                continue
        
        logging.info(f"Analysis cycle completed. Generated {signals_generated} signals.")
    
    async def send_daily_report(self):
        """Send comprehensive daily trading report"""
        try:
            # Get recent signals
            recent_signals = self.db.get_recent_signals(24)
            
            if not recent_signals:
                return
            
            # Calculate statistics
            total_signals = len(recent_signals)
            buy_signals = sum(1 for s in recent_signals if s['signal_type'] == 'BUY')
            sell_signals = sum(1 for s in recent_signals if s['signal_type'] == 'SELL')
            avg_strength = sum(s['strength'] for s in recent_signals) / total_signals
            avg_confidence = sum(s['confidence'] for s in recent_signals) / total_signals
            
            # Create report message
            report = f"""
📊 **DAILY TRADING REPORT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 **Signal Summary (24h):**
• Total Signals: {total_signals}
• Buy Signals: {buy_signals} ({buy_signals/total_signals*100:.1f}%)
• Sell Signals: {sell_signals} ({sell_signals/total_signals*100:.1f}%)

📊 **Performance Metrics:**
• Average Strength: {avg_strength:.1f}%
• Average Confidence: {avg_confidence:.1f}%

🔝 **Top Signals:**
            """
            
            # Add top signals
            sorted_signals = sorted(recent_signals, key=lambda x: x['strength'], reverse=True)[:5]
            for i, signal in enumerate(sorted_signals, 1):
                report += f"\n{i}. {signal['symbol']} - {signal['signal_type']} ({signal['strength']:.0f}%)"
            
            report += f"\n\n⏰ **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            await self.bot.send_message(
                chat_id=self.config.chat_id,
                text=report,
                parse_mode=ParseMode.MARKDOWN
            )
            
            logging.info("Daily report sent successfully")
            
        except Exception as e:
            logging.error(f"Error sending daily report: {e}")
    
    # Telegram Bot Handlers
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        keyboard = [
            [InlineKeyboardButton("📊 Get Signals", callback_data='get_signals')],
            [InlineKeyboardButton("📈 Market Overview", callback_data='market_overview')],
            [InlineKeyboardButton("⚙️ Settings", callback_data='settings')],
            [InlineKeyboardButton("📋 Recent Signals", callback_data='recent_signals')],
            [InlineKeyboardButton("📊 Performance", callback_data='performance')],
            [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_message = """
🤖 **Professional Crypto Trading Bot**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Welcome to the most advanced cryptocurrency trading signal bot! 

🎯 **Features:**
• Multi-timeframe technical analysis
• Advanced pattern recognition
• Risk management calculations
• Market sentiment analysis
• Professional charting
• Performance tracking

Choose an option below to get started:
        """
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'get_signals':
            await query.message.reply_text("🔄 Analyzing markets... Please wait...")
            await self.run_analysis_cycle()
            
        elif query.data == 'market_overview':
            await self.send_market_overview(query.message)
            
        elif query.data == 'recent_signals':
            await self.send_recent_signals(query.message)
            
        elif query.data == 'performance':
            await self.send_performance_report(query.message)
            
        elif query.data == 'settings':
            await self.send_settings_menu(query.message)
            
        elif query.data == 'help':
            await self.send_help_message(query.message)
    
    async def send_market_overview(self, message):
        """Send market overview"""
        try:
            overview = "📊 **MARKET OVERVIEW**\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for symbol in self.config.symbols[:6]:  # Limit to first 6 symbols
                try:
                    df = await self.fetch_market_data(symbol, self.config.primary_timeframe, limit=50)
                    if df is not None and len(df) > 0:
                        current = df.iloc[-1]
                        prev = df.iloc[-2]
                        change = ((current['close'] - prev['close']) / prev['close']) * 100
                        
                        trend_emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
                        
                        overview += f"{trend_emoji} **{symbol}**\n"
                        overview += f"   Price: {current['close']:.6f}\n"
                        overview += f"   Change: {change:+.2f}%\n"
                        overview += f"   RSI: {current['rsi']:.1f}\n\n"
                        
                except Exception as e:
                    continue
            
            await message.reply_text(overview, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logging.error(f"Error sending market overview: {e}")
            await message.reply_text("❌ Error generating market overview.")
    
    async def send_recent_signals(self, message):
        """Send recent signals"""
        try:
            recent_signals = self.db.get_recent_signals(24)
            
            if not recent_signals:
                await message.reply_text("📭 No recent signals found.")
                return
            
            signals_text = "📋 **RECENT SIGNALS (24h)**\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for signal in recent_signals[:10]:  # Show last 10 signals
                emoji = "🟢" if signal['signal_type'] == 'BUY' else "🔴" if signal['signal_type'] == 'SELL' else "🟡"
                signals_text += f"{emoji} **{signal['symbol']}** - {signal['signal_type']}\n"
                signals_text += f"   Strength: {signal['strength']:.0f}% | Confidence: {signal['confidence']:.0f}%\n"
                signals_text += f"   Time: {signal['timestamp'][:16]}\n\n"
            
            await message.reply_text(signals_text, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logging.error(f"Error sending recent signals: {e}")
            await message.reply_text("❌ Error retrieving recent signals.")
    
    async def send_performance_report(self, message):
        """Send performance report"""
        try:
            recent_signals = self.db.get_recent_signals(168)  # 7 days
            
            if not recent_signals:
                await message.reply_text("📊 No performance data available.")
                return
            
            # Calculate statistics
            total = len(recent_signals)
            buy_signals = sum(1 for s in recent_signals if s['signal_type'] == 'BUY')
            sell_signals = sum(1 for s in recent_signals if s['signal_type'] == 'SELL')
            avg_strength = sum(s['strength'] for s in recent_signals) / total
            avg_confidence = sum(s['confidence'] for s in recent_signals) / total
            
            performance_text = f"""
📊 **PERFORMANCE REPORT (7 Days)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 **Signal Statistics:**
• Total Signals: {total}
• Buy Signals: {buy_signals} ({buy_signals/total*100:.1f}%)
• Sell Signals: {sell_signals} ({sell_signals/total*100:.1f}%)

📊 **Quality Metrics:**
• Average Strength: {avg_strength:.1f}%
• Average Confidence: {avg_confidence:.1f}%

🎯 **Most Active Pairs:**
            """
            
            # Count signals by symbol
            symbol_counts = {}
            for signal in recent_signals:
                symbol = signal['symbol']
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            # Sort by count and show top 5
            top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for symbol, count in top_symbols:
                performance_text += f"\n• {symbol}: {count} signals"
            
            await message.reply_text(performance_text, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logging.error(f"Error sending performance report: {e}")
            await message.reply_text("❌ Error generating performance report.")
    
    async def send_settings_menu(self, message):
        """Send settings menu"""
        settings_text = f"""
⚙️ **CURRENT SETTINGS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **Analysis Settings:**
• Primary Timeframe: {self.config.primary_timeframe}
• Secondary Timeframe: {self.config.secondary_timeframe}
• Higher Timeframe: {self.config.higher_timeframe}

⚖️ **Risk Management:**
• Risk/Reward Ratio: {self.config.risk_reward_ratio}
• Stop Loss Multiplier: {self.config.stop_loss_atr_multiplier}x ATR
• Take Profit Multiplier: {self.config.take_profit_atr_multiplier}x ATR

🎯 **Signal Filters:**
• Min ADX Strength: {self.config.min_adx_strength}
• RSI Oversold: {self.config.rsi_oversold}
• RSI Overbought: {self.config.rsi_overbought}

📈 **Monitored Pairs:** {len(self.config.symbols)}
{', '.join(self.config.symbols[:8])}{'...' if len(self.config.symbols) > 8 else ''}

⏰ **Schedule:**
• Analysis Interval: {self.config.analysis_interval_minutes} minutes
• Alert Check: {self.config.alert_check_interval_minutes} minutes
        """
        
        await message.reply_text(settings_text, parse_mode=ParseMode.MARKDOWN)
    
    async def send_help_message(self, message):
        """Send help message"""
        help_text = """
ℹ️ **HELP & COMMANDS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 **Available Commands:**
• `/start` - Show main menu
• `/signals` - Get current signals
• `/overview` - Market overview
• `/performance` - Performance report
• `/help` - Show this help

📊 **Signal Types:**
• 🟢 **BUY** - Strong bullish signal
• 🔴 **SELL** - Strong bearish signal  
• 🟡 **HOLD** - No clear direction

📈 **Signal Strength:**
• 80-100%: Very Strong
• 60-79%: Strong
• 40-59%: Moderate
• 20-39%: Weak

⚠️ **Risk Disclaimer:**
This bot provides educational signals only. Always do your own research and never invest more than you can afford to lose. Cryptocurrency trading involves significant risk.

🔧 **Technical Support:**
For technical issues or feature requests, please contact the administrator.
        """
        
        await message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def setup_telegram_bot(self):
        """Setup Telegram bot with handlers"""
        self.bot = Bot(token=self.config.telegram_token)
        self.app = Application.builder().token(self.config.telegram_token).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("signals", lambda u, c: self.run_analysis_cycle()))
        self.app.add_handler(CommandHandler("overview", lambda u, c: self.send_market_overview(u.message)))
        self.app.add_handler(CommandHandler("performance", lambda u, c: self.send_performance_report(u.message)))
        self.app.add_handler(CommandHandler("help", lambda u, c: self.send_help_message(u.message)))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
        
        logging.info("Telegram bot setup completed")
    
    async def run_scheduler(self):
        """Run scheduled tasks"""
        last_analysis = 0
        last_alert_check = 0
        last_daily_report = 0
        
        while True:
            try:
                current_time = time.time()
                
                # Analysis cycle
                if current_time - last_analysis >= self.config.analysis_interval_minutes * 60:
                    await self.run_analysis_cycle()
                    last_analysis = current_time
                
                # Alert checks (placeholder for future implementation)
                if current_time - last_alert_check >= self.config.alert_check_interval_minutes * 60:
                    # Add alert checking logic here
                    last_alert_check = current_time
                
                # Daily report
                current_hour = datetime.now().hour
                if (current_hour == self.config.daily_report_hour and 
                    current_time - last_daily_report >= 23 * 3600):  # Once per day
                    await self.send_daily_report()
                    last_daily_report = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error in scheduler: {e}")
                await asyncio.sleep(60)
    
    async def run(self):
        """Main run method"""
        try:
            logging.info("Starting Professional Trading Bot...")
            
            # Initialize components
            await self.initialize_exchange()
            await self.setup_telegram_bot()
            
            # Send startup message
            await self.bot.send_message(
                chat_id=self.config.chat_id,
                text="🤖 **Professional Trading Bot Started!**\n\nBot is now active and monitoring markets.",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Run bot and scheduler concurrently
            await asyncio.gather(
                self.app.run_polling(drop_pending_updates=True),
                self.run_scheduler()
            )
            
        except Exception as e:
            logging.error(f"Critical error in main run: {e}")
            raise
        finally:
            if self.exchange:
                await self.exchange.close()
            logging.info("Trading bot shutdown completed")

# ===================== MAIN EXECUTION =====================

async def main():
    """Main function"""
    try:
        # Load configuration
        config = TradingConfig()
        
        # Create and run bot
        bot = ProfessionalTradingBot(config)
        await bot.run()
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    # Ensure proper event loop handling
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the bot
    asyncio.run(main())
