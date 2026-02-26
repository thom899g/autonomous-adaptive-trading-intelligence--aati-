"""
Robust market data collection with multi-exchange support, error handling, and Firebase persistence.
Implements retry logic, rate limiting, and data validation.
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import ccxt
import pandas as pd
import numpy as np
from ccxt.base.errors import NetworkError, ExchangeError, RequestTimeout
import structlog

from config.config import get_config, ExchangeType
from core.firebase_client import FirebaseClient

logger = structlog.get_logger(__name__)


@dataclass
class MarketData:
    """Type-safe market data container with validation."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "exchange": self.exchange
        }
    
    @classmethod
    def from_ohlcv(cls, symbol: