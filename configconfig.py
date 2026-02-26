"""
Centralized configuration management for AATI system.
Uses Pydantic for type-safe environment validation with explicit defaults.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

from pydantic import BaseSettings, Field, validator
from pydantic.networks import AnyHttpUrl
import structlog

logger = structlog.get_logger(__name__)


class ExchangeType(str, Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class AATIConfig(BaseSettings):
    """Main configuration with environment variable fallbacks."""
    
    # System
    TRADING_MODE: TradingMode = Field(default=TradingMode.PAPER, env="TRADING_MODE")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    HEARTBEAT_INTERVAL: int = Field(default=60, env="HEARTBEAT_INTERVAL")
    
    # Exchange
    PRIMARY_EXCHANGE: ExchangeType = Field(default=ExchangeType.BINANCE, env="PRIMARY_EXCHANGE")
    EXCHANGE_API_KEY: Optional[str] = Field(default=None, env="EXCHANGE_API_KEY")
    EXCHANGE_SECRET: Optional[str] = Field(default=None, env="EXCHANGE_SECRET")
    
    # Firebase
    FIREBASE_PROJECT_ID: str = Field(..., env="FIREBASE_PROJECT_ID")
    FIREBASE_CREDENTIALS_PATH: Path = Field(default=Path("./config/firebase-key.json"), env="FIREBASE_CREDENTIALS")
    
    # Trading Parameters
    ASSET_PAIRS: List[str] = Field(default=["BTC/USDT", "ETH/USDT"], env="ASSET_PAIRS")
    MAX_POSITION_SIZE: float = Field(default=0.1, ge=0.01, le=1.0, env="MAX_POSITION_SIZE")
    STOP_LOSS_PERCENT: float = Field(default=0.02, ge=0.001, le=0.1, env="STOP_LOSS_PERCENT")
    
    # Risk Limits
    DAILY_LOSS_LIMIT: float = Field(default=0.05, env="DAILY_LOSS_LIMIT")
    MAX_DRAWDOWN: float = Field(default=0.15, env="MAX_DRAWDOWN")
    
    # Pattern Recognition
    ML_MODEL_PATH: Path = Field(default=Path("./models/pattern_model.joblib"), env="ML_MODEL_PATH")
    TRAINING_WINDOW_DAYS: int = Field(default=30, env="TRAINING_WINDOW_DAYS")
    
    # External Services
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("FIREBASE_CREDENTIALS_PATH")
    def validate_firebase_credentials(cls, v: Path) -> Path:
        if not v.exists():
            logger.error("Firebase credentials file not found", path=str(v))
            raise FileNotFoundError(f"Firebase credentials not found at {v}")
        return v
    
    @validator("EXCHANGE_API_KEY", "EXCHANGE_SECRET")
    def validate_live_credentials(cls, v, values):
        if values.get("TRADING_MODE") == TradingMode.LIVE and not v:
            raise ValueError("API credentials required for LIVE trading mode")
        return v


# Global configuration instance
_config: Optional[AATIConfig] = None


def get_config() -> AATIConfig:
    """Singleton configuration access with lazy initialization."""
    global _config
    if _config is None:
        try:
            _config = AATIConfig()
            logger.info("Configuration loaded successfully", mode=_config.TRADING_MODE)
        except Exception as e:
            logger.critical("Failed to load configuration", error=str(e))
            raise
    return _config