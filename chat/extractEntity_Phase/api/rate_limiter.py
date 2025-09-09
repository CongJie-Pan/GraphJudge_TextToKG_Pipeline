"""
Rate limiting and token tracking for API calls.

This module provides intelligent rate limiting and token usage tracking
for GPT-5-mini API calls, ensuring compliance with OpenAI's rate limits
while optimizing performance.
"""

import time
import random
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from extractEntity_Phase.infrastructure.logging import get_logger


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    rpm_limit: int = 60  # Requests per minute
    tpm_limit: int = 90000  # Tokens per minute
    tpd_limit: int = 2000000  # Tokens per day
    concurrent_limit: int = 3  # Maximum concurrent requests
    retry_attempts: int = 5  # Maximum retry attempts
    base_delay: float = 5.0  # Base delay between requests (seconds)
    jitter_range: float = 0.2  # Jitter range (±20%)


@dataclass
class TokenUsage:
    """Token usage tracking for a time period."""
    tokens: List[int] = field(default_factory=list)
    last_reset: float = field(default_factory=time.time)
    limit: int = 0
    period_seconds: int = 0
    
    def add_tokens(self, token_count: int) -> bool:
        """Add tokens and check if within limit."""
        current_time = time.time()
        
        # Reset if period has passed
        if current_time - self.last_reset >= self.period_seconds:
            self.tokens.clear()
            self.last_reset = current_time
        
        # Check if adding would exceed limit
        if sum(self.tokens) + token_count > self.limit:
            return False
        
        # Record token usage
        self.tokens.append(token_count)
        return True
    
    def get_usage(self) -> Dict[str, int]:
        """Get current usage statistics."""
        current_usage = sum(self.tokens)
        return {
            'used': current_usage,
            'remaining': self.limit - current_usage,
            'percentage': (current_usage / self.limit) * 100 if self.limit > 0 else 0
        }


class RateLimiter:
    """
    Intelligent rate limiter for API calls with token tracking.
    
    Features:
    - Request per minute (RPM) limiting
    - Token per minute (TPM) tracking
    - Token per day (TPD) tracking
    - Concurrent request limiting
    - Intelligent retry strategies
    - Jitter to avoid synchronized requests
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config or RateLimitConfig()
        self.logger = get_logger()
        
        # Token usage trackers
        self.minute_tracker = TokenUsage(
            limit=self.config.tpm_limit,
            period_seconds=60
        )
        self.day_tracker = TokenUsage(
            limit=self.config.tpd_limit,
            period_seconds=86400  # 24 hours
        )
        
        # Request tracking
        self.request_times: List[float] = []
        self.semaphore = asyncio.Semaphore(self.config.concurrent_limit)
        
        # Statistics
        self.total_requests = 0
        self.rate_limited_requests = 0
        self.token_limited_requests = 0
    
    def calculate_delay(self) -> float:
        """
        Calculate delay between requests based on RPM limit.
        
        Returns:
            Delay in seconds
        """
        # Ensure minimum delay based on RPM limit
        min_delay = max(self.config.base_delay, 60.0 / self.config.rpm_limit)
        
        # Add random jitter to avoid synchronized requests
        jitter = random.uniform(1 - self.config.jitter_range, 1 + self.config.jitter_range)
        
        return min_delay * jitter
    
    def can_make_request(self, estimated_tokens: int = 0) -> Tuple[bool, str]:
        """
        Check if a request can be made without exceeding limits.
        
        Args:
            estimated_tokens: Estimated token usage for the request
            
        Returns:
            Tuple of (can_proceed, reason)
        """
        current_time = time.time()
        
        # Check RPM limit
        if len(self.request_times) >= self.config.rpm_limit:
            # Remove old requests outside the 1-minute window
            self.request_times = [t for t in self.request_times 
                                if current_time - t < 60]
            
            if len(self.request_times) >= self.config.rpm_limit:
                return False, "RPM limit exceeded"
        
        # Check TPM limit
        if not self.minute_tracker.add_tokens(estimated_tokens):
            return False, "TPM limit exceeded"
        
        # Check TPD limit
        if not self.day_tracker.add_tokens(estimated_tokens):
            return False, "TPD limit exceeded"
        
        return True, "OK"
    
    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """
        Wait if necessary to comply with rate limits.
        
        Args:
            estimated_tokens: Estimated token usage for the request
        """
        can_proceed, reason = self.can_make_request(estimated_tokens)
        
        if not can_proceed:
            if "RPM" in reason:
                # Wait for RPM reset
                wait_time = 60 - (time.time() % 60) + 1
                self.logger.warning(f"RPM limit exceeded. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            elif "TPM" in reason:
                # Wait for TPM reset
                wait_time = 60 - (time.time() % 60) + 1
                self.logger.warning(f"TPM limit exceeded. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            elif "TPD" in reason:
                # Wait for TPD reset (next day)
                wait_time = 86400 - (time.time() % 86400) + 1
                self.logger.warning(f"TPD limit exceeded. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
    
    async def acquire_slot(self) -> None:
        """Acquire a concurrent request slot."""
        await self.semaphore.acquire()
    
    def release_slot(self) -> None:
        """Release a concurrent request slot."""
        self.semaphore.release()
    
    def record_request(self, token_count: int = 0) -> None:
        """
        Record a completed request.
        
        Args:
            token_count: Actual tokens used in the request
        """
        current_time = time.time()
        self.request_times.append(current_time)
        self.total_requests += 1
        
        # Update token usage with actual count
        if token_count > 0:
            # Remove the estimated tokens and add actual
            self.minute_tracker.tokens.pop() if self.minute_tracker.tokens else None
            self.day_tracker.tokens.pop() if self.day_tracker.tokens else None
            
            self.minute_tracker.add_tokens(token_count)
            self.day_tracker.add_tokens(token_count)
    
    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary with minute and day usage statistics
        """
        return {
            'minute': self.minute_tracker.get_usage(),
            'day': self.day_tracker.get_usage(),
            'requests': {
                'total': self.total_requests,
                'rate_limited': self.rate_limited_requests,
                'token_limited': self.token_limited_requests
            }
        }
    
    def get_rate_limit_info(self) -> Dict[str, any]:
        """
        Get comprehensive rate limit information.
        
        Returns:
            Dictionary with rate limit configuration and current status
        """
        usage_stats = self.get_usage_stats()
        
        return {
            'config': {
                'rpm_limit': self.config.rpm_limit,
                'tpm_limit': self.config.tpm_limit,
                'tpd_limit': self.config.tpd_limit,
                'concurrent_limit': self.config.concurrent_limit,
                'base_delay': self.config.base_delay
            },
            'current_usage': usage_stats,
            'can_make_request': self.can_make_request()[0],
            'available_slots': self.semaphore._value,
            'estimated_delay': self.calculate_delay()
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire_slot()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release_slot()


class RetryStrategy:
    """
    Intelligent retry strategy for different types of API errors.
    """
    
    def __init__(self, max_attempts: int = 5):
        """
        Initialize retry strategy.
        
        Args:
            max_attempts: Maximum number of retry attempts
        """
        self.max_attempts = max_attempts
        self.logger = get_logger()
    
    async def execute_with_retry(self, operation, *args, **kwargs):
        """
        Execute operation with intelligent retry logic.
        
        Args:
            operation: Async function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                if attempt >= self.max_attempts:
                    self.logger.error(f"Operation failed after {self.max_attempts} attempts: {e}")
                    raise last_exception
                
                # Determine wait time based on error type
                wait_time = self._calculate_wait_time(error_msg, attempt)
                
                self.logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed: {e}. "
                    f"Retrying in {wait_time:.1f} seconds..."
                )
                
                await asyncio.sleep(wait_time)
        
        # This should never be reached, but just in case
        raise last_exception
    
    def _calculate_wait_time(self, error_msg: str, attempt: int) -> float:
        """
        Calculate wait time based on error type and attempt number.
        
        Args:
            error_msg: Error message (lowercase)
            attempt: Current attempt number
            
        Returns:
            Wait time in seconds
        """
        if any(phrase in error_msg for phrase in ["rate_limit", "rate limit", "rate_limit_exceeded"]):
            # Rate limit error: Progressive delay with jitter
            base_wait = 5 * (1.5 ** attempt)
            jitter = random.uniform(0.8, 1.2)
            return min(base_wait * jitter, 300)  # Cap at 5 minutes
            
        elif any(phrase in error_msg for phrase in ["overloaded", "busy", "capacity", "server_error"]):
            # Server overload: Longer exponential backoff
            base_wait = 10 * (2 ** attempt)
            return min(base_wait, 600)  # Cap at 10 minutes
            
        elif "timeout" in error_msg:
            # Timeout error: Moderate delay
            base_wait = 5 * (1.5 ** attempt)
            return min(base_wait, 60)  # Cap at 1 minute
            
        else:
            # Other errors: Standard exponential backoff
            base_wait = 3 * (2 ** attempt)
            return min(base_wait, 30)  # Cap at 30 seconds


# Convenience functions for backward compatibility
def create_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Create a rate limiter instance."""
    return RateLimiter(config)


def get_default_rate_limiter() -> RateLimiter:
    """Get a rate limiter with default configuration."""
    return RateLimiter()


def create_retry_strategy(max_attempts: int = 5) -> RetryStrategy:
    """Create a retry strategy instance."""
    return RetryStrategy(max_attempts)
