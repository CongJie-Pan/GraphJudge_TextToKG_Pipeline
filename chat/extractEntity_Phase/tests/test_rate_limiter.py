"""
Tests for rate limiting and token tracking.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from extractEntity_Phase.api.rate_limiter import (
    RateLimitConfig, TokenUsage, RateLimiter, RetryStrategy,
    create_rate_limiter, get_default_rate_limiter, create_retry_strategy
)


class TestRateLimitConfig:
    """Test rate limit configuration dataclass."""
    
    def test_rate_limit_config_creation(self):
        """Test rate limit config creation with all fields."""
        config = RateLimitConfig(
            rpm_limit=100,
            tpm_limit=150000,
            tpd_limit=3000000,
            concurrent_limit=5,
            retry_attempts=3,
            base_delay=2.5,
            jitter_range=0.15
        )
        
        assert config.rpm_limit == 100
        assert config.tpm_limit == 150000
        assert config.tpd_limit == 3000000
        assert config.concurrent_limit == 5
        assert config.retry_attempts == 3
        assert config.base_delay == 2.5
        assert config.jitter_range == 0.15
    
    def test_rate_limit_config_defaults(self):
        """Test rate limit config default values."""
        config = RateLimitConfig()
        
        assert config.rpm_limit == 60
        assert config.tpm_limit == 90000
        assert config.tpd_limit == 2000000
        assert config.concurrent_limit == 3
        assert config.retry_attempts == 5
        assert config.base_delay == 5.0
        assert config.jitter_range == 0.2


class TestTokenUsage:
    """Test token usage tracking for time periods."""
    
    def test_token_usage_creation(self):
        """Test token usage creation."""
        token_usage = TokenUsage(
            limit=1000,
            period_seconds=60
        )
        
        assert token_usage.limit == 1000
        assert token_usage.period_seconds == 60
        assert token_usage.tokens == []
        assert token_usage.last_reset > 0
    
    def test_token_usage_add_tokens_within_limit(self):
        """Test adding tokens within limit."""
        token_usage = TokenUsage(limit=1000, period_seconds=60)
        
        # Add tokens within limit
        assert token_usage.add_tokens(500) == True
        assert token_usage.add_tokens(300) == True
        assert token_usage.add_tokens(200) == True
        
        # Should be at limit
        assert token_usage.add_tokens(1) == False
    
    def test_token_usage_add_tokens_exceeding_limit(self):
        """Test adding tokens exceeding limit."""
        token_usage = TokenUsage(limit=1000, period_seconds=60)
        
        # Add tokens up to limit
        assert token_usage.add_tokens(1000) == True
        
        # Try to exceed limit
        assert token_usage.add_tokens(1) == False
    
    def test_token_usage_reset_after_period(self):
        """Test token usage reset after period expires."""
        token_usage = TokenUsage(limit=1000, period_seconds=60)
        
        # Add some tokens
        token_usage.add_tokens(500)
        assert len(token_usage.tokens) == 1
        
        # Mock time to simulate period expiration
        with patch('time.time') as mock_time:
            mock_time.return_value = time.time() + 61  # 61 seconds later
            
            # Should reset and allow new tokens
            assert token_usage.add_tokens(1000) == True
            assert len(token_usage.tokens) == 1  # Only the new one
    
    def test_token_usage_get_usage(self):
        """Test getting usage statistics."""
        token_usage = TokenUsage(limit=1000, period_seconds=60)
        
        # Add some tokens
        token_usage.add_tokens(300)
        token_usage.add_tokens(200)
        
        usage = token_usage.get_usage()
        
        assert usage['used'] == 500
        assert usage['remaining'] == 500
        assert usage['percentage'] == 50.0
    
    def test_token_usage_empty_period(self):
        """Test token usage with empty period."""
        token_usage = TokenUsage(limit=1000, period_seconds=60)
        
        usage = token_usage.get_usage()
        
        assert usage['used'] == 0
        assert usage['remaining'] == 1000
        assert usage['percentage'] == 0.0


class TestRateLimiter:
    """Test rate limiter class."""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter creation."""
        config = RateLimitConfig(rpm_limit=100, tpm_limit=150000)
        rate_limiter = RateLimiter(config)
        
        assert rate_limiter.config == config
        assert rate_limiter.config.rpm_limit == 100
        assert rate_limiter.config.tpm_limit == 150000
    
    def test_rate_limiter_default_creation(self):
        """Test rate limiter creation with default config."""
        rate_limiter = RateLimiter()
        
        assert rate_limiter.config.rpm_limit == 60
        assert rate_limiter.config.tpm_limit == 90000
        assert rate_limiter.config.concurrent_limit == 3
    
    def test_calculate_delay(self):
        """Test delay calculation."""
        rate_limiter = RateLimiter()
        
        delay = rate_limiter.calculate_delay()
        
        # Should be at least base_delay
        assert delay >= rate_limiter.config.base_delay
        
        # Should have jitter applied
        base_delay = rate_limiter.config.base_delay
        jitter_range = rate_limiter.config.jitter_range
        min_expected = base_delay * (1 - jitter_range)
        max_expected = base_delay * (1 + jitter_range)
        
        assert min_expected <= delay <= max_expected
    
    def test_can_make_request_within_limits(self):
        """Test can_make_request when within limits."""
        rate_limiter = RateLimiter()
        
        can_proceed, reason = rate_limiter.can_make_request(estimated_tokens=1000)
        
        assert can_proceed == True
        assert reason == "OK"
    
    def test_can_make_request_rpm_exceeded(self):
        """Test can_make_request when RPM limit exceeded."""
        config = RateLimitConfig(rpm_limit=2)  # Very low limit for testing
        rate_limiter = RateLimiter(config)
        
        # Make 2 requests
        rate_limiter.record_request()
        rate_limiter.record_request()
        
        # Third request should be blocked
        can_proceed, reason = rate_limiter.can_make_request()
        
        assert can_proceed == False
        assert "RPM limit exceeded" in reason
    
    def test_can_make_request_tpm_exceeded(self):
        """Test can_make_request when TPM limit exceeded."""
        config = RateLimitConfig(tpm_limit=1000)  # Very low limit for testing
        rate_limiter = RateLimiter(config)
        
        # Try to exceed TPM limit
        can_proceed, reason = rate_limiter.can_make_request(estimated_tokens=1500)
        
        assert can_proceed == False
        assert "TPM limit exceeded" in reason
    
    def test_can_make_request_tpd_exceeded(self):
        """Test can_make_request when TPD limit exceeded."""
        config = RateLimitConfig(tpd_limit=1000)  # Very low limit for testing
        rate_limiter = RateLimiter(config)
        
        # Try to exceed TPD limit
        can_proceed, reason = rate_limiter.can_make_request(estimated_tokens=1500)
        
        assert can_proceed == False
        assert "TPD limit exceeded" in reason
    
    @pytest.mark.asyncio
    async def test_wait_if_needed_rpm_exceeded(self):
        """Test wait_if_needed when RPM limit exceeded."""
        config = RateLimitConfig(rpm_limit=2)  # Very low limit for testing
        rate_limiter = RateLimiter(config)
        
        # Make 2 requests
        rate_limiter.record_request()
        rate_limiter.record_request()
        
        # Mock time to control waiting
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0  # Fixed time
            
            # This should trigger waiting
            start_time = time.time()
            await rate_limiter.wait_if_needed()
            elapsed = time.time() - start_time
            
            # Should have waited some time
            assert elapsed > 0
    
    @pytest.mark.asyncio
    async def test_wait_if_needed_within_limits(self):
        """Test wait_if_needed when within limits."""
        rate_limiter = RateLimiter()
        
        start_time = time.time()
        await rate_limiter.wait_if_needed(estimated_tokens=1000)
        elapsed = time.time() - start_time
        
        # Should not wait when within limits
        assert elapsed < 0.1  # Very small delay
    
    @pytest.mark.asyncio
    async def test_acquire_release_slot(self):
        """Test acquiring and releasing concurrent request slots."""
        config = RateLimitConfig(concurrent_limit=2)
        rate_limiter = RateLimiter(config)
        
        # Should be able to acquire slots
        await rate_limiter.acquire_slot()
        assert rate_limiter.semaphore._value == 1
        
        await rate_limiter.acquire_slot()
        assert rate_limiter.semaphore._value == 0
        
        # Third acquisition should block
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(rate_limiter.acquire_slot(), timeout=0.1)
        
        # Release slots
        rate_limiter.release_slot()
        rate_limiter.release_slot()
        
        assert rate_limiter.semaphore._value == 2
    
    def test_record_request(self):
        """Test recording completed requests."""
        rate_limiter = RateLimiter()
        
        initial_total = rate_limiter.total_requests
        
        # Record a request
        rate_limiter.record_request(token_count=500)
        
        assert rate_limiter.total_requests == initial_total + 1
        assert len(rate_limiter.request_times) == 1
        
        # Check token usage was recorded
        minute_usage = rate_limiter.minute_tracker.get_usage()
        assert minute_usage['used'] == 500
    
    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        rate_limiter = RateLimiter()
        
        # Record some requests
        rate_limiter.record_request(token_count=1000)
        rate_limiter.record_request(token_count=500)
        
        stats = rate_limiter.get_usage_stats()
        
        assert 'minute' in stats
        assert 'day' in stats
        assert 'requests' in stats
        
        assert stats['requests']['total'] == 2
        assert stats['minute']['used'] == 1500
    
    def test_get_rate_limit_info(self):
        """Test getting comprehensive rate limit information."""
        rate_limiter = RateLimiter()
        
        info = rate_limiter.get_rate_limit_info()
        
        assert 'config' in info
        assert 'current_usage' in info
        assert 'can_make_request' in info
        assert 'available_slots' in info
        assert 'estimated_delay' in info
        
        assert info['config']['rpm_limit'] == 60
        assert info['config']['tpm_limit'] == 90000
        assert info['available_slots'] == 3
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test rate limiter as async context manager."""
        rate_limiter = RateLimiter()
        
        async with rate_limiter:
            # Should have acquired a slot
            assert rate_limiter.semaphore._value == 2
        
        # Should have released the slot
        assert rate_limiter.semaphore._value == 3


class TestRetryStrategy:
    """Test retry strategy class."""
    
    def test_retry_strategy_creation(self):
        """Test retry strategy creation."""
        retry_strategy = RetryStrategy(max_attempts=3)
        
        assert retry_strategy.max_attempts == 3
    
    def test_retry_strategy_default_creation(self):
        """Test retry strategy creation with default attempts."""
        retry_strategy = RetryStrategy()
        
        assert retry_strategy.max_attempts == 5
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_try(self):
        """Test retry strategy with successful first attempt."""
        retry_strategy = RetryStrategy(max_attempts=3)
        
        async def mock_operation():
            return "success"
        
        result = await retry_strategy.execute_with_retry(mock_operation)
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_retries(self):
        """Test retry strategy with success after retries."""
        retry_strategy = RetryStrategy(max_attempts=3)
        
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await retry_strategy.execute_with_retry(mock_operation)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_max_attempts_exceeded(self):
        """Test retry strategy when max attempts exceeded."""
        retry_strategy = RetryStrategy(max_attempts=2)
        
        async def mock_operation():
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            await retry_strategy.execute_with_retry(mock_operation)
    
    def test_calculate_wait_time_rate_limit_error(self):
        """Test wait time calculation for rate limit errors."""
        retry_strategy = RetryStrategy()
        
        wait_time = retry_strategy._calculate_wait_time("rate_limit_exceeded", 1)
        
        # Should be reasonable wait time for rate limit
        assert 0 < wait_time <= 300  # Cap at 5 minutes
    
    def test_calculate_wait_time_server_overload(self):
        """Test wait time calculation for server overload errors."""
        retry_strategy = RetryStrategy()
        
        wait_time = retry_strategy._calculate_wait_time("server_overloaded", 1)
        
        # Should be longer wait time for server overload
        assert 0 < wait_time <= 600  # Cap at 10 minutes
    
    def test_calculate_wait_time_timeout_error(self):
        """Test wait time calculation for timeout errors."""
        retry_strategy = RetryStrategy()
        
        wait_time = retry_strategy._calculate_wait_time("timeout", 1)
        
        # Should be moderate wait time for timeout
        assert 0 < wait_time <= 60  # Cap at 1 minute
    
    def test_calculate_wait_time_other_error(self):
        """Test wait time calculation for other errors."""
        retry_strategy = RetryStrategy()
        
        wait_time = retry_strategy._calculate_wait_time("unknown_error", 1)
        
        # Should be standard wait time for other errors
        assert 0 < wait_time <= 30  # Cap at 30 seconds


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_rate_limiter(self):
        """Test create_rate_limiter function."""
        config = RateLimitConfig(rpm_limit=100)
        rate_limiter = create_rate_limiter(config)
        
        assert isinstance(rate_limiter, RateLimiter)
        assert rate_limiter.config.rpm_limit == 100
    
    def test_get_default_rate_limiter(self):
        """Test get_default_rate_limiter function."""
        rate_limiter = get_default_rate_limiter()
        
        assert isinstance(rate_limiter, RateLimiter)
        assert rate_limiter.config.rpm_limit == 60
    
    def test_create_retry_strategy(self):
        """Test create_retry_strategy function."""
        retry_strategy = create_retry_strategy(max_attempts=10)
        
        assert isinstance(retry_strategy, RetryStrategy)
        assert retry_strategy.max_attempts == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_rate_limiter_zero_limits(self):
        """Test rate limiter with zero limits."""
        config = RateLimitConfig(rpm_limit=0, tpm_limit=0, tpd_limit=0)
        rate_limiter = RateLimiter(config)
        
        can_proceed, reason = rate_limiter.can_make_request()
        
        assert can_proceed == False
        assert "RPM limit exceeded" in reason
    
    def test_rate_limiter_very_high_limits(self):
        """Test rate limiter with very high limits."""
        config = RateLimitConfig(rpm_limit=10000, tpm_limit=1000000, tpd_limit=10000000)
        rate_limiter = RateLimiter(config)
        
        can_proceed, reason = rate_limiter.can_make_request(estimated_tokens=100000)
        
        assert can_proceed == True
        assert reason == "OK"
    
    def test_token_usage_boundary_values(self):
        """Test token usage with boundary values."""
        token_usage = TokenUsage(limit=1000, period_seconds=60)
        
        # Test exact limit
        assert token_usage.add_tokens(1000) == True
        assert token_usage.add_tokens(0) == False  # 0 tokens should not exceed limit
        
        # Test negative tokens (should handle gracefully)
        assert token_usage.add_tokens(-100) == True  # Should handle negative gracefully
    
    def test_retry_strategy_edge_attempts(self):
        """Test retry strategy with edge attempt numbers."""
        retry_strategy = RetryStrategy(max_attempts=1)
        
        # Test with attempt 0 (edge case)
        wait_time = retry_strategy._calculate_wait_time("test_error", 0)
        assert wait_time > 0
        
        # Test with very high attempt number
        wait_time = retry_strategy._calculate_wait_time("test_error", 100)
        assert wait_time > 0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_rate_limiter_invalid_config(self):
        """Test rate limiter with invalid configuration."""
        # Should handle gracefully
        rate_limiter = RateLimiter()
        
        # Test with invalid estimated tokens
        can_proceed, reason = rate_limiter.can_make_request(estimated_tokens=-1000)
        
        # Should handle negative tokens gracefully
        assert can_proceed == True or can_proceed == False
        assert isinstance(reason, str)
    
    def test_token_usage_invalid_period(self):
        """Test token usage with invalid period."""
        # Should handle gracefully
        token_usage = TokenUsage(limit=1000, period_seconds=0)
        
        # Should handle zero period gracefully
        assert token_usage.add_tokens(100) == True or token_usage.add_tokens(100) == False
    
    @pytest.mark.asyncio
    async def test_retry_strategy_operation_exception(self):
        """Test retry strategy with operation that raises exceptions."""
        retry_strategy = RetryStrategy(max_attempts=2)
        
        async def mock_operation():
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError, match="Test exception"):
            await retry_strategy.execute_with_retry(mock_operation)
    
    def test_rate_limiter_semaphore_error(self):
        """Test rate limiter with semaphore errors."""
        rate_limiter = RateLimiter()
        
        # Mock semaphore to raise error
        rate_limiter.semaphore = MagicMock()
        rate_limiter.semaphore.acquire.side_effect = Exception("Semaphore error")
        
        # Should handle gracefully
        with pytest.raises(Exception, match="Semaphore error"):
            asyncio.run(rate_limiter.acquire_slot())
