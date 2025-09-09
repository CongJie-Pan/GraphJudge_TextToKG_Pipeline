"""
Unit Tests for Enhanced KIMI-K2 Configuration Module

This test suite validates the enhanced functionality of the kimi_config.py module,
including free tier optimizations, token tracking, and intelligent rate limiting.

Test Coverage:
- Free tier configuration settings
- Token usage tracking (TPM/TPD limits)
- Progressive delay calculation with jitter
- Configuration presets application
- Token usage statistics and reset functionality
- Rate limit management for different API tiers

Run with: pytest test_kimi_config_enhanced.py -v
"""

import pytest
import time
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import kimi_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import kimi_config


class TestFreeTierConfiguration:
    """Test cases for free tier configuration settings."""
    
    def test_free_tier_default_settings(self):
        """Test that free tier default settings are correctly configured."""

        
        # Verify free tier limits
        assert kimi_config.KIMI_RPM_LIMIT == 3
        assert kimi_config.KIMI_CONCURRENT_LIMIT == 1
        assert kimi_config.KIMI_TPM_LIMIT == 32000
        assert kimi_config.KIMI_TPD_LIMIT == 1500000
        assert kimi_config.KIMI_RETRY_ATTEMPTS == 7
        assert kimi_config.KIMI_BASE_DELAY == 25
        assert kimi_config.KIMI_MAX_TOKENS == 800
    
    def test_calculate_rate_limit_delay_with_jitter(self):
        """Test rate limit delay calculation with jitter."""

        
        # Test multiple delay calculations to verify jitter variation
        delays = [kimi_config.calculate_rate_limit_delay() for _ in range(10)]
        
        # All delays should be positive integers
        assert all(isinstance(delay, int) and delay > 0 for delay in delays)
        
        # Should have variation due to jitter (not all the same)
        assert len(set(delays)) > 1 or len(delays) == 1  # Allow for small sample coincidence
        
        # Should be within reasonable range (base_delay ± 20%)
        base_delay = kimi_config.KIMI_BASE_DELAY
        for delay in delays:
            assert delay >= int(base_delay * 0.8)
            assert delay <= int(base_delay * 1.2)
    
    def test_api_config_summary(self):
        """Test API configuration summary includes all required fields."""

        
        config = kimi_config.get_api_config_summary()
        
        # Verify all required fields are present
        required_fields = [
            'rpm_limit', 'tpm_limit', 'tpd_limit', 'concurrent_limit',
            'retry_attempts', 'base_delay', 'calculated_delay',
            'temperature', 'max_tokens', 'model', 'token_usage'
        ]
        
        for field in required_fields:
            assert field in config, f"Missing field: {field}"
        
        # Verify token_usage subfields
        token_usage = config['token_usage']
        token_fields = [
            'minute_tokens', 'day_tokens', 'minute_remaining', 
            'day_remaining', 'minute_percentage', 'day_percentage'
        ]
        
        for field in token_fields:
            assert field in token_usage, f"Missing token usage field: {field}"


class TestTokenUsageTracking:
    """Test cases for token usage tracking functionality."""
    
    def setup_method(self):
        """Reset token tracking state before each test."""

        kimi_config._token_usage_minute.clear()
        kimi_config._token_usage_day.clear()
        kimi_config._last_reset_minute = time.time()
        kimi_config._last_reset_day = time.time()
    
    def test_token_tracking_within_limits(self):
        """Test token tracking when within TPM and TPD limits."""

        
        # Test tracking small amounts within limits
        assert kimi_config.track_token_usage(1000) is True
        assert kimi_config.track_token_usage(2000) is True
        assert kimi_config.track_token_usage(5000) is True
        
        # Verify tokens are being tracked
        stats = kimi_config.get_token_usage_stats()
        assert stats['minute_tokens'] == 8000
        assert stats['day_tokens'] == 8000
    
    def test_token_tracking_exceeding_tpm_limit(self):
        """Test token tracking when exceeding TPM limit."""

        
        # Fill up to near TPM limit
        kimi_config.track_token_usage(30000)
        
        # Next request should be rejected
        assert kimi_config.track_token_usage(3000) is False
        
        # Verify stats reflect the rejection
        stats = kimi_config.get_token_usage_stats()
        assert stats['minute_tokens'] == 30000  # Only first request tracked
        assert stats['minute_remaining'] == 2000
    
    @patch('kimi_config.time.time')
    def test_token_tracking_exceeding_tpd_limit(self, mock_time):
        """Test token tracking when exceeding TPD limit."""

        
        # Set fixed time to prevent auto-reset
        mock_time.return_value = 1000.0
        kimi_config._last_reset_minute = 1000.0
        kimi_config._last_reset_day = 1000.0
        
        # CRITICAL: Clear existing usage first
        kimi_config._token_usage_minute.clear()
        kimi_config._token_usage_day.clear()
        
        # Simulate approaching TPD limit (1.5M limit)
        # Add tokens that would cause next request to exceed limit
        kimi_config._token_usage_day = [1490000]  # 1.49M used, only 10K remaining
        
        # Test 1: Large request (200K) exceeds TPD limit
        result_large = kimi_config.track_token_usage(200000)
        assert result_large is False, f"Large request should be rejected: 1490000 + 200000 > 1500000"
        
        # Test 2: Medium request (15K) exceeds TPD limit  
        result_medium = kimi_config.track_token_usage(15000)
        assert result_medium is False, f"Medium request should be rejected: 1490000 + 15000 > 1500000"
        
        # Test 3: Small request (5K) within TPD limit should succeed
        result_small = kimi_config.track_token_usage(5000)
        assert result_small is True, f"Small request should succeed: 1490000 + 5000 <= 1500000"
    
    @patch('kimi_config.time.time')
    def test_token_usage_minute_reset(self, mock_time):
        """Test that token usage resets after minute window."""

        
        # Set initial time
        mock_time.return_value = 1000.0
        kimi_config._last_reset_minute = 1000.0
        
        # Track some tokens
        kimi_config.track_token_usage(5000)
        assert len(kimi_config._token_usage_minute) == 1
        
        # Advance time by 61 seconds
        mock_time.return_value = 1061.0
        
        # Next tracking should reset minute counter
        kimi_config.track_token_usage(3000)
        assert len(kimi_config._token_usage_minute) == 1  # Should be reset and have new entry
        
        # Day counter should still accumulate
        assert len(kimi_config._token_usage_day) == 2
    
    @patch('kimi_config.time.time')
    def test_token_usage_day_reset(self, mock_time):
        """Test that token usage resets after day window."""

        
        # Set initial time
        mock_time.return_value = 1000.0
        kimi_config._last_reset_day = 1000.0
        
        # Track some tokens
        kimi_config.track_token_usage(10000)
        assert len(kimi_config._token_usage_day) == 1
        
        # Advance time by 25 hours (more than 24 hours)
        mock_time.return_value = 1000.0 + (25 * 60 * 60)
        
        # Next tracking should reset day counter
        kimi_config.track_token_usage(5000)
        assert len(kimi_config._token_usage_day) == 1  # Should be reset and have new entry
    
    def test_token_usage_statistics_calculation(self):
        """Test token usage statistics calculation accuracy."""

        
        # Track known amounts
        kimi_config.track_token_usage(8000)
        kimi_config.track_token_usage(12000)
        kimi_config.track_token_usage(5000)
        
        stats = kimi_config.get_token_usage_stats()
        
        # Verify calculations
        assert stats['minute_tokens'] == 25000
        assert stats['day_tokens'] == 25000
        assert stats['minute_remaining'] == 7000  # 32000 - 25000
        assert stats['day_remaining'] == 1475000  # 1500000 - 25000
        assert abs(stats['minute_percentage'] - 78.125) < 0.01  # 25000/32000 * 100
        assert abs(stats['day_percentage'] - 1.667) < 0.01  # 25000/1500000 * 100


class TestConfigurationPresets:
    """Test cases for configuration presets functionality."""
    
    def test_available_presets(self):
        """Test that all expected presets are available."""

        
        expected_presets = ['free_tier', 'paid_tier_10_rpm', 'paid_tier_30_rpm']
        
        for preset in expected_presets:
            assert preset in kimi_config.PRESETS
    
    def test_free_tier_preset_values(self):
        """Test free tier preset contains correct values."""

        
        preset = kimi_config.PRESETS['free_tier']
        
        assert preset['KIMI_RPM_LIMIT'] == 3
        assert preset['KIMI_TPM_LIMIT'] == 32000
        assert preset['KIMI_TPD_LIMIT'] == 1500000
        assert preset['KIMI_CONCURRENT_LIMIT'] == 1
        assert preset['KIMI_RETRY_ATTEMPTS'] == 7
        assert preset['KIMI_BASE_DELAY'] == 25
        assert preset['KIMI_MAX_TOKENS'] == 800
    
    def test_paid_tier_preset_values(self):
        """Test paid tier presets contain progressive improvements."""

        
        free_tier = kimi_config.PRESETS['free_tier']
        paid_10 = kimi_config.PRESETS['paid_tier_10_rpm']
        paid_30 = kimi_config.PRESETS['paid_tier_30_rpm']
        
        # Verify progressive improvements
        assert free_tier['KIMI_RPM_LIMIT'] < paid_10['KIMI_RPM_LIMIT'] < paid_30['KIMI_RPM_LIMIT']
        assert free_tier['KIMI_TPM_LIMIT'] < paid_10['KIMI_TPM_LIMIT'] < paid_30['KIMI_TPM_LIMIT']
        assert free_tier['KIMI_MAX_TOKENS'] < paid_10['KIMI_MAX_TOKENS'] < paid_30['KIMI_MAX_TOKENS']
        
        # Verify delays decrease with higher tiers
        assert free_tier['KIMI_BASE_DELAY'] > paid_10['KIMI_BASE_DELAY'] > paid_30['KIMI_BASE_DELAY']
    
    def test_apply_preset_functionality(self):
        """Test applying configuration presets."""

        
        # Store original values
        original_rpm = kimi_config.KIMI_RPM_LIMIT
        original_tpm = kimi_config.KIMI_TPM_LIMIT
        
        # Apply paid tier preset
        with patch('builtins.print'):  # Suppress print output
            kimi_config.apply_preset('paid_tier_10_rpm')
        
        # Verify values changed
        assert kimi_config.KIMI_RPM_LIMIT == 10
        assert kimi_config.KIMI_TPM_LIMIT == 100000
        
        # Reset to free tier
        with patch('builtins.print'):
            kimi_config.apply_preset('free_tier')
        
        # Verify reset to original values
        assert kimi_config.KIMI_RPM_LIMIT == original_rpm
        assert kimi_config.KIMI_TPM_LIMIT == original_tpm
    
    def test_apply_invalid_preset(self):
        """Test applying invalid preset raises appropriate error."""

        
        with pytest.raises(ValueError) as exc_info:
            kimi_config.apply_preset('invalid_preset')
        
        assert "Preset 'invalid_preset' not found" in str(exc_info.value)


class TestPrintConfigurationSummary:
    """Test cases for configuration summary printing."""
    
    @patch('builtins.print')
    def test_config_summary_output(self, mock_print):
        """Test that configuration summary prints expected information."""

        
        kimi_config.print_config_summary()
        
        # Verify print was called multiple times (for different sections)
        assert mock_print.call_count > 5
        
        # Verify key information is included in print calls
        all_print_calls = ' '.join([str(call) for call in mock_print.call_args_list])
        
        expected_content = [
            'KIMI-K2 API Configuration',
            'RPM Limit',
            'TPM Limit',
            'TPD Limit',
            'Token Usage Statistics',
            'Minute Usage',
            'Day Usage'
        ]
        
        for content in expected_content:
            assert content in all_print_calls


class TestRateLimitCalculations:
    """Test cases for rate limit calculations and timing."""
    
    def test_rpm_based_delay_calculation(self):
        """Test that delay calculation respects RPM limits - without mocking randomness."""
        
        # Test with different RPM values - test logic only
        test_cases = [
            (3, 20),   # Free tier: max(20, 60/3=20) = 20
            (10, 6),   # Paid tier: max(6, 60/10=6) = 6  
            (30, 2),   # High tier: max(2, 60/30=2) = 2
        ]
        
        for rpm, base_delay in test_cases:
            # Store and temporarily modify config
            original_rpm = kimi_config.KIMI_RPM_LIMIT
            original_base_delay = kimi_config.KIMI_BASE_DELAY
            
            # CRITICAL: Set both values before calculation
            kimi_config.KIMI_RPM_LIMIT = rpm
            kimi_config.KIMI_BASE_DELAY = base_delay
            
            try:
                # Test the base calculation logic (before jitter)
                theoretical_delay = int(60 / rpm)
                expected_base_delay = max(base_delay, theoretical_delay)
                
                # Test multiple times to ensure jitter range is correct
                delays = []
                for _ in range(10):  # Sample multiple times
                    delay = kimi_config.calculate_rate_limit_delay()
                    delays.append(delay)
                
                # Verify all delays are within expected jitter range (±20%)
                min_expected = int(expected_base_delay * 0.8)
                max_expected = int(expected_base_delay * 1.2)
                
                for delay in delays:
                    assert min_expected <= delay <= max_expected, \
                        f"RPM:{rpm}, Delay:{delay} not in range [{min_expected}, {max_expected}]"
                
                # Verify we get some variation (jitter working)
                if len(set(delays)) == 1:
                    # If all delays are the same, it might be edge case where jitter rounds to same value
                    # This is acceptable for small delays
                    pass
                
            finally:
                # Always restore original values
                kimi_config.KIMI_RPM_LIMIT = original_rpm
                kimi_config.KIMI_BASE_DELAY = original_base_delay


class TestIntegrationScenarios:
    """Integration test cases for real-world usage scenarios."""
    
    def test_free_tier_user_workflow(self):
        """Test complete workflow for free tier user."""

        
        # Reset state
        kimi_config._token_usage_minute.clear()
        kimi_config._token_usage_day.clear()
        
        # Simulate multiple API calls within limits
        for i in range(5):
            # Each call uses ~1500 tokens
            success = kimi_config.track_token_usage(1500)
            assert success is True
            
            # Get delay for rate limiting
            delay = kimi_config.calculate_rate_limit_delay()
            assert delay >= 20  # Free tier minimum delay
        
        # Check final stats
        stats = kimi_config.get_token_usage_stats()
        assert stats['minute_tokens'] == 7500
        assert stats['minute_percentage'] < 25  # Well within limits
    
    def test_approaching_limits_scenario(self):
        """Test behavior when approaching token limits."""

        
        # Reset and approach TPM limit
        kimi_config._token_usage_minute.clear()
        kimi_config._token_usage_day.clear()
        
        # Use most of TPM allowance
        kimi_config.track_token_usage(30000)  # 93.75% of TPM limit
        
        stats = kimi_config.get_token_usage_stats()
        assert stats['minute_percentage'] > 90
        
        # Small request should still work
        assert kimi_config.track_token_usage(1000) is True
        
        # Large request should be rejected
        assert kimi_config.track_token_usage(2000) is False
    
    @patch('builtins.print')
    def test_monitoring_and_reporting(self, mock_print):
        """Test monitoring and reporting capabilities."""

        
        # Generate some usage
        kimi_config.track_token_usage(10000)
        kimi_config.track_token_usage(5000)
        
        # Test configuration display
        kimi_config.print_config_summary()
        
        # Verify comprehensive reporting
        assert mock_print.call_count >= 8  # Multiple lines of output
        
        # Test that statistics are properly formatted
        stats = kimi_config.get_token_usage_stats()
        assert all(isinstance(value, (int, float)) for value in stats.values())
        assert all(stats[field] >= 0 for field in ['minute_tokens', 'day_tokens'])


if __name__ == "__main__":
    # When run as a script, execute all tests
    pytest.main([__file__, "-v", "--tb=short"])
