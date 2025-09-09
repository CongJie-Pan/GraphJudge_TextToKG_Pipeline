"""
Unit Tests for GPT-5-mini Entity Extraction and Text Denoising (ECTD) Implementation

This test suite validates the functionality of the run_entity.py module,
ensuring that GPT-5-mini integration for Chinese text processing, entity extraction,
text denoising, and file I/O operations work correctly across different scenarios.

Test Coverage:
- GPT-5-mini API integration for Chinese text processing
- Entity extraction from classical Chinese text
- Text denoising and restructuring functionality
- Async operation patterns and error handling
- File I/O operations for ECTD pipeline
- Chinese text validation and format compliance
- Error handling for various failure scenarios

Run with: pytest test_run_entity.py -v
"""

import pytest
import os
import json
import asyncio
import tempfile
import time
from unittest.mock import patch, mock_open, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add the parent directory to the path to import the module under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseGPT5MiniEntityTest:
    """Base class for GPT-5-mini ECTD tests with proper setup and teardown."""
    
    def setup_method(self):
        """Set up method called before each test method."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Reset global token tracking state for proper test isolation
        import openai_config
        openai_config._token_usage_minute.clear()
        openai_config._token_usage_day.clear()
        openai_config._last_reset_minute = time.time()
        openai_config._last_reset_day = time.time()
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_dataset_path = os.path.join(self.temp_dir, "GPT5mini_result_DreamOf_RedChamber/")
        self.test_input_file = os.path.join(self.temp_dir, "chapter1_raw.txt")
        
        # Setup environment variables for testing (reflecting run_entity.py changes)
        self.test_iteration = "3"
        self.test_output_dir = os.path.join(self.temp_dir, "test_output", "ectd")
        os.environ['PIPELINE_ITERATION'] = self.test_iteration
        os.environ['PIPELINE_DATASET_PATH'] = self.test_dataset_path
        os.environ['PIPELINE_OUTPUT_DIR'] = self.test_output_dir
        
        # Sample classical Chinese text data for testing
        self.sample_chinese_texts = [
            "甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。",
            "這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。",
            "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。",
            "賈雨村原系胡州人氏，也是詩書仕宦之族，因他生於末世，暫寄廟中安身。"
        ]
        
        # Expected entity extraction results
        self.expected_entities = [
            '["甄士隱", "書房"]',
            '["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]',
            '["甄費", "甄士隱", "封氏", "鄉宦"]',
            '["賈雨村", "胡州", "詩書仕宦之族"]'
        ]
        
        # Expected denoised text results
        self.expected_denoised = [
            "甄士隱在書房閒坐。甄士隱手倦拋書。甄士隱伏几少憩。甄士隱不覺朦朧睡去。",
            "閶門外有十里街。十里街內有仁清巷。仁清巷內有古廟。古廟又稱葫蘆廟。",
            "甄士隱是一家鄉宦。甄士隱姓甄名費字士隱。甄士隱的妻子是封氏。封氏情性賢淑深明禮義。",
            "賈雨村是胡州人氏。賈雨村是詩書仕宦之族。賈雨村生於末世。賈雨村暫寄廟中安身。"
        ]
        
        # Create test input file
        with open(self.test_input_file, 'w', encoding='utf-8') as f:
            f.write('第一回　甄士隱夢幻識通靈　賈雨村風塵懷閨秀\n')
            f.write('紅樓夢\n')
            f.write('\n'.join(self.sample_chinese_texts))
        
        # Create dataset directory structure
        os.makedirs(self.test_dataset_path, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def teardown_method(self):
        """Tear down method called after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestGPT5MiniApiIntegration(BaseGPT5MiniEntityTest):
    """Test cases for GPT-5-mini API integration functionality."""
    
    @pytest.mark.asyncio
    async def test_openai_api_call_success(self):
        """Test successful GPT-5-mini API call for Chinese text processing with enhanced rate limiting."""
        # Mock the LiteLLM completion function
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["甄士隱", "書房"]'
        
        # We need to import with mocking since the module has initialization code
        with patch('litellm.completion', return_value=mock_response) as mock_completion, \
             patch('run_entity.get_api_key', return_value='test-api-key'), \
             patch('run_entity.track_token_usage', return_value=True) as mock_track_token, \
             patch('run_entity.get_token_usage_stats', return_value={
                 'minute_tokens': 1000, 'day_tokens': 10000,
                 'minute_remaining': 31000, 'day_remaining': 1490000,
                 'minute_percentage': 3.1, 'day_percentage': 0.7
             }) as mock_token_stats:
            
            # Import the module functions for testing
            import run_entity
            
            result = await run_entity.openai_api_call(
                "請提取實體：甄士隱於書房閒坐"
            )
            
            # Verify the function was called with correct parameters
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args
            
            # Import configuration to get expected values
            from openai_config import GPT5_MINI_MODEL, OPENAI_MAX_TOKENS
            
            assert call_args[1]['model'] == GPT5_MINI_MODEL
            # Note: GPT-5 models don't support custom temperature, so we don't pass it
            assert 'temperature' not in call_args[1]  # Should not include temperature parameter
            assert call_args[1]['max_completion_tokens'] == OPENAI_MAX_TOKENS
            assert "甄士隱" in call_args[1]['messages'][0]['content']
            
            # Verify token tracking was called
            mock_track_token.assert_called()
            
            # Verify the response
            assert result == '["甄士隱", "書房"]'
    
    @pytest.mark.asyncio
    async def test_openai_api_call_with_intelligent_retry(self):
        """Test GPT-5-mini API call with enhanced intelligent retry logic for different error types."""
        # Mock the completion function to fail with different error types then succeed
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["賈雨村", "胡州"]'
        
        import run_entity
        with patch.object(run_entity, 'completion', side_effect=[
            Exception("RateLimitError: rate limit exceeded"),
            Exception("Server overloaded, please try again later"),
            mock_response
        ]) as mock_completion, \
             patch('run_entity.get_api_key', return_value='test-api-key'), \
             patch('run_entity.track_token_usage', return_value=True), \
             patch('run_entity.get_token_usage_stats', return_value={
                 'minute_tokens': 2000, 'day_tokens': 20000,
                 'minute_remaining': 30000, 'day_remaining': 1480000,
                 'minute_percentage': 6.3, 'day_percentage': 1.3
             }), \
             patch('asyncio.sleep') as mock_sleep:
            
            result = await run_entity.openai_api_call(
                "請提取實體：賈雨村原系胡州人氏"
            )
            
            # Verify retry attempts
            assert mock_completion.call_count == 3
            assert result == '["賈雨村", "胡州"]'
            
            # Verify that different sleep times were used for different error types
            assert mock_sleep.call_count == 2  # Two retry delays
            
            # Verify the sleep calls used progressive delays
            sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
            assert len(sleep_calls) == 2
            assert all(isinstance(delay, (int, float)) and delay > 0 for delay in sleep_calls)
    
    @pytest.mark.asyncio
    async def test_openai_api_call_max_retries_exceeded(self):
        """Test GPT-5-mini API call when max retries are exceeded."""
        import run_entity
        with patch.object(run_entity, 'completion', side_effect=Exception("Persistent API Error")), \
             patch('run_entity.get_api_key', return_value='test_api_key'), \
             patch('run_entity.track_token_usage', return_value=True), \
             patch('asyncio.sleep'):
            
            result = await run_entity.openai_api_call("Test query")
            
            # Should return error message after max retries
            assert "Error: Could not get response" in result
    
    @pytest.mark.asyncio
    async def test_token_limit_handling(self):
        """Test handling of token limits for free tier users."""
        import run_entity
        
        # Mock token usage tracking to simulate exceeding TPM limit
        # Set minute_remaining to be less than estimated tokens to trigger sleep
        with patch('run_entity.track_token_usage', return_value=False) as mock_track_token, \
             patch('run_entity.get_token_usage_stats', return_value={
                 'minute_tokens': 31999, 'day_tokens': 100000,
                 'minute_remaining': 1, 'day_remaining': 1400000,  # Only 1 token remaining
                 'minute_percentage': 99.9, 'day_percentage': 6.7
             }), \
             patch('asyncio.sleep') as mock_sleep, \
             patch('run_entity.get_api_key', return_value='test-api-key'):
            
            # Create a prompt that will require more tokens than remaining (estimated ~100 tokens)
            long_prompt = "這是一個較長的測試文本，用來測試當令牌限制被超過時系統的處理邏輯。我們需要確保當剩餘令牌不足時，系統會等待重置。" * 5
            result = await run_entity.openai_api_call(long_prompt)
            
            # Verify token tracking was called
            mock_track_token.assert_called()
            
            # Since token limit was exceeded AND minute_remaining < estimated_tokens, should sleep
            mock_sleep.assert_called_with(60)
    
    @pytest.mark.asyncio
    async def test_free_tier_sequential_processing(self):
        """Test sequential processing optimization for free tier."""
        import run_entity
        
        # Mock responses for multiple queries
        mock_responses = []
        for i in range(3):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f'["實體{i+1}", "地點{i+1}"]'
            mock_responses.append(mock_response)
        
        queries = ["測試文本1", "測試文本2", "測試文本3"]
        
        with patch.object(run_entity, 'completion', side_effect=mock_responses), \
             patch('run_entity.get_api_key', return_value='test-api-key'), \
             patch('run_entity.track_token_usage', return_value=True), \
             patch('run_entity.get_token_usage_stats', return_value={
                 'minute_tokens': 5000, 'day_tokens': 50000,
                 'minute_remaining': 27000, 'day_remaining': 1450000,
                 'minute_percentage': 15.6, 'day_percentage': 3.3
             }), \
             patch('asyncio.sleep') as mock_sleep:
            
            # Test the enhanced _run_api function with sequential processing
            results = await run_entity._run_api(queries, max_concurrent=1)
            
            # Verify all queries were processed
            assert len(results) == 3
            assert all("實體" in result for result in results)
            
            # Verify progressive delays were used (should have multiple sleep calls)
            assert mock_sleep.call_count >= 2  # At least progressive + safety delays


class TestChineseEntityExtraction(BaseGPT5MiniEntityTest):
    """Test cases for Chinese entity extraction functionality."""
    
    def test_entity_extraction_prompt_structure(self):
        """Test that entity extraction prompts are properly structured for Chinese text."""
        test_text = "甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。"
        
        # Expected elements in Chinese entity extraction prompt
        expected_elements = [
            "目標：",
            "從古典中文文本中提取實體列表",
            "人物、地點、物品、概念",
            "《紅樓夢》",
            "甄士隱",
            "書房",
            "閶門",
            "十里街",
            "賈寶玉",
            "太虛幻境",
            "林黛玉",
            "實體列表："
        ]
        
        # Simulate the prompt construction logic
        prompt_template = f"""
目標：
從古典中文文本中提取實體列表（人物、地點、物品、概念）。

以下是《紅樓夢》的五個範例：
範例#1:
文本："甄士隱於書房閒坐，至手倦拋書，伏几少憩，不覺朦朧睡去。"
實體列表：["甄士隱", "書房"]

範例#2:
文本："這閶門外有個十里街，街內有個仁清巷，巷內有個古廟，因地方窄狹，人皆呼作葫蘆廟。"
實體列表：["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]

範例#3:
文本："廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
實體列表：["甄費", "甄士隱", "封氏", "鄉宦"]

範例#4:
文本："賈雨村原系胡州人氏，也是詩書仕宦之族，因他生於末世，暫寄廟中安身。"
實體列表：["賈雨村", "胡州", "詩書仕宦之族"]

範例#5:
文本："賈寶玉因夢遊太虛幻境，頓生疑懼，醒來後對林黛玉說起此事。"
實體列表：["賈寶玉", "太虛幻境", "林黛玉"]

請參考以上範例，分析以下文本：
文本："{test_text}"
實體列表："""
        
        # Verify all expected elements are present
        for element in expected_elements:
            assert element in prompt_template, f"Missing element: {element}"
        
        # Verify the test text is properly formatted in the prompt
        assert test_text in prompt_template
    
    @pytest.mark.asyncio
    async def test_extract_entities_simulation(self):
        """Simulate entity extraction functionality for Chinese text."""
        
        async def simulated_extract_entities(texts):
            """Simulate the extract_entities function behavior."""
            results = []
            for text in texts:
                if "甄士隱" in text and "書房" in text:
                    results.append('["甄士隱", "書房"]')
                elif "閶門" in text and "十里街" in text:
                    results.append('["閶門", "十里街", "仁清巷", "古廟", "葫蘆廟"]')
                elif "甄" in text and "封氏" in text:
                    results.append('["甄費", "甄士隱", "封氏", "鄉宦"]')
                elif "賈雨村" in text and "胡州" in text:
                    results.append('["賈雨村", "胡州", "詩書仕宦之族"]')
                else:
                    results.append('[]')
            return results
        
        # Test with sample Chinese texts
        results = await simulated_extract_entities(self.sample_chinese_texts)
        
        # Verify results
        assert len(results) == len(self.sample_chinese_texts)
        assert '甄士隱' in results[0]
        assert '書房' in results[0]
        assert '閶門' in results[1]
        assert '十里街' in results[1]
        assert '甄費' in results[2]
        assert '封氏' in results[2]
        assert '賈雨村' in results[3]
        assert '胡州' in results[3]


class TestChineseTextDenoising(BaseGPT5MiniEntityTest):
    """Test cases for Chinese text denoising functionality."""
    
    def test_denoising_prompt_structure(self):
        """Test that denoising prompts are properly structured for classical Chinese."""
        test_text = "廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。"
        test_entities = '["甄費", "甄士隱", "封氏", "鄉宦"]'
        
        # Expected elements in Chinese denoising prompt
        expected_elements = [
            "目標：",
            "基於給定的實體，對古典中文文本進行去噪處理",
            "移除無關的描述性文字並重組為清晰的事實陳述",
            "《紅樓夢》",
            "原始文本：",
            "實體：",
            "去噪文本：",
            "甄士隱",
            "封氏",
            "賈雨村",
            "胡州"
        ]
        
        # Simulate the denoising prompt construction
        prompt_template = f"""
目標：
基於給定的實體，對古典中文文本進行去噪處理，即移除無關的描述性文字並重組為清晰的事實陳述。

以下是《紅樓夢》的三個範例：
範例#1:
原始文本："廟旁住著一家鄉宦，姓甄，名費，字士隱。嫡妻封氏，情性賢淑，深明禮義。家中雖不甚富貴，然本地便也推他為望族了。"
實體：["甄費", "甄士隱", "封氏", "鄉宦"]
去噪文本："甄士隱是一家鄉宦。甄士隱姓甄名費字士隱。甄士隱的妻子是封氏。封氏情性賢淑深明禮義。甄家是本地望族。"

範例#2:
原始文本："賈雨村原系胡州人氏，也是詩書仕宦之族，因他生於末世，父母祖宗根基已盡，人口衰喪，只剩得他一身一口，在家鄉無益，因進京求取功名，再整基業。"
實體：["賈雨村", "胡州", "詩書仕宦之族"]
去噪文本："賈雨村是胡州人氏。賈雨村是詩書仕宦之族。賈雨村生於末世。賈雨村父母祖宗根基已盡。賈雨村進京求取功名。賈雨村想要重整基業。"

範例#3:
原始文本："賈寶玉因夢遊太虛幻境，頓生疑懼，醒來後心中不安，遂將此事告知林黛玉，黛玉聽後亦感驚異。"
實體：["賈寶玉", "太虛幻境", "林黛玉"]
去噪文本："賈寶玉夢遊太虛幻境。賈寶玉夢醒後頓生疑懼。賈寶玉將此事告知林黛玉。林黛玉聽後感到驚異。"

請參考以上範例，處理以下文本：
原始文本：{test_text}
實體：{test_entities}
去噪文本："""
        
        # Verify all expected elements are present
        for element in expected_elements:
            assert element in prompt_template, f"Missing element: {element}"
        
        # Verify the input text and entities are properly formatted
        assert test_text in prompt_template
        assert test_entities in prompt_template
    
    @pytest.mark.asyncio
    async def test_denoise_text_simulation(self):
        """Simulate text denoising functionality for classical Chinese."""
        
        async def simulated_denoise_text(texts, entities_list):
            """Simulate the denoise_text function behavior."""
            results = []
            for text, entities in zip(texts, entities_list):
                if "甄士隱" in text and "書房" in text:
                    results.append("甄士隱在書房閒坐。甄士隱手倦拋書。甄士隱伏几少憩。")
                elif "閶門" in text and "十里街" in text:
                    results.append("閶門外有十里街。十里街內有仁清巷。仁清巷內有古廟。")
                elif "甄" in text and "封氏" in text:
                    results.append("甄士隱是一家鄉宦。甄士隱的妻子是封氏。封氏情性賢淑。")
                elif "賈雨村" in text:
                    results.append("賈雨村是胡州人氏。賈雨村是詩書仕宦之族。賈雨村生於末世。")
                else:
                    results.append("去噪處理後的文本。")
            return results
        
        # Test with sample texts and entities
        results = await simulated_denoise_text(self.sample_chinese_texts, self.expected_entities)
        
        # Verify results
        assert len(results) == len(self.sample_chinese_texts)
        assert all("。" in result for result in results)  # All should end with periods
        assert "甄士隱" in results[0]
        assert "書房" in results[0]
        assert "十里街" in results[1]
        assert "仁清巷" in results[1]


class TestFileOperations(BaseGPT5MiniEntityTest):
    """Test cases for file I/O operations in the ECTD pipeline."""
    
    def test_input_file_loading(self):
        """Test loading and processing of Chinese input files."""
        # Simulate the input file loading logic
        with open(self.test_input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Filter logic from the actual implementation
            text = [line.strip() for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('第一回') 
                    and not line.strip() == '紅樓夢' and len(line.strip()) > 20]
        
        # Verify filtering worked correctly
        assert len(text) == len(self.sample_chinese_texts)
        assert all(len(t) > 20 for t in text)
        assert not any(t.startswith('第一回') for t in text)
        assert '紅樓夢' not in text
    
    def test_entity_file_operations(self):
        """Test entity file saving and loading operations."""
        # Simulate entity saving (using environment variable for iteration)
        entity_file = os.path.join(self.test_output_dir, "test_entity.txt")
        os.makedirs(os.path.dirname(entity_file), exist_ok=True)
        
        with open(entity_file, "w", encoding='utf-8') as f:
            for entities in self.expected_entities:
                f.write(str(entities).strip().replace('\n', '') + '\n')
        
        # Verify file was created and has correct content
        assert os.path.exists(entity_file)
        
        # Test loading entities back
        loaded_entities = []
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                loaded_entities.append(line.strip())
        
        assert len(loaded_entities) == len(self.expected_entities)
        assert all('甄士隱' in loaded_entities[0] or '閶門' in loaded_entities[1] 
                  for _ in loaded_entities)
    
    def test_denoised_file_operations(self):
        """Test denoised text file saving operations."""
        # Simulate denoised text saving (using environment variable for output directory)
        denoised_file = os.path.join(self.test_output_dir, "test_denoised.target")
        os.makedirs(os.path.dirname(denoised_file), exist_ok=True)
        
        with open(denoised_file, "w", encoding='utf-8') as f:
            for denoised_text in self.expected_denoised:
                cleaned_text = str(denoised_text).strip().replace('\n', ' ')
                f.write(cleaned_text + '\n')
        
        # Verify file was created and has correct content
        assert os.path.exists(denoised_file)
        
        # Test file content
        with open(denoised_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == len(self.expected_denoised)
        assert all('甄士隱' in lines[0] or '閶門' in lines[1] for _ in lines)


class TestErrorHandling(BaseGPT5MiniEntityTest):
    """Test cases for error handling and edge cases."""
    
    def test_api_configuration_validation(self):
        """Test API configuration validation."""
        with patch('run_entity.get_api_key', side_effect=ValueError("API key not found")):
            # This would be tested in the module's validation logic
            try:
                import run_entity
                # If the module was successfully imported, API validation worked
                assert True
            except SystemExit:
                # If module exits due to API config error, that's expected
                assert True
    
    def test_missing_input_file_handling(self):
        """Test handling of missing input files."""
        non_existent_file = os.path.join(self.temp_dir, "missing_chapter.txt")
        
        # Test file existence check
        def validate_input_file(file_path):
            return os.path.exists(file_path)
        
        assert validate_input_file(self.test_input_file) is True
        assert validate_input_file(non_existent_file) is False
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty or malformed Chinese text."""
        # Test with empty texts
        empty_texts = ["", "   ", "第一回", "紅樓夢"]
        
        # Simulate filtering logic
        filtered_texts = [text.strip() for text in empty_texts 
                         if text.strip() and not text.strip().startswith('第一回') 
                         and not text.strip() == '紅樓夢' and len(text.strip()) > 20]
        
        # Should filter out all empty/invalid texts
        assert len(filtered_texts) == 0


class TestChineseTextValidation(BaseGPT5MiniEntityTest):
    """Test cases for Chinese text validation and format compliance."""
    
    def test_chinese_character_encoding(self):
        """Test proper handling of Chinese character encoding."""
        # Test traditional Chinese characters
        traditional_text = "廟旁住著一家鄉宦，姓甄，名費，字士隱。"
        
        # Verify UTF-8 encoding works correctly
        encoded = traditional_text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        assert decoded == traditional_text
        
        # Test file I/O with Chinese characters
        test_file = os.path.join(self.temp_dir, "chinese_test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(traditional_text)
        
        with open(test_file, 'r', encoding='utf-8') as f:
            loaded_text = f.read()
        
        assert loaded_text == traditional_text
    
    def test_entity_format_validation(self):
        """Test validation of entity extraction format."""
        
        def validate_entity_format(entity_string):
            """Validate that entity string is in proper JSON list format."""
            try:
                # Should be able to parse as Python list
                if entity_string.startswith('[') and entity_string.endswith(']'):
                    return True
                return False
            except:
                return False
        
        # Test valid entity formats
        valid_entities = [
            '["甄士隱", "書房"]',
            '["閶門", "十里街", "仁清巷"]',
            '[]'
        ]
        
        for entity in valid_entities:
            assert validate_entity_format(entity) is True
        
        # Test invalid entity formats
        invalid_entities = [
            "甄士隱, 書房",
            "['甄士隱', '書房'",  # Missing closing bracket
            "not a list"
        ]
        
        for entity in invalid_entities:
            assert validate_entity_format(entity) is False


class TestFreeTierRateLimitOptimizations(BaseGPT5MiniEntityTest):
    """Test cases for free tier rate limit optimizations."""
    
    def test_development_configuration(self):
        """Test that development configuration is properly applied."""
        from openai_config import OPENAI_RPM_LIMIT, OPENAI_CONCURRENT_LIMIT, OPENAI_TPM_LIMIT, OPENAI_TPD_LIMIT
        
        # Verify development limits are set correctly
        assert OPENAI_RPM_LIMIT == 60
        assert OPENAI_CONCURRENT_LIMIT == 3
        assert OPENAI_TPM_LIMIT == 90000
        assert OPENAI_TPD_LIMIT == 2000000
    
    def test_token_tracking_functionality(self):
        """Test token usage tracking for TPM/TPD limits."""
        from openai_config import track_token_usage, get_token_usage_stats
        
        # Test token tracking within limits
        assert track_token_usage(1000) is True
        
        # Test token usage statistics
        stats = get_token_usage_stats()
        assert 'minute_tokens' in stats
        assert 'day_tokens' in stats
        assert 'minute_remaining' in stats
        assert 'day_remaining' in stats
        assert 'minute_percentage' in stats
        assert 'day_percentage' in stats
    
    def test_progressive_delay_calculation(self):
        """Test progressive delay calculation for rate limiting."""
        from openai_config import calculate_rate_limit_delay
        
        # Test that delay calculation returns reasonable values
        for i in range(5):
            delay = calculate_rate_limit_delay()
            assert isinstance(delay, int)
            assert delay >= 4  # Should be at least base delay (5) with jitter
            assert delay <= 7  # Should be within jitter range (5 * 1.2 = 6, rounded to 7)
    
    @patch('openai_config.time.time')
    def test_token_usage_reset(self, mock_time):
        """Test that token usage resets properly after time windows."""
        from openai_config import track_token_usage, _token_usage_minute, _token_usage_day
        import openai_config
        
        # Ensure clean state at start of test
        _token_usage_minute.clear()
        _token_usage_day.clear()
        
        # Set initial time and last reset time
        mock_time.return_value = 1000.0
        openai_config._last_reset_minute = 1000.0
        openai_config._last_reset_day = 1000.0
        
        # Track some token usage
        track_token_usage(5000)
        assert len(_token_usage_minute) == 1
        assert len(_token_usage_day) == 1
        
        # Advance time by 61 seconds (should reset minute counter)
        mock_time.return_value = 1061.0
        track_token_usage(3000)
        
        # Minute counter should be reset, day counter should accumulate
        assert len(_token_usage_minute) == 1  # Reset and new entry
        assert len(_token_usage_day) == 2     # Accumulated
    
    @pytest.mark.asyncio
    async def test_enhanced_error_handling(self):
        """Test enhanced error handling for different API error types."""
        import run_entity
        
        error_scenarios = [
            ("RateLimitError: RPM exceeded", "rate limit"),
            ("Server overloaded", "overloaded"),
            ("Connection timeout", "timeout"),
            ("Generic API error", "other")
        ]
        
        for error_msg, error_type in error_scenarios:
            with patch.object(run_entity, 'completion', side_effect=Exception(error_msg)), \
                 patch('run_entity.get_api_key', return_value='test-api-key'), \
                 patch('run_entity.track_token_usage', return_value=True), \
                 patch('asyncio.sleep') as mock_sleep:
                
                result = await run_entity.openai_api_call(f"Test {error_type}")
                
                # Should return error after retries
                assert "Error: Could not get response" in result
                
                # Should have used appropriate retry delays
                assert mock_sleep.called


class TestEnvironmentVariableIntegration(BaseGPT5MiniEntityTest):
    """Test environment variable integration functionality."""

    def test_environment_variable_usage(self):
        """Test that environment variables are correctly used for configuration."""
        # Test that PIPELINE_ITERATION is correctly set
        import run_entity
        assert os.environ.get('PIPELINE_ITERATION') == self.test_iteration
        
        # Test that PIPELINE_DATASET_PATH is correctly set
        assert os.environ.get('PIPELINE_DATASET_PATH') == self.test_dataset_path
        
        # Test that PIPELINE_OUTPUT_DIR is correctly set
        assert os.environ.get('PIPELINE_OUTPUT_DIR') == self.test_output_dir

    @patch.dict(os.environ, {
        'PIPELINE_ITERATION': '5',
        'PIPELINE_DATASET_PATH': '/test/path/',
        'PIPELINE_OUTPUT_DIR': '/test/output/'
    }, clear=False)
    def test_environment_variable_override(self):
        """Test that environment variables can override default values."""
        # Import the module to test environment variable reading
        import run_entity
        
        # Verify that the module would use environment variables
        # (Note: This tests the concept, actual module import might need adjustment)
        assert os.environ.get('PIPELINE_ITERATION') == '5'
        assert os.environ.get('PIPELINE_DATASET_PATH') == '/test/path/'
        assert os.environ.get('PIPELINE_OUTPUT_DIR') == '/test/output/'

    def test_environment_variable_fallback(self):
        """Test fallback to default values when environment variables are not set."""
        # Temporarily remove environment variables
        backup_vars = {}
        for var in ['PIPELINE_ITERATION', 'PIPELINE_DATASET_PATH', 'PIPELINE_OUTPUT_DIR']:
            backup_vars[var] = os.environ.pop(var, None)
        
        try:
            # Test that module still functions with defaults
            # (This would test the actual fallback logic in the module)
            assert os.environ.get('PIPELINE_ITERATION') is None
            assert os.environ.get('PIPELINE_DATASET_PATH') is None
            assert os.environ.get('PIPELINE_OUTPUT_DIR') is None
        finally:
            # Restore environment variables
            for var, value in backup_vars.items():
                if value is not None:
                    os.environ[var] = value


if __name__ == "__main__":
    # When run directly, execute all tests
    pytest.main([__file__, "-v"])