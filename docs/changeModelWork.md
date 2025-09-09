# LLM Model Change Plan: KIMI-K2 to GPT-5-mini

## Code Analysis Summary

### 1. Current Implementation Analysis

#### run_triple.py (KIMI-K2 Model)
- **Model**: Uses `moonshot/kimi-k2-0711-preview` via LiteLLM
- **Configuration**: Imports from `kimi_config.py` with KIMI-specific settings
- **Rate Limiting**: Complex rate limiting for KIMI free tier (RPM limits, token tracking)
- **API Call**: Uses `completion()` with KIMI model parameters
- **Error Handling**: KIMI-specific error handling and retry logic

#### run_entity.py (GPT-5-mini Model) 
- **Model**: Uses `gpt-5-mini` via LiteLLM
- **Configuration**: Imports from `openai_config.py` with OpenAI-specific settings
- **Rate Limiting**: OpenAI rate limiting (RPM limits, token tracking)
- **API Call**: Uses `completion()` with GPT-5-mini parameters
- **Error Handling**: OpenAI-specific error handling and retry logic

#### test_run_triple.py
- **Test Coverage**: Comprehensive testing of KIMI-K2 API integration
- **Mocking**: Mocks KIMI-specific configurations and responses
- **Validation**: Tests KIMI model parameters and error handling

### 2. LiteLLM Documentation Analysis

#### GPT-5-mini Model Usage
```python
# Direct model usage
response = completion(
    model="gpt-5-mini",
    messages=messages
)

# Environment setup
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

#### Key Differences from KIMI
- **Model Name**: `gpt-5-mini` (vs `moonshot/kimi-k2-0711-preview`)
- **API Base**: Uses OpenAI API directly (vs Moonshot API)
- **Rate Limits**: OpenAI's higher limits vs KIMI's restrictive free tier
- **Token Parameters**: `max_completion_tokens` for GPT-5 models

## Revision Plan

### Phase 1: Configuration File Updates

#### 1.1 Update kimi_config.py → openai_config.py
**File**: `Miscellaneous/KgGen/GraphJudge/chat/kimi_config.py`
**Changes**:
- Rename file to `openai_config.py`
- Replace KIMI model constants with GPT-5-mini equivalents
- Update rate limiting parameters for OpenAI (higher limits)
- Remove KIMI-specific token tracking complexity

**Key Changes**:
```python
# Before (KIMI)
KIMI_MODEL = "moonshot/kimi-k2-0711-preview"
KIMI_RPM_LIMIT = 10  # Very restrictive
KIMI_MAX_TOKENS = 800

# After (GPT-5-mini)
GPT5_MINI_MODEL = "gpt-5-mini"
OPENAI_RPM_LIMIT = 3500  # Much higher
OPENAI_MAX_TOKENS = 4000  # Higher token limit
```

#### 1.2 Update config.py
**File**: `Miscellaneous/KgGen/GraphJudge/chat/config.py`
**Changes**:
- Replace `get_moonshot_api_config()` with `get_openai_api_config()`
- Update environment variable names
- Ensure OpenAI API key handling

### Phase 2: Core Implementation Updates

#### 2.1 Update run_triple.py Model Configuration
**File**: `Miscellaneous/KgGen/GraphJudge/chat/run_triple.py`
**Changes**:
- Import from `openai_config` instead of `kimi_config`
- Replace KIMI model with GPT-5-mini
- Simplify rate limiting (remove KIMI-specific complexity)
- Update API call parameters for GPT-5-mini

**Key Changes**:
```python
# Before
from kimi_config import (
    KIMI_RPM_LIMIT, KIMI_CONCURRENT_LIMIT, KIMI_RETRY_ATTEMPTS, 
    KIMI_BASE_DELAY, KIMI_TEMPERATURE, KIMI_MAX_TOKENS, KIMI_MODEL,
    calculate_rate_limit_delay, print_config_summary, track_token_usage, get_token_usage_stats
)

# After
from openai_config import (
    OPENAI_RPM_LIMIT, OPENAI_CONCURRENT_LIMIT, OPENAI_RETRY_ATTEMPTS, 
    OPENAI_BASE_DELAY, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, GPT5_MINI_MODEL,
    calculate_rate_limit_delay, get_api_config_summary, track_token_usage, get_token_usage_stats
)
```

#### 2.2 Update API Call Function
**Function**: `kimi_api_call()` → `openai_api_call()`
**Changes**:
- Rename function to reflect OpenAI usage
- Update model parameter to `GPT5_MINI_MODEL`
- Simplify rate limiting logic (remove KIMI-specific delays)
- Update error handling for OpenAI API errors

#### 2.3 Update Rate Limiting Logic
**Changes**:
- Remove KIMI-specific progressive delays (+3s per request)
- Use OpenAI's higher rate limits for better performance
- Simplify token tracking (OpenAI has higher limits)
- Remove free tier restrictions

### Phase 3: Test Suite Updates

#### 3.1 Update test_run_triple.py
**File**: `Miscellaneous/KgGen/GraphJudge/chat/unit_test/test_run_triple.py`
**Changes**:
- Update test class names from `TestKimi*` to `TestOpenAI*`
- Replace KIMI model mocks with GPT-5-mini mocks
- Update expected model parameters in assertions
- Remove KIMI-specific rate limiting tests
- Add OpenAI-specific test cases

**Key Test Updates**:
```python
# Before
assert call_args[1]['model'] == "moonshot/kimi-k2-0711-preview"
assert call_args[1]['max_tokens'] == 800

# After  
assert call_args[1]['model'] == "gpt-5-mini"
assert call_args[1]['max_completion_tokens'] == 4000
```

#### 3.2 Update Test Configuration
**Changes**:
- Replace KIMI API key mocks with OpenAI API key mocks
- Update test dataset paths (remove KIMI prefix)
- Update environment variable names in tests

### Phase 4: Documentation and Comments

#### 4.1 Update File Headers
**Changes**:
- Replace "KIMI-K2" references with "GPT-5-mini"
- Update model descriptions and capabilities
- Remove KIMI-specific limitations mentions

#### 4.2 Update Function Documentation
**Changes**:
- Update docstrings to reflect GPT-5-mini usage
- Remove KIMI-specific parameter explanations
- Add GPT-5-mini specific information

### Phase 5: Environment and Dependencies

#### 5.1 Update Environment Variables
**Changes**:
- Replace `MOONSHOT_API_KEY` with `OPENAI_API_KEY`
- Update dataset path prefixes (remove KIMI references)
- Ensure OpenAI API key is properly configured

#### 5.2 Update Requirements
**Changes**:
- Ensure `litellm` supports GPT-5-mini
- Verify OpenAI API access and billing setup
- Update any KIMI-specific dependencies

## Implementation Strategy

### Condense Principle Application
1. **Minimal Changes**: Only modify what's necessary for model change
2. **Preserve Structure**: Keep existing function signatures and flow
3. **Remove Complexity**: Eliminate KIMI-specific rate limiting complexity
4. **Maintain Quality**: Keep existing error handling and validation logic

### Step-by-Step Implementation
1. **Backup Current Code**: Create backup of working KIMI implementation
2. **Update Configuration**: Modify config files first
3. **Update Core Logic**: Change model references and API calls
4. **Update Tests**: Modify test suite to match new implementation
5. **Test Integration**: Verify end-to-end functionality
6. **Update Documentation**: Reflect changes in comments and headers

### Risk Mitigation
1. **Incremental Changes**: Make changes in small, testable increments
2. **Preserve Functionality**: Ensure core pipeline logic remains intact
3. **Comprehensive Testing**: Test all major functions after changes
4. **Rollback Plan**: Keep backup for quick rollback if issues arise

## Expected Benefits

### Performance Improvements
- **Higher Rate Limits**: OpenAI's 3500 RPM vs KIMI's 10 RPM
- **Better Token Limits**: 4000 tokens vs 800 tokens
- **Faster Processing**: Reduced rate limiting delays
- **Improved Reliability**: More stable API service

### Code Simplification
- **Removed Complexity**: Eliminate KIMI-specific rate limiting logic
- **Cleaner Code**: Simpler error handling and retry logic
- **Better Maintainability**: Standard OpenAI API patterns
- **Reduced Dependencies**: Fewer custom rate limiting functions

### Quality Improvements
- **Better Chinese Understanding**: GPT-5-mini's superior Chinese language capabilities
- **More Consistent Output**: Better structured responses
- **Enhanced Error Handling**: Standard OpenAI error patterns
- **Improved Testing**: Cleaner test structure without KIMI complexity

## Success Criteria

### Functional Requirements
- [ ] Pipeline runs successfully with GPT-5-mini model
- [ ] All existing functionality preserved
- [ ] Rate limiting works correctly with OpenAI limits
- [ ] Error handling functions properly
- [ ] Tests pass with new model configuration

### Performance Requirements
- [ ] Processing speed improved (no more +3s delays)
- [ ] Higher throughput (3500 RPM vs 10 RPM)
- [ ] Better token utilization (4000 vs 800 tokens)
- [ ] Reduced API call failures

### Quality Requirements
- [ ] Code complexity reduced
- [ ] Maintainability improved
- [ ] Documentation updated
- [ ] Test coverage maintained

## Timeline Estimate

- **Phase 1 (Configuration)**: 1-2 hours
- **Phase 2 (Core Updates)**: 2-3 hours  
- **Phase 3 (Test Updates)**: 2-3 hours
- **Phase 4 (Documentation)**: 1 hour
- **Phase 5 (Environment)**: 1 hour
- **Testing & Validation**: 2-3 hours

**Total Estimated Time**: 9-13 hours

## Next Steps

1. **Review Plan**: Validate approach with team
2. **Create Backup**: Backup current working implementation
3. **Start Implementation**: Begin with Phase 1 (Configuration)
4. **Iterative Testing**: Test each phase before proceeding
5. **Final Validation**: End-to-end testing of complete pipeline
6. **Documentation Update**: Update all related documentation
