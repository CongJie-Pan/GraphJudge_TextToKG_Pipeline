# GraphJudge Model Migration Plan: Gemini to Perplexity API

## Overview

This document outlines the comprehensive plan to migrate the GraphJudge system from using Google's Gemini API to Perplexity API's sonar-reasoning model. The migration aims to leverage Perplexity's advanced reasoning capabilities while maintaining the existing functionality and compatibility with the evaluation pipeline.

## Current System Analysis

### 1. Current Gemini Implementation Summary

The `run_gemini_gj.py` file implements a sophisticated Graph Judge system with the following key components:

- **GeminiGraphJudge Class**: Main adapter that integrates with GeminiRAG_ground system
- **Dual Processing Modes**: Standard binary judgment and explainable reasoning modes
- **Gold Label Bootstrapping**: Two-stage process using RapidFuzz and LLM semantic evaluation
- **Async Processing**: Concurrent request handling with rate limiting
- **Comprehensive Logging**: Terminal logging with timestamped files
- **Error Handling**: Robust error handling with retry mechanisms
- **Output Formats**: CSV for compatibility + JSON for detailed reasoning

The system currently uses:
- Model: `gemini-2.5-pro`
- Concurrent limit: 2 requests
- Retry attempts: 3
- Base delay: 1.0 seconds

### 2. Perplexity API Implementation Reference

The `streamlit_simple_perplexity_qa_fixed.py` demonstrates Perplexity API usage with:

- **Model Selection**: sonar-pro, sonar-reasoning, sonar-reasoning-pro (using sonar-reasoning as default)
- **Streaming Support**: Real-time response streaming
- **Reasoning Effort Control**: Low/Medium/High reasoning intensity (using medium as default)
- **Citation Extraction**: Automatic source citation handling
- **Error Handling**: Comprehensive error management
- **Response Processing**: HTML tag cleaning and content formatting

## Migration Strategy

### [X] Phase 0: File Renaming Impact Assessment

#### 0.1 File Renaming: `run_gemini_gj.py` → `run_gj.py`

**Impact Analysis:**

The renaming of `run_gemini_gj.py` to `run_gj.py` affects multiple components in the GraphJudge system:

**Direct References to Update:**
1. **CLI Stage Manager** (`stage_manager.py`):
   - Line 11: Documentation comment
   - Line 514: Class documentation
   - Line 522: Method documentation  
   - Line 527: Script path reference
   - Line 848: Script mapping dictionary

2. **CLI Module** (`__init__.py`):
   - Line 21: Import list

3. **Integration Demo** (`integration_demo.py`):
   - Line 38: Import statement
   - Line 106: Module availability check
   - Line 112-113: Mock patching
   - Line 233-250: Dynamic attribute access
   - Line 307: Usage instructions
   - Line 341: Documentation reference

4. **Test Files**:
   - `test_run_gemini_gj.py`: Complete test suite (1000+ lines)
   - `test_run_kimi_gj.py`: Migration references

5. **Documentation Files**:
   - `README.md`: Usage examples
   - `GRAPH_JUDGE_COMPATIBILITY_EXAMPLE.md`: References

**Required Changes:**

```python
# In stage_manager.py
# Line 11: Update documentation
# 3. Graph Judge - run_gj.py

# Line 514: Update class docstring
# Wraps run_gj.py to provide graph judgment with explainable reasoning

# Line 522: Update method docstring
# """Execute Graph Judge stage using run_gj.py."""

# Line 527: Update script path
script_path = self._get_script_path("run_gj.py")

# Line 848: Update script mapping
'graph_judge': 'run_gj.py',

# In __init__.py
# Line 21: Update import list
run_gj.py, convert_Judge_To_jsonGraph.py)

# In integration_demo.py
# Line 38: Update import
import run_gj

# Line 106: Update module check
if GEMINI_GJ_AVAILABLE and hasattr(run_gj, 'GeminiGraphJudge'):

# Line 112-113: Update mock patching
with unittest.mock.patch('run_gj.DreamOfRedChamberQA', return_value=mock_qa_system):
    judge = run_gj.GeminiGraphJudge()

# Line 233-250: Update dynamic attribute access
original_input_file = getattr(run_gj, 'input_file', None)
setattr(run_gj, 'input_file', input_file)
is_valid = run_gj.validate_input_file()
# ... continue for all attribute references

# Line 307: Update usage instructions
print(f"   3. Run run_gj.py for full evaluation")

# Line 341: Update documentation reference
print("📚 For more details, check the implementation in run_gj.py")
```

**Test File Updates:**

```python
# In test_run_gemini_gj.py (rename to test_run_gj.py)
# Line 4: Update module description
# This test suite validates the functionality of the run_gj.py module,

# Line 17: Update test command
# Run with: pytest test_run_gj.py -v

# Line 66-96: Update all import statements
import run_gj
print("✓ Successfully imported run_gj module")
# ... update all run_gemini_gj references to run_gj

# Line 1395, 1422, 1432: Update sys.argv patching
with patch('sys.argv', ['run_gj.py'] + test_args):

# Line 1531: Update test description
# """Test compatibility with run_gj.py environment variable usage."""
```

**Documentation Updates:**

```markdown
# In README.md
# Update all usage examples
python run_gj.py
python run_gj.py --bootstrap \
    --triples-file ../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration1/test_generated_graphs.txt \
    --source-file ../datasets/KIMI_result_DreamOf_RedChamber/Iteration1/test_denoised.target \
    --output ../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration1/gold_bootstrap.csv

# In GRAPH_JUDGE_COMPATIBILITY_EXAMPLE.md
# Update all file references
```

#### 0.2 Migration Steps for File Renaming

**Step 1: Backup and Rename**
```bash
# Backup original file
cp run_gemini_gj.py run_gemini_gj_backup.py

# Rename to new filename
mv run_gemini_gj.py run_gj.py
```

**Step 2: Update All References**
```bash
# Update all Python files
find . -name "*.py" -exec sed -i 's/run_gemini_gj/run_gj/g' {} \;

# Update documentation files
find . -name "*.md" -exec sed -i 's/run_gemini_gj/run_gj/g' {} \;

# Update test file names
mv test_run_gemini_gj.py test_run_gj.py
```

**Step 3: Validate Changes**
```bash
# Check for any remaining references
grep -r "run_gemini_gj" . --exclude-dir=__pycache__ --exclude-dir=.git

# Run tests to ensure functionality
python -m pytest test_run_gj.py -v
```

**Step 4: Update Logging Configuration**
```python
# In run_gj.py, update log filename generation
# Line 206: Update log filename pattern
log_filename = f"run_gj_log_{timestamp}.txt"
```

### [X] Phase 1: Core API Integration

#### 1.1 Replace Gemini Dependencies

**Current Dependencies:**
```python
# Gemini-specific imports
from dream_of_red_chamber_qa import DreamOfRedChamberQA, QAResponse
```

**New Dependencies:**
```python
# Perplexity API imports
from litellm import completion, acompletion
import os
from dotenv import load_dotenv
```

#### 1.2 Environment Configuration

**Current (.env):**
```
GOOGLE_API_KEY=your_gemini_api_key
```

**New (.env):**
```
PERPLEXITYAI_API_KEY=your_perplexity_api_key
```

#### 1.3 Model Configuration Updates

**Current Gemini Config:**
```python
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_CONCURRENT_LIMIT = 2
GEMINI_RETRY_ATTEMPTS = 3
GEMINI_BASE_DELAY = 1.0
```

**New Perplexity Config:**
```python
PERPLEXITY_MODEL = "perplexity/sonar-reasoning"
PERPLEXITY_CONCURRENT_LIMIT = 3  # Perplexity allows higher concurrency
PERPLEXITY_RETRY_ATTEMPTS = 3
PERPLEXITY_BASE_DELAY = 0.5  # Faster response times
PERPLEXITY_REASONING_EFFORT = "medium"  # For graph judgment accuracy
```

### [X] Phase 2: Core Class Refactoring

#### 2.1 Create PerplexityGraphJudge Class

Replace `GeminiGraphJudge` with `PerplexityGraphJudge`:

```python
class PerplexityGraphJudge:
    """
    Graph Judge adapter that integrates Perplexity API for graph triple validation.
    Replaces Gemini RAG system with Perplexity's sonar-reasoning capabilities.
    """
    
    def __init__(self, model_name: str = PERPLEXITY_MODEL, 
                 reasoning_effort: str = "medium", 
                 enable_console_logging: bool = False):
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.enable_logging = enable_console_logging
        self.temperature = 0.2
        self.max_tokens = 2000
        
        # Validate API key
        if not os.getenv('PERPLEXITYAI_API_KEY'):
            raise ValueError("PERPLEXITYAI_API_KEY not found in environment")
```

#### 2.2 Update Prompt Engineering

**Current Gemini Prompt:**
```python
def _create_graph_judgment_prompt(self, instruction: str) -> str:
    prompt = f"""
你是一個知識圖譜驗證專家。請判斷以下三元組陳述是否為事實正確。
...
"""
```

**New Perplexity Prompt:**
```python
def _create_graph_judgment_prompt(self, instruction: str) -> str:
    # Extract triple from instruction
    triple_part = instruction.replace("Is this true: ", "").replace(" ?", "")
    
    prompt = f"""
You are a knowledge graph validation expert. Please evaluate whether the following triple statement is factually correct.

Task Requirements:
1. Output only "Yes" or "No" (no additional text)
2. Base your judgment on reliable information sources
3. For "Dream of the Red Chamber" related content, pay special attention to literary accuracy
4. For real-world facts, refer to the latest authoritative information

Evaluation Rules:
- If the triple is syntactically correct and factually accurate, answer "Yes"
- If the triple is syntactically incorrect or factually wrong, answer "No"

Triple to evaluate:
{triple_part}

Please answer only "Yes" or "No":
"""
    return prompt
```

#### 2.3 Implement Perplexity API Calls

**Replace Gemini API calls with Perplexity:**

```python
async def judge_graph_triple(self, instruction: str, input_text: str = None) -> str:
    """Judge a graph triple using Perplexity API"""
    
    max_retries = PERPLEXITY_RETRY_ATTEMPTS
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            prompt = self._create_graph_judgment_prompt(instruction)
            
            # Perplexity API call
            response = await acompletion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort
            )
            
            # Extract response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                answer = response.choices[0].message.content.strip()
                judgment = self._parse_response(answer)
                return judgment
            else:
                raise Exception("Invalid response format")
                
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return "No"  # Conservative default
            await asyncio.sleep(PERPLEXITY_BASE_DELAY * (2 ** retry_count))
```

### [Optional-CanSkip]Phase 3: Enhanced Features Integration

#### 3.1 Streaming Support (Optional)

Add streaming capability for real-time processing feedback:

```python
async def judge_graph_triple_streaming(self, instruction: str, 
                                     stream_container=None) -> str:
    """Streaming version of graph triple judgment"""
    
    prompt = self._create_graph_judgment_prompt(instruction)
    
    response = await acompletion(
        model=self.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        reasoning_effort=self.reasoning_effort,
        stream=True
    )
    
    full_answer = ""
    async for chunk in response:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                full_answer += delta.content
                
                # Update stream container if provided
                if stream_container:
                    with stream_container:
                        st.markdown(f"Processing: {full_answer}")
    
    return self._parse_response(full_answer)
```

#### 3.2 Citation Integration

Leverage Perplexity's citation capabilities:

```python
def _extract_citations(self, response) -> List[Dict]:
    """Extract citations from Perplexity response"""
    citations = []
    
    if hasattr(response, 'citations') and response.citations:
        for i, citation in enumerate(response.citations):
            citations.append({
                "number": str(i + 1),
                "title": self._extract_title_from_url(citation),
                "url": citation,
                "type": "perplexity_citation"
            })
    
    return citations
```

### [Optional-CanSkip] Phase 4: Explainable Mode Enhancement

#### 4.1 Enhanced Reasoning Prompts

Update explainable judgment prompts for Perplexity:

```python
def _create_explainable_judgment_prompt(self, instruction: str) -> str:
    triple_part = instruction.replace("Is this true: ", "").replace(" ?", "")
    
    prompt = f"""
You are a professional knowledge graph validation expert. Please provide a detailed analysis of the following triple statement with structured judgment results.

Triple to evaluate: {triple_part}

Please provide complete analysis in the following format:

1. Judgment Result: [Only "Yes" or "No"]

2. Confidence Level: [Numerical value between 0.0 and 1.0 indicating your certainty]

3. Detailed Reasoning: [Explain your judgment rationale including:
   - Triple syntax correctness analysis
   - Factual accuracy assessment
   - Relevant background knowledge or evidence
   - For "Dream of the Red Chamber" content, refer to original text]

4. Evidence Sources: [List evidence types supporting the judgment: source_text, domain_knowledge, literary_history, etc.]

5. Error Type: [If judgment is "No", specify error type: entity_mismatch, factual_error, structural_error, temporal_inconsistency, source_unsupported. If "Yes", answer "None"]

6. Alternative Suggestions: [If judgment is "No", provide correct triple suggestions in format: subject-relation-object]

Please ensure your response is structured, logical, and based on reliable sources.
"""
    return prompt
```

#### 4.2 Structured Response Parsing

Enhance response parsing for Perplexity's output format:

```python
def _parse_explainable_response(self, response) -> ExplainableJudgment:
    """Parse Perplexity response for explainable graph judgment"""
    
    if not response or not hasattr(response, 'choices'):
        return self._create_error_judgment("Invalid response format")
    
    answer = response.choices[0].message.content.strip()
    
    try:
        # Enhanced parsing for Perplexity's structured output
        judgment = self._extract_judgment(answer)
        confidence = self._extract_confidence(answer)
        reasoning = self._extract_reasoning(answer)
        evidence_sources = self._extract_evidence_sources(answer)
        error_type = self._extract_error_type(answer)
        alternative_suggestions = self._extract_alternatives(answer)
        
        return ExplainableJudgment(
            judgment=judgment,
            confidence=confidence,
            reasoning=reasoning,
            evidence_sources=evidence_sources,
            alternative_suggestions=alternative_suggestions,
            error_type=error_type,
            processing_time=0.0
        )
        
    except Exception as e:
        return self._create_error_judgment(f"Parsing error: {str(e)}")
```

### [Optional-CanSkip] Phase 5: Configuration and Testing

#### 5.1 Update Configuration Constants

```python
# Perplexity API Configuration
PERPLEXITY_MODEL = "perplexity/sonar-reasoning"
PERPLEXITY_CONCURRENT_LIMIT = 3
PERPLEXITY_RETRY_ATTEMPTS = 3
PERPLEXITY_BASE_DELAY = 0.5
PERPLEXITY_REASONING_EFFORT = "medium"

# Model selection options (sonar-reasoning is the default)
PERPLEXITY_MODELS = {
    "sonar-pro": "perplexity/sonar-pro",
    "sonar-reasoning": "perplexity/sonar-reasoning",
    "sonar-reasoning-pro": "perplexity/sonar-reasoning-pro"
}
```

#### 5.2 Environment Setup

Create environment validation:

```python
def validate_perplexity_environment():
    """Validate Perplexity API environment setup"""
    api_key = os.getenv('PERPLEXITYAI_API_KEY')
    if not api_key:
        raise ValueError("PERPLEXITYAI_API_KEY not found in environment variables")
    
    # Test API connection
try:
    test_response = asyncio.run(acompletion(
        model="perplexity/sonar-reasoning",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=10
    ))
        print("✓ Perplexity API connection successful")
        return True
    except Exception as e:
        print(f"✗ Perplexity API connection failed: {e}")
        return False
```

#### 5.3 Testing Strategy

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test full pipeline
3. **Performance Tests**: Compare response times
4. **Accuracy Tests**: Validate judgment quality
5. **Compatibility Tests**: Ensure CSV output format

### [Optional-CanSkip]Phase 6: Migration Steps

#### Step 1: Backup and Rename Current Implementation
```bash
# Backup original file
cp run_gemini_gj.py run_gemini_gj_backup.py

# Rename to new filename
mv run_gemini_gj.py run_gj.py

# Update all references in the codebase
find . -name "*.py" -exec sed -i 's/run_gemini_gj/run_gj/g' {} \;
find . -name "*.md" -exec sed -i 's/run_gemini_gj/run_gj/g' {} \;

# Rename test files
mv test_run_gemini_gj.py test_run_gj.py

# Validate no remaining references
grep -r "run_gemini_gj" . --exclude-dir=__pycache__ --exclude-dir=.git
```

#### Step 2: Install Perplexity Dependencies
```bash
pip install litellm python-dotenv
```

#### Step 3: Update Environment Variables
```bash
# Remove Gemini API key
# Add Perplexity API key
export PERPLEXITYAI_API_KEY="your_api_key_here"
```

#### Step 4: Update File References
1. Update all import statements in CLI modules
2. Update script path references in stage manager
3. Update test file imports and references
4. Update documentation examples
5. Update logging configuration

#### Step 5: Implement Core Changes
1. Replace Gemini imports with Perplexity imports
2. Update model configuration
3. Refactor API calls
4. Update prompt engineering
5. Enhance error handling

#### Step 6: Testing and Validation
1. Run unit tests with new file names
2. Validate output formats
3. Compare performance metrics
4. Test error scenarios
5. Verify CLI pipeline integration

#### Step 7: Documentation Update
1. Update README.md with new file names
2. Update configuration documentation
3. Update API usage examples
4. Update migration guides

## Expected Benefits

### 1. Performance Improvements
- **Faster Response Times**: Perplexity typically responds faster than Gemini
- **Higher Concurrency**: Support for more concurrent requests
- **Better Streaming**: Real-time response streaming capabilities

### 2. Enhanced Capabilities
- **Advanced Reasoning**: Sonar-reasoning models provide better logical analysis
- **Citation Support**: Automatic source citation extraction
- **Flexible Reasoning Effort**: Adjustable reasoning intensity levels

### 3. Cost Optimization
- **Competitive Pricing**: Perplexity may offer better pricing for high-volume usage
- **Efficient Token Usage**: Optimized prompts for better token efficiency

## Risk Mitigation

### 1. File Renaming Risks
- **Import Path Updates**: Ensure all import statements are updated correctly
- **CLI Integration**: Verify stage manager and pipeline integration
- **Test Suite Updates**: Update all test files and references
- **Documentation Consistency**: Update all documentation references

### 2. Compatibility Risks
- **Maintain CSV Output Format**: Ensure evaluation pipeline compatibility
- **Preserve Async Processing**: Keep existing concurrency patterns
- **Backward Compatibility**: Maintain existing function signatures

### 3. Performance Risks
- **Rate Limiting**: Implement proper rate limiting for Perplexity API
- **Error Handling**: Robust error handling for API failures
- **Fallback Mechanisms**: Graceful degradation on API issues

### 4. Quality Risks
- **Prompt Optimization**: Thorough testing of new prompts
- **Response Validation**: Validate judgment quality against known test cases
- **A/B Testing**: Compare results with Gemini implementation

## Success Metrics

### 1. Technical Metrics
- Response time improvement > 20%
- Success rate maintenance > 95%
- Error rate reduction > 10%

### 2. Quality Metrics
- Judgment accuracy maintenance or improvement
- Reasoning quality enhancement
- Citation relevance improvement

### 3. Operational Metrics
- API cost reduction
- Processing throughput increase
- User satisfaction maintenance

## Timeline

### Week 1: File Renaming and Core Migration
- File renaming and reference updates
- Environment setup and dependency installation
- Basic API integration
- Core class refactoring

### Week 2: Feature Enhancement
- Explainable mode implementation
- Citation integration
- Streaming support
- CLI integration updates

### Week 3: Testing and Validation
- Comprehensive testing with new file names
- Performance optimization
- Documentation updates
- Pipeline integration validation

### Week 4: Deployment and Monitoring
- Production deployment
- Performance monitoring
- User feedback collection
- Migration guide updates

## Conclusion

This migration plan provides a comprehensive roadmap for transitioning from Gemini to Perplexity API while maintaining system functionality and improving performance. The phased approach ensures minimal disruption while maximizing the benefits of Perplexity's advanced reasoning capabilities.

The key success factors include:
1. Maintaining backward compatibility
2. Thorough testing and validation
3. Performance monitoring and optimization
4. User experience preservation
5. Cost-effective implementation

By following this plan, the GraphJudge system will benefit from Perplexity's advanced AI capabilities while maintaining its robust evaluation pipeline and user-friendly interface.

## Phase 7: Code Modularization Plan

### 7.1 Overview of Modularization Strategy

After completing the Perplexity API migration, the final step is to modularize the `run_gj.py` file (1951 lines) into smaller, maintainable modules. Each module will be limited to 350 lines maximum and stored in the `Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/` directory.

### 7.2 Current Code Structure Analysis

The `run_gj.py` file contains the following major components:

1. **Imports and Dependencies** (Lines 1-100)
2. **Data Structures and Configuration** (Lines 101-200)
3. **Logging System** (Lines 201-300)
4. **Core GraphJudge Class** (Lines 301-900)
5. **Gold Label Bootstrapping** (Lines 901-1300)
6. **Processing Pipeline** (Lines 1301-1600)
7. **Utility Functions** (Lines 1601-1800)
8. **Main Execution** (Lines 1801-1951)

### 7.3 Proposed Module Structure

#### 7.3.1 Core Modules (graphJudge_Phase/)

```
graphJudge_Phase/
├── __init__.py                    # Module initialization and exports
├── config.py                      # Configuration constants and settings
├── data_structures.py             # NamedTuple definitions and data models
├── logging_system.py              # TerminalLogger and logging utilities
├── graph_judge_core.py            # Main PerplexityGraphJudge class
├── prompt_engineering.py          # Prompt creation and response parsing
├── gold_label_bootstrapping.py    # Bootstrap functionality
├── processing_pipeline.py         # Main processing orchestration
├── utilities.py                   # Helper functions and validations
└── main.py                        # Entry point and CLI handling
```

#### 7.3.2 Detailed Module Breakdown

**1. config.py (≤200 lines)**
```python
# Configuration constants
PERPLEXITY_MODEL = "perplexity/sonar-reasoning"
PERPLEXITY_CONCURRENT_LIMIT = 3
PERPLEXITY_RETRY_ATTEMPTS = 3
PERPLEXITY_BASE_DELAY = 0.5
PERPLEXITY_REASONING_EFFORT = "medium"

# Gold Label Bootstrapping configuration
GOLD_BOOTSTRAP_CONFIG = {
    'fuzzy_threshold': 0.8,
    'sample_rate': 0.15,
    'llm_batch_size': 10,
    'max_source_lines': 1000,
    'random_seed': 42
}

# Logging configuration
LOG_DIR = "logs/iteration2"

# Dataset configuration
folder = "KIMI_result_DreamOf_RedChamber"
iteration = int(os.environ.get('PIPELINE_ITERATION', '2'))
input_file = os.environ.get('PIPELINE_INPUT_FILE', f"../datasets/{folder}/Graph_Iteration{iteration}/test_instructions_context_kimi_v2.json")
output_file = os.environ.get('PIPELINE_OUTPUT_FILE', f"../datasets/{folder}/Graph_Iteration{iteration}/pred_instructions_context_perplexity_itr{iteration}.csv")
```

**2. data_structures.py (≤150 lines)**
```python
from typing import NamedTuple, List, Optional, Dict

class TripleData(NamedTuple):
    """Data structure for knowledge graph triples"""
    subject: str
    predicate: str
    object: str
    source_line: str = ""
    line_number: int = 0

class BootstrapResult(NamedTuple):
    """Result of gold label bootstrapping for a single triple"""
    triple: TripleData
    source_idx: int
    fuzzy_score: float
    auto_expected: Optional[bool]
    llm_evaluation: Optional[str]
    expected: Optional[bool]
    note: str

class ExplainableJudgment(NamedTuple):
    """Data structure for explainable graph judgment results"""
    judgment: str
    confidence: float
    reasoning: str
    evidence_sources: List[str]
    alternative_suggestions: List[Dict]
    error_type: Optional[str]
    processing_time: float
```

**3. logging_system.py (≤200 lines)**
```python
import os
import logging
from datetime import datetime
from .config import LOG_DIR

def setup_terminal_logging():
    """Set up terminal logging to capture output to a timestamped log file"""
    # Implementation from original file

class TerminalLogger:
    """Simple terminal logger that captures output to file"""
    # Implementation from original file
```

**4. graph_judge_core.py (≤350 lines)**
```python
import asyncio
import os
from typing import Optional
from litellm import acompletion
from .config import PERPLEXITY_MODEL, PERPLEXITY_RETRY_ATTEMPTS, PERPLEXITY_BASE_DELAY
from .data_structures import ExplainableJudgment
from .prompt_engineering import PromptEngineer

class PerplexityGraphJudge:
    """Graph Judge adapter that integrates Perplexity API for graph triple validation"""
    
    def __init__(self, model_name: str = PERPLEXITY_MODEL, 
                 reasoning_effort: str = "medium", 
                 enable_console_logging: bool = False):
        # Implementation from original file
    
    async def judge_graph_triple(self, instruction: str, input_text: str = None) -> str:
        # Implementation from original file
    
    async def judge_graph_triple_with_explanation(self, instruction: str, input_text: str = None) -> ExplainableJudgment:
        # Implementation from original file
```

**5. prompt_engineering.py (≤300 lines)**
```python
import re
from typing import List, Optional, Dict
from .data_structures import ExplainableJudgment

class PromptEngineer:
    """Handles prompt creation and response parsing for Perplexity API"""
    
    @staticmethod
    def create_graph_judgment_prompt(instruction: str) -> str:
        # Implementation from original file
    
    @staticmethod
    def create_explainable_judgment_prompt(instruction: str) -> str:
        # Implementation from original file
    
    @staticmethod
    def parse_response(response) -> str:
        # Implementation from original file
    
    @staticmethod
    def parse_explainable_response(response) -> ExplainableJudgment:
        # Implementation from original file
    
    # All parsing helper methods (_extract_judgment, _extract_confidence, etc.)
```

**6. gold_label_bootstrapping.py (≤350 lines)**
```python
import asyncio
import json
import csv
import random
from typing import List, Optional
from .data_structures import TripleData, BootstrapResult
from .config import GOLD_BOOTSTRAP_CONFIG
from .graph_judge_core import PerplexityGraphJudge

class GoldLabelBootstrapper:
    """Handles the two-stage gold label bootstrapping process"""
    
    def __init__(self, graph_judge: PerplexityGraphJudge):
        self.graph_judge = graph_judge
    
    def load_triples_from_file(self, triples_file: str) -> List[TripleData]:
        # Implementation from original file
    
    def load_source_text(self, source_file: str) -> List[str]:
        # Implementation from original file
    
    def stage1_rapidfuzz_matching(self, triples: List[TripleData], source_lines: List[str]) -> List[BootstrapResult]:
        # Implementation from original file
    
    async def stage2_llm_semantic_evaluation(self, uncertain_results: List[BootstrapResult], source_lines: List[str]) -> List[BootstrapResult]:
        # Implementation from original file
    
    def sample_uncertain_cases(self, results: List[BootstrapResult]) -> List[BootstrapResult]:
        # Implementation from original file
    
    def save_bootstrap_results(self, results: List[BootstrapResult], output_file: str) -> bool:
        # Implementation from original file
    
    async def bootstrap_gold_labels(self, triples_file: str, source_file: str, output_file: str) -> bool:
        # Implementation from original file
```

**7. processing_pipeline.py (≤350 lines)**
```python
import asyncio
import csv
import json
from typing import List, Optional
from pathlib import Path
from .config import PERPLEXITY_CONCURRENT_LIMIT, PERPLEXITY_BASE_DELAY, output_file
from .graph_judge_core import PerplexityGraphJudge
from .data_structures import ExplainableJudgment

class ProcessingPipeline:
    """Orchestrates the graph judgment evaluation process"""
    
    def __init__(self, graph_judge: PerplexityGraphJudge):
        self.graph_judge = graph_judge
    
    async def process_instructions(self, data_eval, explainable_mode: bool = False, reasoning_file_path: Optional[str] = None):
        # Implementation from original file
    
    def generate_reasoning_file_path(self, csv_output_path: str, custom_path: Optional[str] = None) -> str:
        # Implementation from original file
    
    def save_reasoning_file(self, reasoning_results: List[Dict], output_path: str) -> bool:
        # Implementation from original file
```

**8. utilities.py (≤250 lines)**
```python
import os
import json
from typing import List

def validate_input_file(input_file: str) -> bool:
    """Validate that the input file exists and has the correct format"""
    # Implementation from original file

def create_output_directory(output_file: str):
    """Ensure the output directory exists before writing results"""
    # Implementation from original file

def validate_perplexity_environment() -> bool:
    """Validate Perplexity API environment setup"""
    # Implementation from original file

def get_gemini_completion(instruction, input_text=None):
    """Get completion from Perplexity API for graph judgment"""
    # Implementation from original file
```

**9. main.py (≤300 lines)**
```python
import asyncio
import argparse
import sys
from .config import *
from .logging_system import setup_terminal_logging, TerminalLogger
from .graph_judge_core import PerplexityGraphJudge
from .gold_label_bootstrapping import GoldLabelBootstrapper
from .processing_pipeline import ProcessingPipeline
from .utilities import validate_input_file, create_output_directory

def parse_arguments():
    """Parse command line arguments for different operation modes"""
    # Implementation from original file

async def run_gold_label_bootstrapping(args):
    """Run the gold label bootstrapping pipeline"""
    # Implementation from original file

async def run_graph_judgment(explainable_mode: bool = False, reasoning_file_path: Optional[str] = None):
    """Run the graph judgment pipeline (standard or explainable mode)"""
    # Implementation from original file

if __name__ == "__main__":
    """Main execution block with comprehensive error handling and validation"""
    # Implementation from original file
```

**10. __init__.py (≤50 lines)**
```python
"""
GraphJudge Phase - Modularized Perplexity API Graph Judge System
"""

from .config import *
from .data_structures import TripleData, BootstrapResult, ExplainableJudgment
from .graph_judge_core import PerplexityGraphJudge
from .gold_label_bootstrapping import GoldLabelBootstrapper
from .processing_pipeline import ProcessingPipeline
from .utilities import validate_input_file, create_output_directory
from .logging_system import setup_terminal_logging, TerminalLogger

__version__ = "2.0.0"
__author__ = "GraphJudge Team"

# Main exports
__all__ = [
    'PerplexityGraphJudge',
    'GoldLabelBootstrapper', 
    'ProcessingPipeline',
    'TripleData',
    'BootstrapResult',
    'ExplainableJudgment',
    'validate_input_file',
    'create_output_directory',
    'setup_terminal_logging',
    'TerminalLogger'
]
```

### 7.4 Migration Steps for Modularization

#### Step 1: Create Module Directory Structure
```bash
# Create the module directory
mkdir -p Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase

# Create all module files
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/__init__.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/config.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/data_structures.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/logging_system.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/graph_judge_core.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/prompt_engineering.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/gold_label_bootstrapping.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/processing_pipeline.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/utilities.py
touch Miscellaneous/KgGen/GraphJudge/chat/graphJudge_Phase/main.py
```

#### Step 2: Extract and Refactor Code
1. **Extract configuration constants** to `config.py`
2. **Extract data structures** to `data_structures.py`
3. **Extract logging system** to `logging_system.py`
4. **Extract core GraphJudge class** to `graph_judge_core.py`
5. **Extract prompt engineering** to `prompt_engineering.py`
6. **Extract bootstrapping functionality** to `gold_label_bootstrapping.py`
7. **Extract processing pipeline** to `processing_pipeline.py`
8. **Extract utility functions** to `utilities.py`
9. **Extract main execution** to `main.py`
10. **Create module initialization** in `__init__.py`

#### Step 3: Update Import Statements
```python
# Update all internal imports to use relative imports
from .config import PERPLEXITY_MODEL, PERPLEXITY_CONCURRENT_LIMIT
from .data_structures import TripleData, ExplainableJudgment
from .graph_judge_core import PerplexityGraphJudge
# etc.
```

#### Step 4: Update External References
```python
# Update CLI stage manager to use new module
from graphJudge_Phase import PerplexityGraphJudge, GoldLabelBootstrapper

# Update test files to import from new module structure
from graphJudge_Phase.graph_judge_core import PerplexityGraphJudge
from graphJudge_Phase.gold_label_bootstrapping import GoldLabelBootstrapper
```

#### Step 5: Create Backward Compatibility
```python
# In the original run_gj.py location, create a compatibility wrapper
"""
Backward compatibility wrapper for run_gj.py
This file maintains compatibility with existing CLI and test systems
"""

from graphJudge_Phase import *

# Re-export all main components
__all__ = [
    'PerplexityGraphJudge',
    'GoldLabelBootstrapper',
    'ProcessingPipeline',
    'TripleData',
    'BootstrapResult', 
    'ExplainableJudgment'
]

# Maintain the same global instance pattern
try:
    gemini_judge = PerplexityGraphJudge(enable_console_logging=False)
    print("✓ Global Perplexity Graph Judge instance initialized")
except Exception as e:
    print(f"✗ Failed to initialize global Perplexity Graph Judge: {e}")
    gemini_judge = None
```

### 7.5 Benefits of Modularization

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual modules can be tested in isolation
3. **Reusability**: Modules can be imported and used independently
4. **Readability**: Smaller files are easier to understand and navigate
5. **Collaboration**: Multiple developers can work on different modules
6. **Documentation**: Each module can have focused documentation
7. **Performance**: Selective imports reduce memory usage

### 7.6 Testing Strategy for Modularization

1. **Unit Tests**: Test each module independently
2. **Integration Tests**: Test module interactions
3. **Compatibility Tests**: Ensure backward compatibility
4. **Performance Tests**: Verify no performance degradation
5. **CLI Tests**: Ensure CLI functionality remains intact

### 7.7 Timeline for Modularization

**Week 1: Module Creation**
- Create directory structure
- Extract configuration and data structures
- Set up module initialization

**Week 2: Core Module Extraction**
- Extract logging system
- Extract core GraphJudge class
- Extract prompt engineering

**Week 3: Feature Module Extraction**
- Extract gold label bootstrapping
- Extract processing pipeline
- Extract utilities

**Week 4: Integration and Testing**
- Create main entry point
- Update all import statements
- Create backward compatibility wrapper
- Comprehensive testing

### 7.8 Risk Mitigation for Modularization

1. **Import Errors**: Thorough testing of all import paths
2. **Functionality Loss**: Maintain comprehensive test coverage
3. **Performance Impact**: Monitor for any performance degradation
4. **CLI Compatibility**: Ensure CLI integration remains functional
5. **Documentation Updates**: Update all documentation references

This modularization plan ensures that the GraphJudge system becomes more maintainable and scalable while preserving all existing functionality and compatibility.
