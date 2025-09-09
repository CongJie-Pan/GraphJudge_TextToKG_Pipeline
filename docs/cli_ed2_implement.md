# CLI ED2 Implementation Plan: Unified Knowledge Graph Pipeline Integration

## Overview

This document outlines the step-by-step implementation plan for integrating the existing `run_entity.py`, `run_triple.py`, and `graphJudge_Phase` modular system into the unified CLI system (`Miscellaneous\KgGen\GraphJudge\chat\cli`). The goal is to create a seamless, modular pipeline that combines all three components while maintaining backward compatibility and improving maintainability.
Attention: please don't use the moudule code of `Miscellaneous\KgGen\GraphJudge\chat\extractEntity_Phase` because the module haven't developed yet.

## Current Architecture Analysis

### Existing Components

1. **run_entity.py** (862 lines)
   - GPT-5-mini based Entity Extraction & Text Denoising (ECTD)
   - Features: Caching, rate limiting, terminal logging, validation
   - Output: `test_entity.txt`, `test_denoised.target`

2. **run_triple.py** (1070 lines)
   - Enhanced semantic graph generation using GPT-5-mini
   - Features: JSON schema validation, text chunking, post-processing
   - Output: `test_instructions_context_gpt5mini_v2.json`

3. **graphJudge_Phase** (Modular System)
   - Modularized Perplexity API-based graph judgment system with comprehensive architecture
   - Core Components: `PerplexityGraphJudge`, `GoldLabelBootstrapper`, `ProcessingPipeline`, `PromptEngineer`
   - Features: Explainable reasoning, gold label bootstrapping, modular pipeline processing, citation management
   - Output: CSV files with judgment results and optional reasoning JSON files
   - Advanced capabilities: Multiple operation modes, RapidFuzz integration, comprehensive error handling
   - Modular architecture: 9 core modules with comprehensive testing and clean API design

4. **Existing CLI** (`cli/` directory)
   - Unified pipeline architecture with comprehensive stage management system
   - Features: Iteration management, progress monitoring, configuration management, pipeline state tracking
   - Current modules: `stage_manager.py`, `iteration_manager.py`, `config_manager.py`, `cli.py`, `pipeline_monitor.py`
   - Current stages: ECTD, Triple Generation, Graph Judge, Evaluation
   - Advanced features: Pipeline state management with error tracking, progress monitoring, and recovery capabilities

## Implementation Strategy

### Phase 1: Module Integration and Refactoring

#### Step 1.1: Create Enhanced Stage Managers

**Objective**: Integrate the three modules into the existing CLI stage management system.

**Actions**:
1. **Update `stage_manager.py`**:
   - Replace the current `GraphJudgeStage` to use the modular `graphJudge_Phase` system
   - Enhance `ECTDStage` to support both GPT-5-mini and Kimi models
   - Update `TripleGenerationStage` to use the enhanced `run_triple.py` features
   - Add new stage: `GraphJudgePhaseStage` for the modular graph judgment system
   - Integrate with existing `PipelineState` management system for robust error handling and progress tracking

2. **Create new stage classes**:
   ```python
   class GraphJudgePhaseStage(PipelineStage):
       """Modular Graph Judge Stage using graphJudge_Phase system"""
       
   class EnhancedECTDStage(PipelineStage):
       """Enhanced ECTD Stage with GPT-5-mini support and advanced caching"""
       
   class EnhancedTripleGenerationStage(PipelineStage):
       """Enhanced Triple Generation with schema validation, chunking, and post-processing"""
   ```

#### Step 1.2: Configuration Integration

**Objective**: Integrate configuration management for all three modules.

**Actions**:
1. **Update `config_manager.py`**:
   - Add GPT-5-mini configuration options
   - Integrate Perplexity API configuration
   - Add schema validation settings
   - Include text chunking parameters

2. **Create unified configuration structure**:
   ```python
   class UnifiedPipelineConfig:
       ectd_config: Dict[str, Any]  # GPT-5-mini + Kimi options with caching
       triple_generation_config: Dict[str, Any]  # Enhanced features with schema validation
       graph_judge_phase_config: Dict[str, Any]  # Modular graphJudge_Phase system
       pipeline_state_config: Dict[str, Any]  # Pipeline state management integration
       evaluation_config: Dict[str, Any]
   ```

#### Step 1.3: Environment Variable Standardization

**Objective**: Standardize environment variables across all modules.

**Actions**:
1. **Create environment variable mapping**:
   ```python
   ENVIRONMENT_MAPPING = {
       'PIPELINE_ITERATION': 'Current iteration number',
       'PIPELINE_DATASET_PATH': 'Base dataset path',
       'PIPELINE_OUTPUT_DIR': 'Stage-specific output directory',
       'PIPELINE_INPUT_FILE': 'Input file for current stage',
       'PIPELINE_OUTPUT_FILE': 'Output file for current stage',
       'PIPELINE_INPUT_ITERATION': 'Input iteration for triple generation',
       'PIPELINE_GRAPH_ITERATION': 'Graph iteration for output',
       'OPENAI_API_KEY': 'OpenAI API key for GPT-5-mini',
       'PERPLEXITYAI_API_KEY': 'Perplexity API key for graph judgment',
       'GPT5_MINI_MODEL': 'GPT-5-mini model configuration',
       'PERPLEXITY_MODEL': 'Perplexity model configuration',
       'SCHEMA_VALIDATION_ENABLED': 'Enable/disable schema validation',
       'TEXT_CHUNKING_ENABLED': 'Enable/disable text chunking',
       'CACHE_ENABLED': 'Enable/disable caching system',
       'EXPLAINABLE_MODE': 'Enable explainable reasoning mode'
   }
   ```

### Phase 2: Enhanced Stage Implementation

#### Step 2.1: Enhanced ECTD Stage Implementation

**Objective**: Integrate `run_entity.py` functionality into the CLI with enhanced features.

**Implementation Details**:

1. **Create `enhanced_ectd_stage.py`**:
   ```python
   class EnhancedECTDStage(PipelineStage):
       def __init__(self, config: Dict[str, Any]):
           super().__init__("Enhanced ECTD", config)
           self.model_type = config.get('model_type', 'gpt5-mini')  # gpt5-mini or kimi
           self.cache_enabled = config.get('cache_enabled', True)
           self.parallel_workers = config.get('parallel_workers', 5)
       
       async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
           # Model-specific execution logic
           if self.model_type == 'gpt5-mini':
               return await self._execute_gpt5mini_ectd(iteration, iteration_path, **kwargs)
           else:
               return await self._execute_kimi_ectd(iteration, iteration_path, **kwargs)
   ```

2. **Key Features to Integrate**:
   - GPT-5-mini API integration with rate limiting
   - Intelligent caching system
   - Terminal progress logging
   - Input validation and error handling
   - Output file validation

3. **Configuration Options**:
   ```yaml
   ectd_config:
     model_type: "gpt5-mini"  # or "kimi"
     temperature: 0.3
     batch_size: 20
     cache_enabled: true
     parallel_workers: 5
     max_retry_attempts: 3
     rate_limit_delay: 1.0
   ```

#### Step 2.2: Enhanced Triple Generation Stage Implementation

**Objective**: Integrate enhanced `run_triple.py` functionality with schema validation and chunking.

**Implementation Details**:

1. **Create `enhanced_triple_stage.py`**:
   ```python
   class EnhancedTripleGenerationStage(PipelineStage):
       def __init__(self, config: Dict[str, Any]):
           super().__init__("Enhanced Triple Generation", config)
           self.schema_validation = config.get('schema_validation_enabled', True)
           self.text_chunking = config.get('text_chunking_enabled', True)
           self.post_processing = config.get('post_processing_enabled', True)
       
       async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
           # Enhanced execution with all features
           return await self._execute_enhanced_triple_generation(iteration, iteration_path, **kwargs)
   ```

2. **Key Features to Integrate**:
   - Structured JSON output with Pydantic validation
   - Text chunking for large inputs
   - Post-processing with triple parser
   - Multiple output formats (JSON, TXT, enhanced)
   - Quality metrics and statistics

3. **Configuration Options**:
   ```yaml
   triple_generation_config:
     output_format: "json"
     schema_validation_enabled: true
     text_chunking_enabled: true
     post_processing_enabled: true
     max_tokens_per_chunk: 1000
     chunk_overlap: 100
     relation_mapping: "./config/relation_map.json"
   ```

#### Step 2.3: GraphJudge Phase Integration

**Objective**: Integrate the modular `graphJudge_Phase` system into the CLI with full modular architecture support.

**Implementation Details**:

1. **Create `graph_judge_phase_stage.py`**:
   ```python
   from graphJudge_Phase import (
       PerplexityGraphJudge, 
       GoldLabelBootstrapper, 
       ProcessingPipeline
   )
   
   class GraphJudgePhaseStage(PipelineStage):
       def __init__(self, config: Dict[str, Any]):
           super().__init__("GraphJudge Phase", config)
           self.explainable_mode = config.get('explainable_mode', False)
           self.bootstrap_mode = config.get('bootstrap_mode', False)
           self.model_name = config.get('model_name', 'perplexity/sonar-reasoning')
           self.reasoning_effort = config.get('reasoning_effort', 'medium')
           self.enable_console_logging = config.get('enable_console_logging', False)
           
           # Initialize modular components
           self.graph_judge = PerplexityGraphJudge(
               model_name=self.model_name,
               reasoning_effort=self.reasoning_effort,
               enable_console_logging=self.enable_console_logging
           )
           self.bootstrapper = GoldLabelBootstrapper(self.graph_judge)
           self.pipeline = ProcessingPipeline(self.graph_judge)
       
       async def execute(self, iteration: int, iteration_path: str, **kwargs) -> bool:
           # Modular execution with clean component separation
           if self.bootstrap_mode:
               return await self._execute_gold_label_bootstrapping(iteration, iteration_path, **kwargs)
           elif self.explainable_mode:
               return await self._execute_explainable_judgment(iteration, iteration_path, **kwargs)
           else:
               return await self._execute_standard_judgment(iteration, iteration_path, **kwargs)
   ```

2. **Key Modular Components to Integrate**:
   - **PerplexityGraphJudge**: Core graph judgment functionality with multiple model support
   - **GoldLabelBootstrapper**: Two-stage automatic label assignment using RapidFuzz + LLM evaluation
   - **ProcessingPipeline**: Configurable pipeline for batch processing and statistics tracking
   - **PromptEngineer**: Specialized prompt creation for different judgment modes
   - **Data Structures**: TripleData, ExplainableJudgment, BootstrapResult with type safety
   - **Logging System**: Modular terminal logging with timestamped progress files
   - **Utilities**: Input validation, file handling, and environment setup
   - **Configuration System**: Centralized configuration management with validation
   - **Comprehensive Testing**: Unit tests for all components with high coverage

3. **Configuration Options**:
   ```yaml
   graph_judge_phase_config:
     # Core GraphJudge configuration
     explainable_mode: false
     bootstrap_mode: false
     model_name: "perplexity/sonar-reasoning"
     reasoning_effort: "medium"
     enable_console_logging: false
     temperature: 0.2
     max_tokens: 2000
     concurrent_limit: 3
     retry_attempts: 3
     base_delay: 0.5
     
     # Gold label bootstrapping configuration
     gold_bootstrap_config:
       fuzzy_threshold: 0.8
       sample_rate: 0.15
       llm_batch_size: 10
       max_source_lines: 1000
       random_seed: 42
     
     # Processing pipeline configuration
     processing_config:
       enable_statistics: true
       save_reasoning_files: true
       batch_processing: true
   ```

### Phase 3: CLI Enhancement and Integration

#### Step 3.1: Update Main CLI Interface

**Objective**: Enhance the main CLI to support the integrated modules.

**Actions**:

1. **Update `cli.py`**:
   ```python
   class EnhancedKGPipeline(KGPipeline):
       def __init__(self, config_path: Optional[str] = None):
           super().__init__(config_path)
           # Enhanced initialization with new stages
           self.stage_manager = EnhancedStageManager(self.config)
       
       async def run_pipeline(self, input_file: str, iteration: Optional[int] = None, 
                            model_type: str = "gpt5-mini"):
           # Enhanced pipeline execution with model selection
           pass
   ```

2. **Add new CLI commands**:
   ```bash
   # Model-specific execution
   python cli.py run-pipeline --input file.txt --model gpt5-mini
   python cli.py run-pipeline --input file.txt --model kimi
   
   # Enhanced stage execution
   python cli.py run-ectd --model gpt5-mini --parallel-workers 5 --cache-enabled
   python cli.py run-triple-generation --schema-validation --chunking --post-processing
   python cli.py run-graph-judge-phase --model sonar-reasoning --reasoning-effort medium
   
   # GraphJudge Phase modes
   python cli.py run-graph-judge-phase --explainable --reasoning-file reasoning.json
   python cli.py run-graph-judge-phase --verbose --enable-console-logging
   
   # Gold label bootstrapping mode
   python cli.py run-graph-judge-phase --bootstrap \
     --triples-file triples.txt \
     --source-file source.txt \
     --output bootstrap.csv \
     --threshold 0.8 --sample-rate 0.15
   
   # Direct modular system integration
   python cli.py run-pipeline --stage graph_judge_phase --explainable --bootstrap
   ```

#### Step 3.2: Enhanced Stage Manager

**Objective**: Create an enhanced stage manager that supports all integrated modules.

**Implementation**:

1. **Create `enhanced_stage_manager.py`**:
   ```python
   class EnhancedStageManager(StageManager):
       def __init__(self, config):
           super().__init__(config)
           # Initialize enhanced stages with modular graphJudge_Phase integration
           self.stages.update({
               'enhanced_ectd': EnhancedECTDStage(config.ectd_config),
               'enhanced_triple_generation': EnhancedTripleGenerationStage(config.triple_generation_config),
               'graph_judge_phase': GraphJudgePhaseStage(config.graph_judge_phase_config)
           })
           
           # Update stage order
           self.stage_order = [
               'enhanced_ectd', 
               'enhanced_triple_generation', 
               'graph_judge_phase', 
               'evaluation'
           ]
           
           # Initialize pipeline state management
           self.pipeline_state_manager = PipelineStateManager()
   ```

2. **Add stage selection logic**:
   ```python
   def select_stage_variant(self, stage_name: str, model_type: str = None, mode: str = None) -> str:
       """Select appropriate stage variant based on configuration"""
       if stage_name == 'ectd':
           return 'enhanced_ectd' if model_type == 'gpt5-mini' else 'ectd'
       elif stage_name == 'triple_generation':
           return 'enhanced_triple_generation'
       elif stage_name == 'graph_judge':
           return 'graph_judge_phase'  # Always use modular system
       return stage_name
   
   def configure_stage_mode(self, stage_name: str, mode_config: Dict[str, Any]) -> None:
       """Configure modular stage operation mode (standard/explainable/bootstrap)"""
       stage = self.stages.get(stage_name)
       if stage and hasattr(stage, 'update_configuration'):
           stage.update_configuration(mode_config)
       elif stage_name == 'graph_judge_phase' and stage:
           # Direct configuration for modular components
           stage.graph_judge.update_config(mode_config)
   ```

### Phase 4: Configuration and Environment Setup

#### Step 4.1: Unified Configuration System

**Objective**: Create a comprehensive configuration system for all modules.

**Implementation**:

1. **Create `unified_config.py`**:
   ```python
   @dataclass
   class UnifiedPipelineConfig:
       # ECTD Configuration
       ectd_config: Dict[str, Any] = field(default_factory=lambda: {
           'model_type': 'gpt5-mini',
           'temperature': 0.3,
           'batch_size': 20,
           'cache_enabled': True,
           'parallel_workers': 5
       })
       
       # Triple Generation Configuration
       triple_generation_config: Dict[str, Any] = field(default_factory=lambda: {
           'output_format': 'json',
           'schema_validation_enabled': True,
           'text_chunking_enabled': True,
           'post_processing_enabled': True
       })
       
       # Graph Judge Phase Configuration
       graph_judge_phase_config: Dict[str, Any] = field(default_factory=lambda: {
           'explainable_mode': True,
           'model_name': 'sonar-reasoning',
           'reasoning_effort': 'medium'
       })
   ```

2. **Create configuration templates**:
   ```yaml
   # config_templates/gpt5-mini_pipeline.yaml
   ectd_config:
     model_type: "gpt5-mini"
     temperature: 0.3
     batch_size: 20
     cache_enabled: true
   
   triple_generation_config:
     schema_validation_enabled: true
     text_chunking_enabled: true
   
   graph_judge_phase_config:
     explainable_mode: true
     model_name: "sonar-reasoning"
   ```

#### Step 4.2: Environment Variable Management

**Objective**: Standardize and manage environment variables across all modules.

**Implementation**:

1. **Create `environment_manager.py`**:
   ```python
   class EnvironmentManager:
       @staticmethod
       def setup_pipeline_environment(iteration: int, iteration_path: str, 
                                    model_type: str = "gpt5-mini") -> Dict[str, str]:
           """Setup comprehensive environment for pipeline execution"""
           env = os.environ.copy()
           
           # Common pipeline variables
           env.update({
               'PIPELINE_ITERATION': str(iteration),
               'PIPELINE_ITERATION_PATH': iteration_path,
               'PIPELINE_MODEL_TYPE': model_type,
               'PIPELINE_DATASET_PATH': f"../datasets/GPT5Mini_result_DreamOf_RedChamber/",
               'PYTHONIOENCODING': 'utf-8',
               'LANG': 'en_US.UTF-8'
           })
           
           # Model-specific variables
           if model_type == 'gpt5-mini':
               env.update({
                   'GPT5_MINI_MODEL': 'gpt-5-mini',
                   'OPENAI_TEMPERATURE': '0.3',
                   'OPENAI_MAX_TOKENS': '4000'
               })
           
           return env
   ```

### Phase 5: Testing and Validation

#### Step 5.1: Integration Testing

**Objective**: Ensure all integrated modules work correctly together.

**Test Cases**:

1. **End-to-End Pipeline Test**:
   ```python
   async def test_complete_pipeline():
       """Test complete pipeline with GPT-5-mini model"""
       pipeline = EnhancedKGPipeline()
       success = await pipeline.run_pipeline(
           input_file="test_input.txt",
           iteration=1,
           model_type="gpt5-mini"
       )
       assert success == True
   ```

2. **Stage-by-Stage Testing**:
   ```python
   async def test_enhanced_ectd_stage():
       """Test enhanced ECTD stage"""
       stage = EnhancedECTDStage(config)
       success = await stage.execute(iteration=1, iteration_path="test_path")
       assert success == True
   ```

3. **Configuration Testing**:
   ```python
   def test_configuration_loading():
       """Test configuration loading and validation"""
       config = UnifiedPipelineConfig()
       assert config.ectd_config['model_type'] == 'gpt5-mini'
   ```

#### Step 5.2: Backward Compatibility Testing

**Objective**: Ensure existing functionality remains intact.

**Test Cases**:

1. **Legacy Script Compatibility**:
   ```python
   def test_legacy_script_compatibility():
       """Test that original scripts still work independently"""
       # Test run_entity.py independently
       # Test run_triple.py independently
       # Test graphJudge_Phase independently
   ```

2. **CLI Backward Compatibility**:
   ```python
   def test_cli_backward_compatibility():
       """Test that existing CLI commands still work"""
       # Test existing CLI commands
       # Test existing configuration files
   ```

### Phase 6: Documentation and Deployment

#### Step 6.1: Documentation Updates

**Objective**: Update all documentation to reflect the integrated system.

**Actions**:

1. **Update CLI README**:
   - Add new command examples
   - Document configuration options
   - Provide migration guide

2. **Create Integration Guide**:
   - Step-by-step integration instructions
   - Configuration examples
   - Troubleshooting guide

3. **Update API Documentation**:
   - Document new stage classes
   - Update configuration schemas
   - Provide usage examples

#### Step 6.2: Deployment Preparation

**Objective**: Prepare the integrated system for deployment.

**Actions**:

1. **Create Deployment Scripts**:
   ```bash
   # deploy_integrated_cli.sh
   #!/bin/bash
   echo "Deploying Integrated CLI System..."
   # Copy new files
   # Update configurations
   # Run tests
   # Backup existing system
   ```

2. **Create Migration Scripts**:
   ```python
   # migrate_to_integrated_system.py
   def migrate_existing_configurations():
       """Migrate existing configurations to new format"""
       pass
   ```

## Implementation Timeline

### Week 1-2: Phase 1 - Module Integration and Analysis
- [x] Update `stage_manager.py` with enhanced stages and pipeline state integration
- [x] Create comprehensive configuration integration for all three modules
- [x] Standardize environment variables across all modules
- [x] Analyze existing pipeline state management system integration needs
- [x] Update the test of `Miscellaneous\KgGen\GraphJudge\chat\unit_test\test_unified_cli_pipeline.py` and solve the bugs.

### Week 3-4: Phase 2 - Enhanced Stage Implementation
- [x] Implement `EnhancedECTDStage` with GPT-5-mini caching and rate limiting
- [x] Implement `EnhancedTripleGenerationStage` with schema validation and chunking
- [x] Implement `EnhancedGraphJudgeStage` with full integrate `Miscellaneous\KgGen\GraphJudge\chat\graphJudge_Phase` feature set
- [x] Integrate gold label bootstrapping functionality
- [x] Add explainable reasoning and streaming capabilities
- [x] Update the test of `Miscellaneous\KgGen\GraphJudge\chat\unit_test\test_unified_cli_pipeline.py` and solve the bugs.

### Week 5-6: Phase 3 - CLI Enhancement and Advanced Features

#### **Critical Issues Identified and Solutions**

Based on runtime analysis, Phase 3 addresses three critical issues that prevent the CLI from executing smoothly:

#### **Issue 1: Incorrect Model Configuration Display**
**Problem**: The CLI displays `{'model': 'kimi-k2', 'temperature': 0.3, 'batch_size': 20, 'cache_enabled': True}` but the documentation states that `run_entity.py` should use GPT-5-mini.

**Root Cause Analysis**:
- The `PipelineConfig` class in `config_manager.py` defaults to `'model': 'gpt5-mini'` (line 51)
- However, the actual stage execution shows `'model': 'kimi-k2'` in the terminal output
- This indicates a configuration override or fallback mechanism is activating

**Solution**:
1. **Update `config_manager.py`** to ensure proper model selection:
   ```python
   # In PipelineConfig.__post_init__()
   self.ectd_config = {
       'model': 'gpt5-mini',  # Primary model for enhanced processing
       'fallback_model': 'kimi-k2',  # Only for fallback scenarios
       'model_priority': ['gpt5-mini', 'kimi-k2'],  # Explicit priority order
       'force_primary_model': True,  # Prevent automatic fallback
       'validate_model_availability': True,  # Check model accessibility
       # ... other config options
   }
   ```

2. **Enhance stage execution logic** in `stage_manager.py`:
   ```python
   # In ECTDStage.execute()
   def _validate_model_configuration(self):
       """Ensure correct model is being used"""
       primary_model = self.config.get('model', 'gpt5-mini')
       if primary_model != 'gpt5-mini':
           print(f"⚠️  WARNING: Expected GPT-5-mini but configured for {primary_model}")
           if self.config.get('force_primary_model', True):
               self.config['model'] = 'gpt5-mini'
               print(f"✅ Corrected model configuration to: gpt5-mini")
   ```

#### **Issue 2: Lack of Real-time Processing Display**
**Problem**: Users cannot see the ongoing processing status. The CLI only shows output after stage completion, making it appear frozen during long-running operations.

**Current Behavior**:
```
 [DEBUG] Executing command: python run_entity.py...
 [DEBUG] Working directory: .../chat
 [DEBUG] Environment encoding: utf-8
[Long silence - no output for 30+ minutes]
[Suddenly all output appears at once]
```

**Solution Implementation**:

1. **Create Real-time Output Streaming** in `stage_manager.py`:
   ```python
   async def _safe_subprocess_exec_with_streaming(self, cmd_args: List[str], env: Dict[str, str], 
                                                 cwd: str, stage_name: str) -> tuple[int, str]:
       """
       Enhanced subprocess execution with real-time output streaming.
       """
       try:
           print(f"🚀 Starting {stage_name} stage execution...")
           print(f"📝 Real-time progress will be displayed below:")
           print(f"{'='*80}")
           
           # Create subprocess with real-time output
           process = await asyncio.create_subprocess_exec(
               *cmd_args,
               stdout=asyncio.subprocess.PIPE,
               stderr=asyncio.subprocess.STDOUT,
               env=env,
               cwd=cwd
           )
           
           output_lines = []
           
           # Stream output line by line
           while True:
               line = await process.stdout.readline()
               if not line:
                   break
                   
               try:
                   decoded_line = line.decode('utf-8').rstrip()
               except UnicodeDecodeError:
                   decoded_line = line.decode('latin-1', errors='replace').rstrip()
               
               if decoded_line:
                   # Display real-time progress
                   timestamp = datetime.now().strftime("%H:%M:%S")
                   print(f"[{timestamp}] {decoded_line}")
                   output_lines.append(decoded_line)
                   
                   # Flush output immediately for real-time display
                   sys.stdout.flush()
           
           await process.wait()
           full_output = '\n'.join(output_lines)
           
           print(f"{'='*80}")
           print(f"✅ {stage_name} stage execution completed with return code: {process.returncode}")
           
           return process.returncode, full_output
           
       except Exception as e:
           error_msg = f"Real-time subprocess execution failed for {stage_name}: {str(e)}"
           print(f"❌ {error_msg}")
           return 1, error_msg
   ```

2. **Add Progress Indicators** for each stage:
   ```python
   class EnhancedProgressTracker:
       def __init__(self, stage_name: str):
           self.stage_name = stage_name
           self.start_time = datetime.now()
           self.last_update = self.start_time
       
       def log_progress(self, message: str, force_display: bool = False):
           """Log progress with timestamps and elapsed time"""
           current_time = datetime.now()
           elapsed = (current_time - self.start_time).total_seconds()
           
           # Show progress every 30 seconds or on force
           if force_display or (current_time - self.last_update).total_seconds() >= 30:
               print(f"[{current_time.strftime('%H:%M:%S')}] [{self.stage_name}] "
                     f"[Elapsed: {elapsed:.1f}s] {message}")
               self.last_update = current_time
               sys.stdout.flush()
   ```

#### **Issue 3: Output File Path Mismatch**
**Problem**: The CLI expects output files in one location but `run_entity.py` saves them in a different location, causing false failure detection.

**Current Behavior Analysis**:
- **CLI expects**: `D:\...\Iteration3\results\ectd\test_entity.txt`
- **run_entity.py saves to**: `../datasets/KIMI_result_DreamOf_RedChamber/Graph_Iteration3/`
- **Result**: CLI reports "Missing expected output files" despite successful execution

**Root Cause**:
The `_validate_stage_output()` method in `stage_manager.py` uses a different path than what `run_entity.py` actually outputs to.

**Solution Implementation**:

1. **Unified Output Path Configuration**:
   ```python
   # In stage_manager.py - Update environment setup
   def _setup_stage_environment(self, stage_name: str, iteration: int, iteration_path: str) -> Dict[str, str]:
       if self.env_manager:
           return self.env_manager.setup_stage_environment(stage_name, iteration, iteration_path)
       else:
           env = os.environ.copy()
           env['PYTHONIOENCODING'] = 'utf-8'
           env['LANG'] = 'en_US.UTF-8'
           
           # Unified output directory configuration
           if stage_name == "ectd":
               # Match run_entity.py expected output structure
               dataset_base = "../datasets/KIMI_result_DreamOf_RedChamber/"
               env['PIPELINE_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
               env['ECTD_OUTPUT_DIR'] = f"{dataset_base}Graph_Iteration{iteration}"
               # Also set the iteration path for backward compatibility
               env['PIPELINE_ITERATION_PATH'] = iteration_path
           
           env['PIPELINE_ITERATION'] = str(iteration)
           env['PIPELINE_DATASET_PATH'] = "../datasets/KIMI_result_DreamOf_RedChamber/"
           
           return env
   ```

2. **Enhanced Output Validation**:
   ```python
   def _validate_stage_output(self, stage_name: str, env: Dict[str, str]) -> bool:
       """Enhanced validation with multiple path checking"""
       expected_files = []
       
       if stage_name == "ectd":
           # Check both new unified path and legacy path
           primary_output_dir = env.get('ECTD_OUTPUT_DIR', env.get('PIPELINE_OUTPUT_DIR', ''))
           legacy_output_dir = os.path.join(env.get('PIPELINE_ITERATION_PATH', ''), "results", "ectd")
           
           primary_files = [
               os.path.join(primary_output_dir, "test_entity.txt"),
               os.path.join(primary_output_dir, "test_denoised.target")
           ]
           
           legacy_files = [
               os.path.join(legacy_output_dir, "test_entity.txt"), 
               os.path.join(legacy_output_dir, "test_denoised.target")
           ]
           
           # Check primary location first
           if all(os.path.exists(f) for f in primary_files):
               expected_files = primary_files
               print(f"✅ Found output files in primary location: {primary_output_dir}")
           elif all(os.path.exists(f) for f in legacy_files):
               expected_files = legacy_files
               print(f"✅ Found output files in legacy location: {legacy_output_dir}")
           else:
               print(f"❌ Output files not found in either location:")
               print(f"   Primary: {primary_output_dir}")
               print(f"   Legacy: {legacy_output_dir}")
               return False
       
       # Log successful validation
       if expected_files:
           print(f"✅ {stage_name} stage - All expected output files validated:")
           for file_path in expected_files:
               file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
               print(f"   ✓ {file_path} ({file_size:,} bytes)")
           return True
       
       return False
   ```

#### **Implementation Tasks**

##### **Task 3.1: Fix Model Configuration**
- [x] Ensure the cli is completely follow the original code (run_entity.py`, `run_triple.py`, and `graphJudge_Phase` modular system) to execute and run smoothly (Model configuration fixed).
- [x] Update `config_manager.py` to enforce GPT-5-mini for ECTD
- [x] Add model validation in stage execution
- [x] Implement configuration override protection
- [x] Add clear model selection logging

##### **Task 3.2: Implement Real-time Output Streaming**
- [x] Replace `_safe_subprocess_exec` with streaming version
- [x] Add progress tracking for long-running operations  
- [x] Implement timestamped logging
- [x] Add elapsed time indicators

##### **Task 3.3: Fix Output Path Validation**
- [x] Unify output directory configuration
- [x] Update environment variable setup
- [x] Enhance output file validation logic
- [x] Add support for multiple output locations

##### **Task 3.4: Enhanced CLI Interface**
- [x] Add real-time progress indicators
- [x] Improve error messaging and debugging
- [x] Add configuration validation commands
- [x] Implement stage-specific status reporting

##### **Update the Tests**(Wait my conform to implement.)
- [x] Update the Tests of the `test_unified_cli_pipeline.py`.
- [x] Complete Debug the Tests.

#### **Expected Outcomes After Phase 3**

1. **Correct Model Usage**: CLI will consistently use GPT-5-mini for ECTD stage as documented
2. **Real-time Feedback**: Users will see live progress during stage execution
3. **Accurate Status Reporting**: CLI will correctly detect successful stage completion
4. **Improved User Experience**: Clear, informative output with proper error handling

#### **Testing Strategy**

1. **Model Configuration Testing**:
   ```bash
   # Verify correct model is used
   python cli.py run-ectd --verbose --dry-run
   ```

2. **Real-time Output Testing**:
   ```bash
   # Test streaming output
   python cli.py run-pipeline --input small_test.txt --real-time
   ```

3. **Output Validation Testing**:
   ```bash
   # Test path resolution
   python cli.py run-ectd --input test.txt --validate-paths
   ```

### Week 7-8: Phase 4 - Configuration and Environment Setup
- [ ] Create unified configuration system supporting all advanced features
- [ ] Implement comprehensive environment variable management
- [ ] Create configuration templates for different use cases
- [ ] Add support for RapidFuzz and citation management configuration

### Week 9-10: Phase 5 - Testing and Validation
- [ ] Implement comprehensive integration tests for all three modules
- [ ] Test backward compatibility with existing workflows
- [ ] Performance testing with large datasets
- [ ] Test gold label bootstrapping pipeline end-to-end
- [ ] Validate explainable reasoning output quality

### Week 11-12: Phase 6 - Documentation and Deployment
- [ ] Update comprehensive documentation for all features
- [ ] Create deployment scripts and migration tools
- [ ] Final testing with production data
- [ ] User training materials for new advanced features

## Risk Mitigation

### Technical Risks

1. **Module Compatibility Issues**:
   - **Risk**: Different modules may have conflicting dependencies
   - **Mitigation**: Create isolated virtual environments for each module
   - **Fallback**: Maintain separate execution paths

2. **Configuration Conflicts**:
   - **Risk**: Configuration options may conflict between modules
   - **Mitigation**: Use namespaced configuration keys
   - **Fallback**: Provide default configurations

3. **Performance Degradation**:
   - **Risk**: Integration may slow down individual modules
   - **Mitigation**: Implement performance monitoring
   - **Fallback**: Allow individual module execution

### Operational Risks

1. **Backward Compatibility**:
   - **Risk**: Existing workflows may break
   - **Mitigation**: Maintain legacy interfaces
   - **Fallback**: Provide migration scripts

2. **User Adoption**:
   - **Risk**: Users may resist new integrated system
   - **Mitigation**: Provide comprehensive documentation and training
   - **Fallback**: Allow gradual migration

## Success Criteria

### Functional Requirements

1. **Complete Integration**: All three modules work seamlessly within the CLI
2. **Backward Compatibility**: Existing scripts and configurations continue to work
3. **Enhanced Features**: New features from all modules are available
4. **Configuration Flexibility**: Users can configure all aspects of the pipeline

### Performance Requirements

1. **No Performance Degradation**: Integrated system performs as well as individual modules
2. **Scalability**: System can handle large datasets efficiently
3. **Reliability**: System is stable and handles errors gracefully

### Usability Requirements

1. **Intuitive Interface**: CLI is easy to use and understand
2. **Comprehensive Documentation**: All features are well documented
3. **Error Handling**: Clear error messages and recovery options

## Conclusion

This implementation plan provides a comprehensive roadmap for integrating the three existing modules (`run_entity.py`, `run_triple.py`, and the modular `graphJudge_Phase` system) into the unified CLI system. The phased approach ensures minimal disruption to existing workflows while providing significantly enhanced functionality and maintainability through proper modular architecture.

The key benefits of this integration include:

1. **Unified Interface**: Single CLI for all knowledge graph operations with advanced mode support
2. **Enhanced Features**: Access to all advanced features including:
   - GPT-5-mini entity extraction with intelligent caching (run_entity.py)
   - Schema validation and text chunking for triple generation (run_triple.py)  
   - Modular graph judgment with explainable reasoning and gold label bootstrapping (graphJudge_Phase)
3. **Advanced Architecture**: Integration with existing pipeline state management system for robust error handling
4. **Flexible Operation Modes**: Support for standard, explainable, bootstrap, and streaming modes
5. **Comprehensive Monitoring**: Unified logging, progress tracking, citation management, and terminal progress files
6. **Production-Ready Features**: Advanced retry mechanisms, rate limiting, validation, and configuration management

### Advanced Capabilities Integration

The integration leverages sophisticated features from each module:

- **Intelligent Caching System**: From run_entity.py with SHA256-based cache keys and hit/miss tracking
- **Modular Graph Judge Architecture**: Clean separation of concerns with PerplexityGraphJudge, GoldLabelBootstrapper, and ProcessingPipeline components
- **Comprehensive API Integration**: Multiple model support (GPT-5-mini, Perplexity sonar models) with intelligent rate limiting
- **Gold Label Bootstrapping**: Two-stage RapidFuzz + LLM semantic evaluation pipeline with configurable sampling
- **Explainable AI**: Detailed confidence scores, reasoning explanations, and alternative suggestions through structured data types
- **Modular Testing Framework**: Comprehensive unit tests for each component ensuring reliability and maintainability
- **Type-Safe Data Structures**: TripleData, ExplainableJudgment, and BootstrapResult with proper validation
- **Flexible Configuration System**: Centralized configuration management with validation and environment variable support

The implementation follows best practices for enterprise software integration, including backward compatibility, comprehensive testing, pipeline state management, and gradual deployment. This ensures a smooth transition while maximizing the benefits of the integrated system and providing a robust foundation for future enhancements.
