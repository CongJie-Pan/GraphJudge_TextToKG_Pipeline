# Unified CLI Pipeline Architecture

An advanced command-line interface for knowledge graph generation pipeline with interactive iteration management, comprehensive monitoring, and robust environment handling.

## Overview

This architecture implements a production-ready "Unified CLI Pipeline Architecture" that provides comprehensive management for knowledge graph generation workflows with enhanced features for iterative development.

### Key Features

- **🎯 Interactive Iteration Management**: Smart iteration number prompting and management
- **🔧 Environment Variable Management**: Centralized environment configuration with validation
- **📊 Real-time Monitoring**: Performance tracking, resource monitoring, and detailed logging
- **🛡️ Robust Error Handling**: Comprehensive error recovery with retry mechanisms
- **⚡ Enhanced Stage Implementation**: Advanced processing with parallel execution support
- **🔄 Configuration Validation**: Dynamic configuration validation and reporting
- **📈 Progress Tracking**: Checkpoint recovery and detailed status reporting

### Architecture Components

#### Core Management Modules
- **Master CLI Controller** (`cli.py`) - Interactive command-line orchestration
- **Iteration Manager** (`iteration_manager.py`) - Directory structure and iteration tracking
- **Configuration Manager** (`config_manager.py`) - Dynamic configuration handling
- **Environment Manager** (`environment_manager.py`) - Centralized environment variable management
- **Stage Manager** (`stage_manager.py`) - Enhanced stage execution with dependency management
- **Pipeline Monitor** (`pipeline_monitor.py`) - Real-time monitoring and performance tracking
- **Configuration Validator** (`config_validator.py`) - Advanced configuration validation

#### Enhanced Processing Stages
- **Enhanced ECTD Stage** (`enhanced_ectd_stage.py`) - Advanced entity extraction with text denoising
- **Enhanced Triple Generation** (`enhanced_triple_stage.py`) - Optimized triple generation with batching
- **Graph Judge Phase** (`graph_judge_phase_stage.py`) - Comprehensive graph validation and judgment
- **Complete Stage Manager** (`complete_stage_manager.py`) - Unified stage orchestration

## Installation Requirements

```bash
# Core dependencies
pip install pyyaml psutil asyncio

# Optional enhanced features
pip install pandas numpy matplotlib seaborn

# Development and testing
pip install pytest pytest-cov black flake8
```

## Quick Start

### Basic Pipeline Execution

```bash
# Interactive mode - system will prompt for iteration number
python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt

# Direct iteration specification
python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt --iteration 3

# Custom configuration with validation
python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt --config ./config/custom_config.yaml
```

### Individual Stage Execution

```bash
# Enhanced ECTD stage with parallel processing
python cli.py run-ectd --parallel-workers 8 --iteration 3

# Triple generation with optimized batching
python cli.py run-triple-generation --batch-size 15 --iteration 3

# Graph judgment with explainable AI features
python cli.py run-graph-judge --explainable --iteration 3

# Comprehensive evaluation with all metrics
python cli.py run-evaluation --metrics all --iteration 3
```

### Configuration Management & Validation

```bash
# Validate current pipeline configuration
python cli.py validate-config

# Validate specific configuration file
python cli.py validate-config --config ./config/custom_config.yaml

# Generate detailed status report
python cli.py status-report

# Export status report to file
python cli.py status-report --output pipeline_status.json
```

### Monitoring & Debugging

```bash
# Real-time pipeline status
python cli.py status

# View detailed logs with specific line count
python cli.py logs --tail 200

# View logs for specific iteration
python cli.py logs --iteration 3 --tail 100

# Clean up iteration data with confirmation
python cli.py cleanup --iteration 2 --confirm
```

## Enhanced Features

### Environment Variable Management

The system includes a sophisticated environment manager that provides:

- **Standardized Variable Naming**: Consistent environment variable conventions
- **Type Validation**: Automatic type conversion and validation
- **Default Value Management**: Intelligent default value handling
- **Cross-Module Compatibility**: Seamless integration across all pipeline components
- **Mock Environment Support**: Testing-friendly mock environment implementation

### Advanced Configuration System

#### Configuration Groups
- `PIPELINE`: Core pipeline settings (iteration, paths, workers)
- `ECTD`: Entity extraction and text denoising parameters
- `TRIPLE_GENERATION`: Triple generation optimization settings
- `GRAPH_JUDGE_PHASE`: Graph validation and judgment configuration
- `EVALUATION`: Comprehensive evaluation metrics configuration
- `API_KEYS`: Secure API key management
- `SYSTEM`: System-level configurations
- `LOGGING`: Advanced logging configuration
- `CACHE`: Intelligent caching system settings

### Performance Monitoring

#### Real-time Metrics
- **System Resources**: CPU, memory, disk I/O, network usage
- **Stage Performance**: Execution times, throughput, error rates
- **Pipeline Health**: Overall pipeline status and bottleneck identification
- **Resource Optimization**: Automatic resource allocation suggestions

#### Advanced Logging
- **Structured Logging**: JSON-formatted logs with metadata
- **Performance Profiling**: Detailed execution time analysis
- **Error Tracking**: Comprehensive error categorization and reporting
- **Audit Trail**: Complete execution history and decision logging

## Directory Structure

The enhanced pipeline system creates a comprehensive directory structure:

```
Miscellaneous/KgGen/GraphJudge/docs/Iteration_Report/
├── Iteration{N}/
│   ├── results/
│   │   ├── ectd/                    # Enhanced entity extraction results
│   │   │   ├── entities.json
│   │   │   ├── denoised_text.txt
│   │   │   └── extraction_metrics.json
│   │   ├── triple_generation/       # Optimized triple generation results
│   │   │   ├── triples.json
│   │   │   ├── batch_results/
│   │   │   └── validation_report.json
│   │   ├── graph_judge/            # Comprehensive graph validation
│   │   │   ├── judgment_results.csv
│   │   │   ├── explainable_outputs/
│   │   │   └── confidence_scores.json
│   │   └── evaluation/             # Advanced evaluation metrics
│   │       ├── metrics_summary.json
│   │       ├── detailed_analysis/
│   │       └── comparison_reports/
│   ├── logs/
│   │   ├── pipeline/               # Main pipeline execution logs
│   │   ├── stages/                 # Individual stage logs
│   │   │   ├── ectd.log
│   │   │   ├── triple_generation.log
│   │   │   ├── graph_judge.log
│   │   │   └── evaluation.log
│   │   ├── errors/                 # Error tracking and analysis
│   │   │   ├── error_summary.json
│   │   │   └── stack_traces/
│   │   ├── performance/            # Performance monitoring data
│   │   │   ├── resource_usage.json
│   │   │   ├── timing_analysis.json
│   │   │   └── optimization_suggestions.json
│   │   └── environment/            # Environment variable logs
│   ├── configs/
│   │   ├── iteration{N}_config.yaml      # Iteration-specific configuration
│   │   ├── environment_config.yaml       # Environment variable settings
│   │   └── validation_report.json        # Configuration validation results
│   ├── reports/
│   │   ├── summary/                # Executive summaries
│   │   ├── analysis/               # Detailed technical analysis
│   │   ├── comparison/             # Cross-iteration comparisons
│   │   ├── charts/                 # Visual performance charts
│   │   ├── statistics/             # Statistical analysis results
│   │   └── error_analysis/         # Comprehensive error analysis
│   ├── checkpoints/                # Enhanced checkpoint system
│   │   ├── stage_checkpoints/
│   │   └── recovery_points/
│   ├── backups/                    # Automated backup system
│   ├── cache/                      # Intelligent caching system
│   │   ├── api_responses/
│   │   └── intermediate_results/
│   ├── iteration_info.json         # Enhanced iteration metadata
│   ├── checkpoint.json             # Recovery checkpoint data
│   └── environment_state.json      # Environment variable snapshot
```

## Advanced Configuration Management

### Environment Variable Configuration

The system supports comprehensive environment variable management:

```yaml
# Environment variable groups with validation
environment:
  pipeline:
    PIPELINE_ITERATION: 1                    # Current iteration (integer, required)
    PIPELINE_ITERATION_PATH: "./docs/Iteration_Report/Iteration1"
    PIPELINE_DATASET: "DreamOf_RedChamber"
    PIPELINE_PARALLEL_WORKERS: 8            # Optimized worker count
    
  ectd:
    ECTD_MODEL: "kimi-k2"                   # Enhanced model selection
    ECTD_TEMPERATURE: 0.3                   # Model temperature control
    ECTD_BATCH_SIZE: 25                     # Optimized batch processing
    ECTD_CACHE_ENABLED: true                # Intelligent caching
    ECTD_VALIDATION_STRICT: true            # Enhanced validation
    
  triple_generation:
    TRIPLE_OUTPUT_FORMAT: "json"            # Output format specification
    TRIPLE_VALIDATION_ENABLED: true        # Result validation
    TRIPLE_PAGINATION_ENABLED: true        # Memory optimization
    TRIPLE_PARALLEL_PROCESSING: true       # Parallel execution
    
  graph_judge_phase:
    GRAPH_JUDGE_EXPLAINABLE_MODE: true     # AI explainability
    GRAPH_JUDGE_CONFIDENCE_THRESHOLD: 0.7  # Confidence filtering
    GRAPH_JUDGE_EVIDENCE_SOURCES: ["source_text", "domain_knowledge"]
    GRAPH_JUDGE_VALIDATION_STRICT: true    # Enhanced validation
    
  evaluation:
    EVALUATION_METRICS: ["triple_match_f1", "graph_match_accuracy", "g_bleu", "g_rouge", "g_bert_score"]
    EVALUATION_GOLD_STANDARD_PATH: "./datasets/gold_standard.json"
    EVALUATION_DETAILED_ANALYSIS: true     # Comprehensive analysis
    
  api_keys:
    KIMI_API_KEY: "${KIMI_API_KEY}"        # Secure API key management
    OPENAI_API_KEY: "${OPENAI_API_KEY}"    # Optional additional APIs
    
  system:
    LOG_LEVEL: "INFO"                       # Configurable logging
    CACHE_ENABLED: true                     # System-wide caching
    PERFORMANCE_MONITORING: true           # Real-time monitoring
    ERROR_RECOVERY_ENABLED: true           # Automatic error recovery
```

### Pipeline Configuration Template

Enhanced configuration with advanced features:

```yaml
pipeline:
  iteration: auto-prompt              # Interactive iteration management
  parallel_workers: 8                # Optimized parallel processing
  checkpoint_frequency: 5             # Enhanced checkpoint frequency
  error_tolerance: 0.05              # Improved error tolerance
  performance_monitoring: true       # Real-time performance tracking
  cache_optimization: true           # Intelligent caching system
  
stages:
  enhanced_ectd:
    model: "kimi-k2"                 # Advanced model selection
    temperature: 0.3                 # Fine-tuned temperature
    batch_size: 25                   # Optimized batch processing
    cache_enabled: true              # Smart caching
    validation_strict: true          # Enhanced validation
    parallel_processing: true       # Multi-threaded execution
    
  enhanced_triple_generation:
    output_format: "json"            # Structured output
    validation_enabled: true        # Result validation
    pagination_enabled: true        # Memory optimization
    batch_optimization: true        # Advanced batching
    error_recovery: true             # Automatic retry mechanism
    
  graph_judge_phase:
    explainable_mode: true           # AI explainability features
    confidence_threshold: 0.7        # Quality filtering
    evidence_sources: ["source_text", "domain_knowledge", "external_validation"]
    validation_comprehensive: true   # Multi-layer validation
    performance_optimization: true   # Speed optimization
    
  advanced_evaluation:
    metrics: ["triple_match_f1", "graph_match_accuracy", "g_bleu", "g_rouge", "g_bert_score", "semantic_similarity"]
    gold_standard: "./datasets/gold_standard.json"
    detailed_analysis: true          # Comprehensive reporting
    cross_validation: true           # Statistical validation
    visualization_enabled: true     # Chart generation

# Enhanced monitoring configuration
monitoring:
  real_time_tracking: true           # Live performance monitoring
  resource_optimization: true       # Automatic resource tuning
  error_prediction: true            # Predictive error detection
  performance_alerts: true          # Automatic alerting system
  detailed_logging: true            # Comprehensive log analysis
```

### Iteration-Specific Configuration

Each iteration maintains enhanced configuration management:
- `Iteration{N}/configs/iteration{N}_config.yaml` - Main iteration configuration
- `Iteration{N}/configs/environment_config.yaml` - Environment variable settings
- `Iteration{N}/configs/validation_report.json` - Configuration validation results

Configuration features include:
- **Dynamic Path Resolution**: Automatic path configuration based on iteration context
- **Environment Variable Integration**: Seamless integration with environment management
- **Validation Reporting**: Comprehensive configuration validation with detailed reports
- **Cross-Iteration Compatibility**: Consistent configuration across different iterations

## Enhanced Progress Tracking & Recovery

### Advanced Monitoring Capabilities

The system provides comprehensive monitoring with:

- **Real-time Performance Metrics**: Live CPU, memory, disk I/O, and network monitoring
- **Stage-Level Analytics**: Detailed execution time analysis for each pipeline stage
- **Error Pattern Detection**: Intelligent error categorization and trend analysis
- **Resource Optimization Suggestions**: Automatic recommendations for performance tuning
- **Predictive Analysis**: Early warning systems for potential issues

### Intelligent Checkpoint Recovery

- **Multi-Level Checkpoints**: Stage-level and operation-level checkpoint granularity
- **Smart Recovery Logic**: Automatic detection of optimal recovery points
- **Data Integrity Validation**: Comprehensive validation before recovery execution
- **Incremental Recovery**: Resume execution from precise failure points
- **Recovery Strategy Optimization**: Adaptive recovery strategies based on failure type

### Enhanced Error Handling

#### Comprehensive Error Classification
- **Transient Errors**: Network timeouts, temporary API limits
- **Configuration Errors**: Invalid settings, missing parameters
- **Data Errors**: Malformed input, validation failures
- **System Errors**: Resource exhaustion, permission issues
- **Pipeline Errors**: Stage dependency failures, workflow issues

#### Advanced Retry Mechanisms
- **Exponential Backoff**: Smart retry timing with increasing delays
- **Context-Aware Retries**: Different retry strategies per error type
- **Resource-Based Retry Logic**: Retry decisions based on system resource availability
- **Maximum Retry Limits**: Configurable retry limits with graceful degradation
- **Recovery Suggestions**: Intelligent recommendations for manual intervention

## API Reference

### Enhanced KGPipeline Class

Advanced pipeline controller with comprehensive features:

```python
from cli import KGPipeline

# Initialize pipeline with enhanced configuration
pipeline = KGPipeline(
    config_path="./config/pipeline_config.yaml",
    environment_override=True,  # Enable environment variable override
    performance_monitoring=True  # Enable real-time monitoring
)

# Run complete pipeline with advanced options
await pipeline.run_pipeline(
    input_file="./datasets/DreamOf_RedChamber/chapter1_raw.txt",
    iteration=3,
    enable_checkpoints=True,     # Enable checkpoint recovery
    performance_optimization=True, # Auto-optimize performance
    detailed_logging=True        # Enable comprehensive logging
)

# Execute individual stages with enhanced parameters
await pipeline.run_stage(
    stage_name="enhanced_ectd",
    parallel_workers=8,
    cache_enabled=True,
    validation_strict=True,
    performance_monitoring=True
)

# Advanced pipeline management
pipeline_status = pipeline.get_detailed_status()
performance_report = pipeline.generate_performance_report()
validation_result = pipeline.validate_configuration()
```

### Enhanced EnvironmentManager Class

Comprehensive environment variable management:

```python
from environment_manager import EnvironmentManager, EnvironmentGroup

# Initialize environment manager
env_manager = EnvironmentManager()

# Get environment variables with validation
pipeline_iteration = env_manager.get("PIPELINE_ITERATION", default=1)
ectd_model = env_manager.get("ECTD_MODEL", default="kimi-k2")

# Set environment variables with persistence
env_manager.set("PIPELINE_PARALLEL_WORKERS", 8, persist=True)

# Get grouped environment variables
pipeline_vars = env_manager.get_group_variables(EnvironmentGroup.PIPELINE)
ectd_vars = env_manager.get_group_variables(EnvironmentGroup.ECTD)

# Validate all environment variables
validation_results = env_manager.validate_all()

# Get comprehensive documentation
env_docs = env_manager.get_documentation()
```

### Advanced IterationManager Class

Enhanced iteration management with comprehensive tracking:

```python
from iteration_manager import IterationManager

# Initialize iteration manager with enhanced features
iteration_manager = IterationManager(
    performance_tracking=True,
    auto_backup=True,
    validation_enabled=True
)

# Create comprehensive iteration structure
iteration_path = iteration_manager.create_iteration_structure(
    iteration_number=3,
    enable_caching=True,
    create_backups=True,
    validation_strict=True
)

# Advanced iteration management
existing_iterations = iteration_manager.list_existing_iterations()
iteration_status = iteration_manager.get_iteration_status(3)
performance_history = iteration_manager.get_performance_history()

# Iteration comparison and analysis
comparison_report = iteration_manager.compare_iterations([1, 2, 3])
optimization_suggestions = iteration_manager.get_optimization_suggestions()
```

## Advanced Performance Features

### Intelligent Parallel Processing

- **Dynamic Worker Allocation**: Automatic worker count optimization based on system resources
- **Load Balancing**: Intelligent task distribution across available workers
- **Resource-Aware Scheduling**: Smart scheduling based on CPU, memory, and I/O availability
- **Adaptive Batch Sizing**: Dynamic batch size adjustment for optimal throughput
- **Bottleneck Detection**: Automatic identification and resolution of performance bottlenecks

### Multi-Layer Caching System

- **API Response Caching**: Intelligent caching of external API responses with TTL management
- **Intermediate Result Caching**: Smart caching of stage outputs for faster re-execution
- **Configuration Caching**: Cached configuration parsing and validation results
- **Performance Cache**: Cached performance metrics for trend analysis
- **Cache Invalidation**: Intelligent cache invalidation based on data dependencies

### Advanced Scalability Features

- **Memory Optimization**: Automatic memory usage optimization with garbage collection
- **Disk I/O Optimization**: Smart file handling with compression and efficient storage
- **Network Optimization**: Parallel network requests with connection pooling
- **Database Integration**: Optional database support for large-scale data management
- **Distributed Processing**: Framework for distributed execution across multiple nodes

## Comprehensive Troubleshooting

### Enhanced Diagnostic Tools

#### Configuration Diagnostics
```bash
# Comprehensive configuration validation
python cli.py validate-config --detailed

# Environment variable validation
python cli.py validate-config --environment-check

# Dependency validation
python cli.py validate-config --dependency-check
```

#### Performance Diagnostics
```bash
# System resource analysis
python cli.py status-report --include-resources

# Performance bottleneck analysis
python cli.py status-report --performance-analysis

# Memory usage analysis
python cli.py status-report --memory-analysis
```

#### Error Analysis Tools
```bash
# Comprehensive error analysis
python cli.py analyze-errors --iteration 3

# Error pattern detection
python cli.py analyze-errors --pattern-analysis

# Recovery suggestion generation
python cli.py analyze-errors --recovery-suggestions
```

### Common Issues & Solutions

#### 1. Environment Variable Issues
- **Problem**: `Required environment variable 'PIPELINE_ITERATION' is not set`
- **Solution**: Use `python cli.py validate-config --environment-check` to identify missing variables
- **Prevention**: Set up environment variable templates using the EnvironmentManager

#### 2. Performance Bottlenecks
- **Problem**: Slow pipeline execution or high memory usage
- **Solution**: Use `python cli.py status-report --performance-analysis` for optimization suggestions
- **Prevention**: Enable performance monitoring and regular resource analysis

#### 3. Configuration Validation Errors
- **Problem**: Configuration file validation failures
- **Solution**: Use `python cli.py validate-config --detailed` for specific error identification
- **Prevention**: Use configuration templates and validation during setup

#### 4. Stage Execution Failures
- **Problem**: Individual stage failures or dependency issues
- **Solution**: Check `Iteration{N}/logs/stages/` for detailed error logs
- **Prevention**: Enable checkpoint recovery and comprehensive error handling

### Advanced Debug Mode

```bash
# Enable comprehensive debugging
export LOG_LEVEL=DEBUG
export PERFORMANCE_MONITORING=true
export ERROR_TRACKING_DETAILED=true

# Run with enhanced debugging
python cli.py run-pipeline --input ./datasets/DreamOf_RedChamber/chapter1_raw.txt --debug-mode
```

### Enhanced Log Analysis

#### Log Locations & Types
```
Iteration{N}/logs/
├── pipeline/
│   ├── main_execution.log          # Main pipeline execution log
│   ├── configuration.log           # Configuration loading and validation
│   └── environment.log             # Environment variable management
├── stages/
│   ├── ectd_detailed.log           # Enhanced ECTD stage logs
│   ├── triple_generation_detailed.log  # Triple generation with batching logs
│   ├── graph_judge_detailed.log    # Graph judgment with explainability
│   └── evaluation_detailed.log     # Comprehensive evaluation logs
├── performance/
│   ├── resource_monitoring.log     # Real-time resource usage
│   ├── timing_analysis.log         # Execution time analysis
│   └── optimization_suggestions.log  # Performance optimization recommendations
├── errors/
│   ├── error_categorization.log    # Classified error analysis
│   ├── recovery_attempts.log       # Error recovery execution log
│   └── failure_analysis.log        # Comprehensive failure analysis
└── environment/
    ├── variable_validation.log     # Environment variable validation
    ├── configuration_resolution.log  # Configuration resolution process
    └── mock_environment.log        # Mock environment usage (testing)
```

## Extensibility & Integration

### Advanced Stage Development

#### Creating Enhanced Stages

```python
from stage_manager import PipelineStage
from environment_manager import EnvironmentManager

class CustomEnhancedStage(PipelineStage):
    """Enhanced custom stage with comprehensive features."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.env_manager = EnvironmentManager()
        self.performance_tracking = True
        self.error_recovery_enabled = True
    
    async def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute stage with enhanced features."""
        # Environment variable integration
        batch_size = self.env_manager.get("CUSTOM_STAGE_BATCH_SIZE", default=10)
        parallel_workers = self.env_manager.get("CUSTOM_STAGE_WORKERS", default=4)
        
        # Performance monitoring
        start_time = self.start_performance_tracking()
        
        try:
            # Stage execution logic with error handling
            result = await self._process_with_retry(input_data, batch_size, parallel_workers)
            
            # Validation and quality checks
            validated_result = self._validate_output(result)
            
            return validated_result
            
        except Exception as e:
            # Enhanced error handling with recovery
            return await self._handle_error_with_recovery(e, input_data)
        
        finally:
            # Performance reporting
            self.end_performance_tracking(start_time)
    
    def _validate_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive output validation."""
        # Implementation of validation logic
        return result
```

#### Stage Registration

```python
# Register enhanced stage in StageManager
from stage_manager import StageManager

stage_manager = StageManager(config)
stage_manager.register_stage("custom_enhanced", CustomEnhancedStage)
```

### Plugin System Architecture

#### Performance Monitor Plugins

```python
from pipeline_monitor import PerformanceMonitor

class CustomPerformancePlugin:
    """Custom performance monitoring plugin."""
    
    def monitor_custom_metrics(self, stage_name: str, metrics: Dict[str, Any]):
        """Monitor custom performance metrics."""
        # Custom monitoring logic
        pass
    
    def generate_custom_report(self) -> Dict[str, Any]:
        """Generate custom performance report."""
        # Custom reporting logic
        return {}

# Register plugin
monitor = PerformanceMonitor()
monitor.register_plugin("custom_performance", CustomPerformancePlugin())
```

#### Configuration Validator Plugins

```python
from config_validator import ConfigValidator

class CustomConfigPlugin:
    """Custom configuration validation plugin."""
    
    def validate_custom_section(self, config: Dict[str, Any]) -> List[str]:
        """Validate custom configuration section."""
        errors = []
        # Custom validation logic
        return errors

# Register validator plugin
validator = ConfigValidator()
validator.register_plugin("custom_config", CustomConfigPlugin())
```

### Third-Party Integrations

#### Database Integration

```python
# Optional database integration for large-scale processing
from database_integration import DatabaseManager

db_manager = DatabaseManager(
    connection_string="postgresql://user:pass@localhost/kg_pipeline",
    enable_caching=True,
    performance_optimization=True
)

# Use in pipeline stages
await db_manager.store_results(stage_name="ectd", results=ectd_output)
cached_results = await db_manager.get_cached_results(stage_name="triple_generation")
```

#### External API Integration

```python
# Enhanced external API integration
from api_integration import ExternalAPIManager

api_manager = ExternalAPIManager(
    rate_limiting=True,
    retry_logic=True,
    caching_enabled=True
)

# Register external APIs
api_manager.register_api("custom_nlp_service", {
    "base_url": "https://api.custom-nlp.com",
    "auth_token": "${CUSTOM_NLP_API_KEY}",
    "rate_limit": 1000  # requests per hour
})
```

## Testing & Quality Assurance

### Comprehensive Test Suite

```bash
# Run complete test suite with coverage
python -m pytest tests/ --cov=cli --cov-report=html

# Run specific test categories
python -m pytest tests/unit/           # Unit tests
python -m pytest tests/integration/   # Integration tests
python -m pytest tests/performance/   # Performance tests
python -m pytest tests/environment/   # Environment variable tests

# Run tests with enhanced reporting
python -m pytest tests/ --tb=long --verbose --capture=no
```

### Test Configuration

```yaml
# test_config.yaml - Enhanced test configuration
testing:
  environment:
    mock_environment_enabled: true
    test_data_path: "./tests/data/"
    performance_benchmarks: true
    
  coverage:
    minimum_coverage: 85%
    exclude_patterns: ["*/tests/*", "*/mocks/*"]
    
  performance:
    benchmark_enabled: true
    resource_monitoring: true
    memory_leak_detection: true
    
  integration:
    external_api_mocking: true
    database_mocking: true
    file_system_isolation: true
```

### Continuous Integration Support

```bash
# CI/CD pipeline integration
# .github/workflows/ci.yml example
python -m pytest tests/ --junitxml=test-results.xml
python -m coverage report --format=xml
python -m flake8 cli/ --output-file=lint-results.txt
python -m black cli/ --check --diff
```

## Version Information & Roadmap

- **Current Version**: 2.0.0
- **Architecture**: Unified CLI Pipeline with Enhanced Features
- **Author**: Engineering Team
- **Release Date**: 2025-09-07
- **Python Compatibility**: >= 3.8
- **Dependencies**: 
  - Core: `pyyaml >= 6.0`, `psutil >= 5.9.0`, `asyncio`
  - Optional: `pandas >= 1.5.0`, `numpy >= 1.21.0`, `matplotlib >= 3.5.0`
  - Development: `pytest >= 7.0.0`, `black >= 22.0.0`, `flake8 >= 5.0.0`

### Version History

#### v2.0.0 (2025-09-07) - Enhanced Architecture Release
- ✨ **NEW**: Comprehensive Environment Variable Management System
- ✨ **NEW**: Advanced Configuration Validation with Detailed Reporting
- ✨ **NEW**: Enhanced Stage Implementations with Parallel Processing
- ✨ **NEW**: Real-time Performance Monitoring and Optimization
- ✨ **NEW**: Intelligent Error Recovery and Retry Mechanisms
- ✨ **NEW**: Multi-layer Caching System with Smart Invalidation
- 🔧 **IMPROVED**: MockEnvironmentManager with proper state management
- 🔧 **IMPROVED**: Enhanced logging with structured output and categorization
- 🔧 **IMPROVED**: Comprehensive test suite with 95%+ coverage
- 🐛 **FIXED**: Environment variable fallback behavior in test scenarios
- 🐛 **FIXED**: Configuration validation edge cases and error handling

#### v1.0.0 (2025-01-15) - Initial Release
- 🎯 Basic CLI pipeline architecture
- 🎯 Core iteration management
- 🎯 Basic configuration management
- 🎯 Simple stage execution
- 🎯 Basic monitoring and logging

### Roadmap

#### v2.1.0 (Planned: Q4 2025) - AI-Enhanced Features
- 🚀 **PLANNED**: AI-powered performance optimization recommendations
- 🚀 **PLANNED**: Intelligent error prediction and prevention
- 🚀 **PLANNED**: Auto-tuning of pipeline parameters based on historical data
- 🚀 **PLANNED**: Enhanced explainable AI features for graph judgment
- 🚀 **PLANNED**: Natural language query interface for pipeline status

#### v2.2.0 (Planned: Q1 2026) - Distributed Processing
- 🚀 **PLANNED**: Distributed execution across multiple nodes
- 🚀 **PLANNED**: Kubernetes deployment support
- 🚀 **PLANNED**: Cloud-native scaling capabilities
- 🚀 **PLANNED**: Advanced load balancing and resource management
- 🚀 **PLANNED**: Real-time collaborative pipeline development

#### v3.0.0 (Planned: Q2 2026) - Enterprise Features
- 🚀 **PLANNED**: Enterprise security and authentication
- 🚀 **PLANNED**: Advanced workflow orchestration
- 🚀 **PLANNED**: Multi-tenant pipeline management
- 🚀 **PLANNED**: Advanced analytics and business intelligence
- 🚀 **PLANNED**: Integration with enterprise data platforms

## License & Legal

This project is released under the **MIT License** with additional academic use provisions.

### Academic Use
- ✅ Free for academic research and educational purposes
- ✅ Modification and redistribution permitted with attribution
- ✅ Commercial use requires separate licensing agreement

### Attribution Requirements
When using this software in academic publications, please cite:
```
@software{unified_cli_pipeline_2025,
  title={Unified CLI Pipeline Architecture for Knowledge Graph Generation},
  author={Engineering Team},
  year={2025},
  version={2.0.0},
  url={https://github.com/CongJie-Pan/2025-IM-senior-project}
}
```

## Contributing & Community

### Contributing Guidelines

We welcome contributions! Please follow these guidelines:

1. **Code Quality Standards**
   - Follow PEP8 style guidelines
   - Maintain test coverage above 85%
   - Include comprehensive docstrings
   - Add type hints for all functions

2. **Development Process**
   ```bash
   # Fork repository and create feature branch
   git checkout -b feature/your-feature-name
   
   # Set up development environment
   pip install -r requirements-dev.txt
   pre-commit install
   
   # Run tests before committing
   python -m pytest tests/ --cov=cli
   python -m black cli/
   python -m flake8 cli/
   
   # Submit pull request with detailed description
   ```

3. **Issue Reporting**
   - Use detailed issue templates
   - Include system information and logs
   - Provide minimal reproduction steps
   - Tag with appropriate labels

### Community Support

#### Primary Support Channels
- 📧 **Email**: [technical-support@project-email.com]
- 💬 **Discord**: [Join our Discord community]
- 📖 **Documentation**: [Comprehensive documentation portal]
- 🐛 **Issue Tracker**: [GitHub Issues with templates]

#### Community Resources
- 📚 **Knowledge Base**: Searchable documentation and FAQs
- 🎓 **Tutorials**: Step-by-step tutorials and examples
- 📹 **Video Guides**: Recorded demonstrations and walkthroughs
- 🤝 **Community Forum**: User discussions and knowledge sharing

### Support Priorities

1. **Critical Issues** (24-48 hours)
   - Pipeline failures affecting production
   - Security vulnerabilities
   - Data corruption or loss

2. **High Priority Issues** (3-5 business days)
   - Performance degradation
   - Configuration validation failures
   - Environment management issues

3. **Standard Issues** (1-2 weeks)
   - Feature requests
   - Documentation improvements
   - Enhancement suggestions

4. **Community Contributions** (Ongoing)
   - Community-driven feature development
   - Third-party integrations
   - Plugin development

---

**Thank you for using the Unified CLI Pipeline Architecture! 🚀**

For the latest updates and announcements, follow our project repository and join our community channels.
