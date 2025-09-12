# GraphJudge Streamlit Refactoring: Task Breakdown

**Version:** 1.0  
**Date:** 2025-09-12  
**Project:** Refactoring `run_entity.py`, `run_triple.py`, `run_gj.py` for Streamlit Integration  
**Reference:** [spec.md](./spec.md)

---

## Phase 1: Core Module Extraction

### [ ] **Task ID**: REF-001
- **Task Name**: Extract Entity Processor Module from run_entity.py
- **Work Description**:
    - **Why**: The current `run_entity.py` script (~800 lines) contains complex file I/O, logging, and async operations that make it unsuitable for Streamlit integration. Need to extract core GPT-5-mini entity extraction logic into a clean, testable module.
    - **How**: 
        1. **First, read `spec.md` Section 2, 3, 8** to understand refactoring strategy and module requirements
        2. Analyze `chat/run_entity.py` to identify core business logic
        3. Extract GPT-5-mini entity extraction functionality following spec.md guidelines
        4. Create simplified `extract_entities()` function with clean interface as defined in spec.md Section 8
        5. Implement in-memory result handling instead of file-based storage
        6. Add essential error handling without complex retry mechanisms per spec.md Section 10
- **Resources Required**:
    - **Materials**: Access to `chat/run_entity.py`, GPT-5-mini API credentials, development environment
    - **Personnel**: 1 Senior Developer (estimated 2-3 days)
    - **Reference Codes**: 
        - **Primary**: `streamlit_pipeline/docs/spec.md` Sections 2, 3, 8 (refactoring strategy and module contracts)
        - `chat/run_entity.py` lines 691-800 (main function)
        - `chat/openai_config.py` for API configuration
        - `chat/config.py` for API key management
- **Deliverables**:
    - [ ] `streamlit_pipeline/core/entity_processor.py` (~150-200 lines)
    - [ ] `EntityResult` data model in `models.py`
    - [ ] Unit tests in `test_entity_processor.py` (following `docs/Testing_Demands.md` TDD principles)
    - [ ] API integration documentation
- **Dependencies**: None (can start immediately)
- **Constraints**: 
    - Must preserve GPT-5-mini functionality
    - Target 80%+ code reduction from original script
    - Synchronous execution for Streamlit compatibility
- **Completion Status**: ❌ Not Started
- **Notes**: Priority task - foundation for other modules

### [ ] **Task ID**: REF-002
- **Task Name**: Create Shared Data Models and Configuration System
- **Work Description**:
    - **Why**: Need unified data structures (`Triple`, `EntityResult`, `TripleResult`, `JudgmentResult`) and simplified configuration management to replace complex file-based workflows across all modules.
    - **How**:
        1. **First, read `spec.md` Section 8** to understand data model definitions and module contracts
        2. Design clean data models using Python dataclasses following spec.md specifications
        3. Create unified configuration system for API keys and model parameters as outlined in spec.md
        4. Implement validation utilities for input data per spec.md requirements
        5. Set up shared constants and enums
- **Resources Required**:
    - **Materials**: Python dataclasses documentation, existing config patterns
    - **Personnel**: 1 Developer (estimated 1-2 days)
    - **Reference Codes**:
        - **Primary**: `streamlit_pipeline/docs/spec.md` Section 8 (data model definitions and module contracts)
        - `chat/config.py` for existing configuration patterns
        - `spec.md` Section 4 (assumptions & constraints) for configuration requirements
- **Deliverables**:
    - [ ] `streamlit_pipeline/core/models.py` with all data classes
    - [ ] `streamlit_pipeline/core/config.py` for configuration management
    - [ ] `streamlit_pipeline/utils/validation.py` for input validation
    - [ ] Type hints and documentation for all models
- **Dependencies**: None
- **Constraints**: 
    - Must be compatible with all three pipeline stages
    - Keep dependencies minimal
- **Completion Status**: ❌ Not Started  
- **Notes**: Can be developed in parallel with REF-001

### [ ] **Task ID**: REF-003
- **Task Name**: Set Up Test Framework with Mock API Responses
- **Work Description**:
    - **Why**: Need comprehensive test coverage with mocked API responses to enable reliable development and prevent API costs during testing. Target 90%+ test coverage for all refactored modules.
    - **How**:
        1. **First, read `docs/Testing_Demands.md`** to understand TDD principles and testing guidelines
        2. **Then read `spec.md` Section 15** to understand testing strategy and requirements
        3. Set up pytest framework with asyncio support following spec.md guidelines and Testing_Demands.md TDD principles
        4. Create mock API response fixtures for GPT-5-mini and Perplexity per spec.md Section 7 and Testing_Demands.md mock design practices
        5. Implement test utilities for common testing patterns as outlined in Testing_Demands.md best practices
        6. Create integration test harnesses meeting spec.md acceptance criteria and Testing_Demands.md integration testing requirements
- **Resources Required**:
    - **Materials**: pytest documentation, unittest.mock library
    - **Personnel**: 1 Developer (estimated 1-2 days)
    - **Reference Codes**:
        - **Primary**: `docs/Testing_Demands.md` (TDD principles and testing guidelines)
        - `streamlit_pipeline/docs/spec.md` Section 15 (testing strategy & matrix)
        - `chat/unit_test/` folder for existing test patterns
        - Mock examples from original test suites
        - `spec.md` Section 19 (acceptance criteria for test coverage)
- **Deliverables**:
    - [ ] `streamlit_pipeline/tests/` directory structure
    - [ ] `fixtures/` folder with mock API responses
    - [ ] Test configuration and utilities
    - [ ] CI/CD integration for automated testing
- **Dependencies**: REF-002 (needs data models)
- **Constraints**:
    - Must support both unit and integration tests
    - Mock responses must be realistic
- **Completion Status**: ❌ Not Started
- **Notes**: Critical for maintaining code quality

---

## Phase 2: Triple Generation Refactoring

### [ ] **Task ID**: REF-004
- **Task Name**: Extract Triple Generator Module from run_triple.py
- **Work Description**:
    - **Why**: The `run_triple.py` script (~750 lines) contains sophisticated JSON schema validation and text chunking that needs to be preserved while simplifying the overall structure for Streamlit integration.
    - **How**:
        1. **First, read `spec.md` Section 2, 3, 8** to understand triple generation refactoring requirements
        2. Analyze `chat/run_triple.py` to identify core triple generation logic following spec.md guidelines
        3. Extract JSON schema validation with Pydantic models as specified in spec.md Section 3 (FR-T2)
        4. Implement simplified text chunking for large inputs per spec.md Section 3 (FR-T3)
        5. Create clean `generate_triples()` interface as defined in spec.md Section 8
        6. Preserve prompt engineering sophistication while reducing complexity per spec.md Section 3 (FR-T4)
- **Resources Required**:
    - **Materials**: Access to `chat/run_triple.py`, Pydantic documentation
    - **Personnel**: 1 Senior Developer (estimated 3-4 days)
    - **Reference Codes**:
        - **Primary**: `streamlit_pipeline/docs/spec.md` Sections 2, 3, 8 (triple generation requirements and contracts)
        - `chat/run_triple.py` lines 699-750 (main function)
        - JSON schema validation logic in original script
        - Text chunking algorithms
        - `spec.md` Section 2 (current state assessment of run_triple.py)
- **Deliverables**:
    - [ ] `streamlit_pipeline/core/triple_generator.py` (~200-250 lines)
    - [ ] `TripleResult` and `Triple` data models
    - [ ] Text processing utilities for chunking
    - [ ] Schema validation integration
    - [ ] Unit tests in `test_triple_generator.py` (following `docs/Testing_Demands.md` TDD principles)
- **Dependencies**: REF-001, REF-002 (entity processor and data models)
- **Constraints**:
    - Must maintain JSON schema validation capabilities
    - Support text chunking for large inputs
    - Target significant complexity reduction
- **Completion Status**: ❌ Not Started
- **Notes**: Complex module requiring careful analysis

### [ ] **Task ID**: REF-005
- **Task Name**: Create Unified API Client Wrapper
- **Work Description**:
    - **Why**: Multiple modules need consistent API interactions with OpenAI/Perplexity APIs. Need a simplified wrapper that handles common patterns (authentication, rate limiting, error handling) without the complexity of the original scripts.
    - **How**:
        1. **First, read `spec.md` Section 7, 10** to understand API usage patterns and error handling requirements
        2. Create clean API client interface using litellm following spec.md guidelines
        3. Implement basic rate limiting and error handling per spec.md Section 10
        4. Add support for different model types (GPT-5-mini, Perplexity) as specified in spec.md Section 7
        5. Provide consistent response formatting meeting spec.md module contracts
- **Resources Required**:
    - **Materials**: litellm documentation, API documentation for OpenAI/Perplexity
    - **Personnel**: 1 Developer (estimated 2-3 days)
    - **Reference Codes**:
        - **Primary**: `streamlit_pipeline/docs/spec.md` Section 7 (API usage) and Section 10 (error handling)
        - `chat/openai_config.py` for API configuration patterns
        - Rate limiting logic from original scripts
        - `spec.md` Section 8 (module contracts for consistent interfaces)
- **Deliverables**:
    - [ ] `streamlit_pipeline/utils/api_client.py`
    - [ ] Rate limiting implementation
    - [ ] Error handling and retry logic
    - [ ] API client unit tests (following `docs/Testing_Demands.md` TDD principles)
- **Dependencies**: REF-002 (configuration system)
- **Constraints**:
    - Must support both OpenAI and Perplexity APIs
    - Simplified compared to original complex retry mechanisms
- **Completion Status**: ❌ Not Started
- **Notes**: Shared component used by all pipeline stages

---

## Phase 3: Graph Judge Simplification

### [ ] **Task ID**: REF-006
- **Task Name**: Extract Core Graph Judge Logic from run_gj.py
- **Work Description**:
    - **Why**: The `run_gj.py` script (~2200 lines) is the most complex component with multi-API support, gold label bootstrapping, and explainable reasoning. Need to extract core graph judgment functionality while dramatically reducing complexity.
    - **How**:
        1. **First, read `spec.md` Section 2, 3, 8** to understand graph judge refactoring strategy and requirements
        2. Analyze massive `run_gj.py` script to identify essential logic following spec.md guidelines
        3. Focus on Perplexity API integration initially per spec.md Section 3 (FR-GJ2)
        4. Extract core graph judgment algorithms as specified in spec.md Section 3 (FR-GJ3)
        5. Implement basic explainable reasoning mode per spec.md Section 3 (FR-GJ5)
        6. Defer complex multi-API support for future iterations following spec.md strategy
- **Resources Required**:
    - **Materials**: Access to `chat/run_gj.py`, Perplexity API documentation
    - **Personnel**: 1 Senior Developer (estimated 4-5 days)
    - **Reference Codes**:
        - **Primary**: `streamlit_pipeline/docs/spec.md` Sections 2, 3, 8 (graph judge requirements and refactoring strategy)
        - `chat/run_gj.py` lines 2105-2200 (main execution functions)
        - Graph judgment logic and reasoning algorithms
        - Perplexity API integration patterns
        - `spec.md` Section 2 (current state assessment of run_gj.py complexity)
- **Deliverables**:
    - [ ] `streamlit_pipeline/core/graph_judge.py` (~300-400 lines)
    - [ ] `JudgmentResult` data model with confidence scores
    - [ ] Basic explainable reasoning implementation
    - [ ] Perplexity API integration
    - [ ] Unit tests in `test_graph_judge.py` (following `docs/Testing_Demands.md` TDD principles)
- **Dependencies**: REF-004, REF-005 (triple generator and API client)
- **Constraints**:
    - Target 85%+ complexity reduction (2200+ → 300-400 lines)
    - Must maintain core judgment accuracy
    - Defer complex features like gold label bootstrapping
- **Completion Status**: ❌ Not Started
- **Notes**: Most challenging refactoring task due to original complexity

### [ ] **Task ID**: REF-007
- **Task Name**: Implement Simplified Error Handling and Logging
- **Work Description**:
    - **Why**: Original scripts have complex logging systems with file outputs and intricate error handling. Need simplified approach suitable for Streamlit UI with user-friendly error reporting.
    - **How**:
        1. **First, read `spec.md` Section 8, 10, 12** to understand error handling strategy and observability requirements
        2. Design clean error handling strategy returning errors as data per spec.md Section 8
        3. Implement Streamlit-compatible logging following spec.md Section 12 guidelines
        4. Create user-friendly error messages as specified in spec.md Section 10
        5. Add progress indication mechanisms per spec.md Section 3 (FR-I2)
- **Resources Required**:
    - **Materials**: Streamlit documentation, Python logging best practices
    - **Personnel**: 1 Developer (estimated 2 days)
    - **Reference Codes**:
        - **Primary**: `streamlit_pipeline/docs/spec.md` Sections 8, 10, 12 (error handling and observability)
        - Original `TerminalLogger` classes for patterns
        - Streamlit error handling examples
        - `spec.md` Section 3 (FR-I2 for Streamlit-compatible error reporting)
- **Deliverables**:
    - [ ] Unified error handling approach across all modules
    - [ ] Streamlit-compatible progress indication
    - [ ] User-friendly error message system
    - [ ] Logging integration for debugging
- **Dependencies**: All previous tasks (REF-001 through REF-006)
- **Constraints**:
    - Must not throw exceptions, return errors as data
    - User-friendly messages for non-technical users
- **Completion Status**: ❌ Not Started
- **Notes**: Cross-cutting concern affecting all modules

---

## Phase 4: Streamlit Integration & Polish

### [ ] **Task ID**: REF-008
- **Task Name**: Develop Main Streamlit Application Interface
- **Work Description**:
    - **Why**: Need user-friendly web interface that orchestrates the three refactored pipeline stages with clear progress indication and result display.
    - **How**:
        1. **First, read `spec.md` Section 5, 6, 9** to understand user flows and system architecture
        2. Design clean Streamlit UI with text input and results display following spec.md Section 5 user flows
        3. Implement three-stage pipeline orchestration per spec.md Section 6 system architecture
        4. Add progress indicators and real-time feedback as specified in spec.md Section 9 state machines
        5. Create result visualization components meeting spec.md requirements
- **Resources Required**:
    - **Materials**: Streamlit documentation, UI/UX best practices
    - **Personnel**: 1 Developer with Streamlit experience (estimated 3-4 days)
    - **Reference Codes**:
        - **Primary**: `streamlit_pipeline/docs/spec.md` Sections 5, 6, 9 (user flows, architecture, state machines)
        - Streamlit examples and documentation
        - `spec.md` Section 16 (repository structure for UI components)
        - `spec.md` Section 17 (public interfaces per module)
- **Deliverables**:
    - [ ] `streamlit_pipeline/app.py` main application
    - [ ] `streamlit_pipeline/ui/components.py` for reusable UI elements
    - [ ] `streamlit_pipeline/ui/display.py` for result visualization
    - [ ] User experience documentation
- **Dependencies**: All core modules (REF-001, REF-004, REF-006)
- **Constraints**:
    - Must provide clear progress indication
    - Handle errors gracefully with user-friendly messages
- **Completion Status**: ❌ Not Started
- **Notes**: Final integration component

### [ ] **Task ID**: REF-009
- **Task Name**: Implement Session State Management for Data Flow
- **Work Description**:
    - **Why**: Need to manage intermediate results between pipeline stages using Streamlit's session state instead of file-based storage, enabling seamless data flow and user interaction.
    - **How**:
        1. **First, read `spec.md` Section 2, 8, 9** to understand session state requirements and data flow patterns
        2. Design session state schema for pipeline data following spec.md Section 8 data models
        3. Implement data persistence across Streamlit reruns per spec.md Section 2 target architecture
        4. Add state management utilities as specified in spec.md Section 3 (FR-I3)
        5. Handle state cleanup and reset functionality per spec.md Section 9 state machines
- **Resources Required**:
    - **Materials**: Streamlit session state documentation
    - **Personnel**: 1 Developer (estimated 2-3 days)
    - **Reference Codes**:
        - **Primary**: `streamlit_pipeline/docs/spec.md` Sections 2, 8, 9 (architecture, data models, state machines)
        - Streamlit session state examples
        - `spec.md` Section 3 (FR-I3 for session state management requirements)
        - `spec.md` Section 6 (data flow architecture)
- **Deliverables**:
    - [ ] Session state management system
    - [ ] Data persistence utilities
    - [ ] State cleanup and reset mechanisms
    - [ ] Session state debugging tools
- **Dependencies**: REF-008 (main Streamlit app)
- **Constraints**:
    - Must handle large intermediate results efficiently
    - Provide clear state reset mechanisms
- **Completion Status**: ❌ Not Started
- **Notes**: Critical for smooth user experience

### [ ] **Task ID**: REF-010
- **Task Name**: Comprehensive Testing and Documentation
- **Work Description**:
    - **Why**: Ensure all refactored components work together correctly with comprehensive integration tests and complete documentation for maintenance and future development.
    - **How**:
        1. **First, read `docs/Testing_Demands.md`** to understand comprehensive testing principles and quality assurance checklist
        2. **Then read `spec.md` Section 15, 19** to understand testing strategy and acceptance criteria
        3. Create end-to-end integration tests following spec.md Section 15 testing matrix and Testing_Demands.md integration testing requirements
        4. Develop comprehensive documentation per spec.md Section 19 success metrics and Testing_Demands.md documentation guidelines
        5. Set up automated testing pipeline meeting spec.md Section 19 technical requirements
        6. Create deployment and maintenance guides following spec.md Section 14
- **Resources Required**:
    - **Materials**: Testing frameworks, documentation tools
    - **Personnel**: 1 Developer (estimated 2-3 days)
    - **Reference Codes**:
        - **Primary**: `docs/Testing_Demands.md` (comprehensive testing principles and QA checklist)
        - `streamlit_pipeline/docs/spec.md` Sections 14, 15, 19 (deployment, testing, acceptance criteria)
        - Existing test patterns from `chat/unit_test/`
        - Documentation standards from project
        - `spec.md` Section 11 (performance & reliability requirements for benchmarking)
- **Deliverables**:
    - [ ] Complete integration test suite (following `docs/Testing_Demands.md` integration testing guidelines)
    - [ ] API documentation for all modules
    - [ ] Deployment and maintenance documentation
    - [ ] Performance benchmarking results
    - [ ] Quality assurance checklist completion (per `docs/Testing_Demands.md`)
- **Dependencies**: All previous tasks (complete system)
- **Constraints**:
    - Must achieve 90%+ test coverage
    - Documentation must be maintainable
- **Completion Status**: ❌ Not Started
- **Notes**: Quality assurance and project completion

---

## Project Overview and Constraints

### Global Dependencies
- **API Access**: Valid OpenAI and Perplexity API keys
- **Development Environment**: Python 3.8+, Git access, testing frameworks
- **Reference Materials**: Access to original `chat/` scripts for analysis

### Success Criteria
- **Code Reduction**: 70%+ reduction in lines of code per module
- **Test Coverage**: 90%+ unit test coverage for all refactored modules
- **Functionality Parity**: All essential NLP capabilities preserved
- **User Experience**: Simplified execution compared to CLI scripts

### Risk Mitigation
- **R-01**: Complex refactoring → Incremental approach with frequent testing
- **R-02**: API integration issues → Comprehensive mocking and error handling  
- **R-03**: Performance degradation → Regular benchmarking against original scripts

### Timeline Summary
- **Total Duration**: 4 weeks (20 working days)
- **Phase 1**: 5 days (Core module extraction)
- **Phase 2**: 6 days (Triple generation refactoring)
- **Phase 3**: 6 days (Graph judge simplification)  
- **Phase 4**: 5 days (Streamlit integration)

---

**Last Updated**: 2025-09-12  
**Next Review**: Weekly progress review meetings  
**Contact**: Development Team Lead