"""
Simple test script to verify the Streamlit application can be imported and initialized.
"""

import sys
import os

# Add the parent directory to the path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Core modules
        from streamlit_pipeline.core.pipeline import PipelineOrchestrator, PipelineResult
        from streamlit_pipeline.core import config
        from streamlit_pipeline.core.models import EntityResult, TripleResult, JudgmentResult
        print("[OK] Core modules imported successfully")
        
        # UI modules
        from streamlit_pipeline.ui.components import display_input_section
        from streamlit_pipeline.ui.display import display_final_results
        from streamlit_pipeline.ui.error_display import PipelineProgressDisplay
        print("[OK] UI modules imported successfully")
        
        # Utils modules
        from streamlit_pipeline.utils.error_handling import ErrorHandler
        from streamlit_pipeline.utils.api_client import get_api_client
        print("[OK] Utils modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_app_initialization():
    """Test that the main app class can be initialized."""
    print("\nTesting app initialization...")
    
    try:
        # Mock streamlit session state
        import types
        mock_session_state = types.SimpleNamespace()
        mock_session_state.pipeline_results = []
        mock_session_state.current_result = None
        mock_session_state.processing = False
        mock_session_state.run_count = 0
        mock_session_state.config_options = {}
        
        # Import first
        from streamlit_pipeline.core.pipeline import PipelineOrchestrator
        from streamlit_pipeline.core import config
        
        # Test pipeline orchestrator
        orchestrator = PipelineOrchestrator()
        print("[OK] Pipeline orchestrator created successfully")
        
        # Test config
        model_config = config.get_model_config()
        print("[OK] Pipeline config loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] App initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing GraphJudge Streamlit Application")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test app initialization
    if not test_app_initialization():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed! The application should be ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)