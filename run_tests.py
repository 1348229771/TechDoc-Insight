import unittest
import sys
import os

# Set environment variable to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_tests():
    """Run all tests in the tests directory."""
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    print(f"Project root: {project_root}")
    
    try:
        from tests.test_metrics import TestSummarizationMetrics
        from tests.test_data_loader import TestDataLoader
        from tests.test_pipeline import TestPipeline
        
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSummarizationMetrics))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataLoader))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPipeline))
        
        print(f"Found {suite.countTestCases()} tests.")
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result
    except ImportError as e:
        print(f"ImportError: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    print("Running tests...")
    result = run_tests()
    
    if result and result.wasSuccessful():
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed or error occurred.")
        sys.exit(1)
