import unittest
import os

def run_all_tests():
    # Define the path to the tests directory
    tests_directory = os.path.join(os.path.dirname(__file__), 'tests')

    # Discover all test cases in the tests directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=tests_directory)

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit code based on test results
    if not result.wasSuccessful():
        exit(1)  # Non-zero exit code indicates test failure

if __name__ == "__main__":
    run_all_tests()
