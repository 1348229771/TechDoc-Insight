# Testing Instructions

This directory contains unit tests and integration tests for the Automatic Abstract Generation System.

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running Tests

You can run the tests using the `unittest` framework:

```bash
python -m unittest discover tests
```

Or using the provided script (if available):

```bash
python run_tests.py
```

## Troubleshooting

If you encounter OpenMP errors (e.g., "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized"), try setting the following environment variable before running python:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

(The `run_tests.py` script attempts to set this automatically.)
