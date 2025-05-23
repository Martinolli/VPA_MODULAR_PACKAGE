# VPA Logging Documentation

## Overview

The VPA system uses a standardized logging framework provided by the `vpa_logger.py` module. This document explains how to configure and use logging in both development and production environments.

## Basic Usage

### Getting a Logger

The recommended way to get a logger is to use the `get_logger` function:

```python
from vpa_modular.vpa_logger import get_logger

# Get a logger for your module
logger = get_logger(module_name="MyModule")

# Use the logger
logger.info("This is an info message")
logger.error("This is an error message")
```

### Logging Levels

The VPA logger supports the standard Python logging levels:

- `DEBUG`: Detailed information, typically useful only for diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: Indication that something unexpected happened, but the application still works
- `ERROR`: Due to a more serious problem, the application has not been able to perform a function
- `CRITICAL`: A serious error, indicating that the application itself may be unable to continue running

### Specialized Logging Methods

The VPA logger provides specialized methods for common VPA operations:

```python
# Log analysis start
logger.log_analysis_start(ticker="AAPL", timeframes=[{"interval": "1d"}])

# Log analysis completion
logger.log_analysis_complete(ticker="AAPL", signal={"type": "BUY", "strength": "STRONG"})

# Log error during analysis
logger.log_error(ticker="AAPL", error="Failed to retrieve data")

# Log data retrieval status
logger.log_data_retrieval(ticker="AAPL", timeframe="1d", success=True)

# Log pattern detection
logger.log_pattern_detection(ticker="AAPL", pattern="Accumulation", detected=True)

# Log performance metrics
from datetime import datetime
start_time = datetime.now()
# ... perform operation ...
logger.log_performance(operation="Data processing", start_time=start_time)
```

## Configuration

### Development Environment

For development, it's recommended to use more verbose logging:

```python
logger = get_logger(
    module_name="MyModule",
    log_level="DEBUG",
    log_file="logs/development.log"
)
```

### Production Environment

For production, use a more restrictive logging level and enable log rotation:

```python
logger = get_logger(
    module_name="MyModule",
    log_level="INFO",
    log_file="/var/log/vpa/production.log",
    enable_rotation=True
)
```

### Log File Location

By default, logs are stored in `~/.vpa/logs/` with filenames based on the module name. You can specify a custom log file path if needed.

### Log Rotation

Log rotation is enabled by default with these settings:

- Maximum file size: 10MB
- Number of backup files: 5

You can customize these settings when creating a logger:

```python
from vpa_modular.vpa_logger import VPALogger

logger = VPALogger(
    module_name="MyModule",
    log_level="INFO",
    log_file="logs/custom.log",
    enable_rotation=True,
    max_bytes=5242880,  # 5MB
    backup_count=10
)
```

## Best Practices

1. **Use Appropriate Log Levels**: Reserve ERROR and CRITICAL for actual errors, use INFO for normal operations, and DEBUG for detailed diagnostics.

2. **Include Contextual Information**: Always include relevant context in log messages (e.g., ticker symbols, timeframes).

3. **Log Exceptions with Tracebacks**: Use `logger.exception()` to log exceptions with full tracebacks.

4. **Performance Logging**: Use `log_performance()` to track the performance of critical operations.

5. **Standardized Module Names**: Use consistent module names across the application to make log filtering easier.

## Viewing Logs

### Console Output

All logs are output to the console by default, making them visible in the terminal during development.

### Log Files

Log files are stored in the specified location or in the default directory (`~/.vpa/logs/`). You can view them using standard tools:

```bash
# View the last 100 lines of a log file
tail -n 100 ~/.vpa/logs/vpa.log

# Follow a log file in real-time
tail -f ~/.vpa/logs/vpa.log

# Search for specific patterns
grep "ERROR" ~/.vpa/logs/vpa.log
```

## Troubleshooting

### No Logs Appearing

1. Check that the log level is appropriate (e.g., DEBUG messages won't appear if the log level is set to INFO)
2. Verify that the log directory exists and is writable
3. Check for permission issues if using system directories like `/var/log`

### Too Many Logs

If logs are growing too quickly:

1. Increase the log level (e.g., from DEBUG to INFO)
2. Decrease the max_bytes or increase the backup_count for log rotation
3. Be more selective about what is logged at INFO level
