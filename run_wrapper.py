"""Wrapper to run with explicit output flushing and logging to file."""
import sys
import logging

# Force all logging to stdout with flush
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

# Now import and run
from run_trading import main
main()
