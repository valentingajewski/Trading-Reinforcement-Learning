"""Debug runner - captures all errors to a file."""
import sys
import traceback
import logging

# Write everything to a log file
log_file = open("results/debug_run.log", "w", buffering=1)

# Redirect all output
sys.stdout = log_file
sys.stderr = log_file

# Set up logging to the file
handler = logging.StreamHandler(log_file)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

try:
    sys.argv = ["run_trading.py", "--pair", "EURUSD", "--timesteps", "50000",
                "--max-folds", "3", "--device", "cuda"]
    from run_trading import main
    main()
    log_file.write("\n=== COMPLETED SUCCESSFULLY ===\n")
except Exception:
    traceback.print_exc(file=log_file)
    log_file.write("\n=== FAILED ===\n")
finally:
    log_file.flush()
    log_file.close()

# Signal to the console that we're done
real_stdout = sys.__stdout__
real_stdout.write("DEBUG RUN COMPLETE - check results/debug_run.log\n")
real_stdout.flush()
