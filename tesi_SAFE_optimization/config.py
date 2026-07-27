"""
config.py
=========
Shared constants used across multiple pipeline steps: paths, annualisation
window, weight bounds, lambda grid, random seed. Pulled out here so that a
change (e.g. the lambda grid) only needs to happen in one place instead of
being edited in every step3/step5/step6/9a script separately.
"""

import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")

TRADING_DAYS = 252     # annualisation factor
ESTIM_WINDOW = 252     # trailing days for mu and Sigma estimation

W_MIN = 0.01    # minimum weight per selected stock
W_MAX = 0.20    # maximum weight per selected stock

LAMBDA_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

RANDOM_STATE = 42
