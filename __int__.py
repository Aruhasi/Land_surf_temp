import cdsapi
import sys
    c = cdsapi.Client()
with open(__main__.__file__) as f:
    code = f.read()