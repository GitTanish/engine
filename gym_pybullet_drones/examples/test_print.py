import sys
print("Hello stdout", flush=True)
print("Hello stderr", file=sys.stderr, flush=True)
