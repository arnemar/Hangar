import sys
from pathlib import Path

# Ensure project root is importable regardless of how pytest is invoked
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
