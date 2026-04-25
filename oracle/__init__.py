from oracle.engine import OracleEngine, DiagnosticResult

try:
    from oracle.session import run_interactive
except ImportError:
    pass
