"""Request-aware route optimizer for Petals (MO809A final project).

Extends Petals' client-side routing (`RemoteSequenceManager`) to decide, per
request, which model tier and which block replicas/servers to use based on a
declared objective (latency-sensitive vs quality-sensitive).
"""

from router.objectives import Objective, current_objective, parse_objective
from router.policy import RequestAwareSequenceManager
from router.quality import QualityTable
from router.selector import ModelSelector

__all__ = [
    "Objective",
    "current_objective",
    "parse_objective",
    "RequestAwareSequenceManager",
    "QualityTable",
    "ModelSelector",
]
