from .vector_compression import *
from .jacobian_compression import *
from .sampling_scheduler import *
from typing import Dict


def get_jac_compression_operator(jac_compression_config: Dict):
    compression_function = jac_compression_config.get("rule", 'full')
    if compression_function in ['active_norm_sampling',
                                'random_sampling']:
        return SparseApproxMatrix(conf=jac_compression_config)

    return None
