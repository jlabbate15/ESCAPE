"""Public entry point: the ESCAPE class (Saarelma-Connor + EPEDNN).
"""

from src.saarelma_connor.saarelma_connor_api import saarelma_connor
from src.epednn.epednn_call import epednn_class


class ESCAPE(saarelma_connor, epednn_class):
    """
    Built once through the shared ``__init__`` of
    :class:`~src.saarelma_connor.saarelma_connor_base.SaarelmaConnorBase`; the Saarelma-Connor physics model is selected
    per call via :meth:`~src.saarelma_connor.saarelma_connor_base.SaarelmaConnorBase.solve`.

    The mixin order is arbitrary -- the three classes share no attribute
    names, so the MRO never has to choose between them.
    """

__all__ = ['ESCAPE']