try:
    # raise ImportError()
    from .clearn import clearn as flearn
    from .clearn import clearn_ude as flearn_ude
except ImportError:
    # raise ImportError
    from .learn import learn as flearn
    from .learn import learn_ude as flearn_ude

from gvfod.GVFOD import GVFOD
from gvfod.GVFOD import UDE
