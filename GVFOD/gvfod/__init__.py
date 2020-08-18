try:
    # raise ImportError()
    from gvfod.clearn import clearn as flearn
    from gvfod.clearn import clearn_ude as flearn_ude
except ImportError:
    from .learn import learn as flearn
    from .learn import learn_ude as flearn_ude

from .GVFOD import GVFOD
from .GVFOD import OGVFOD
