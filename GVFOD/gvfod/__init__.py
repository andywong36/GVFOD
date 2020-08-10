try:
    raise ImportError()
    from gvfod.clearn import clearn as flearn
except ImportError:
    from .learn import learn as flearn

from .GVFOD import GVFOD
