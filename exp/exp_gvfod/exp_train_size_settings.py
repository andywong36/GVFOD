import importlib
from collections import namedtuple

import hyperopt
import numpy as np
from hyperopt import hp as hp
from sklearn import decomposition as decomp

from exp_gvfod.tuning_utils import factors

Experiment = namedtuple("Experiment",
                        ["clf", "clfname", "use_pca", "use_scaling", "kwargs"])


def save_kwargs(**kwargs):
    return kwargs


contamination = {"contamination": 0.05}

if_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.iforest"), "IForest"),
    clfname="IForest",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_estimators=14,
                       max_features=0.8822204001089627,
                       bootstrap=True,
                       **contamination
                       ),
)

ocsvm_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.ocsvm"), "OCSVM"),
    clfname="OCSVM",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(kernel="sigmoid",
                       nu=0.9961484815664832,
                       coef0=0.10324230397111303,
                       gamma=4.2802822010040366e-05,
                       **contamination)
)

lof_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.lof"), "LOF"),
    clfname="LOF",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(n_neighbors=500,
                       leaf_size=87,
                       metric="chebyshev",
                       **contamination),
)

abod_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.abod"), "ABOD"),
    clfname="ABOD",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(n_neighbors=92,
                       **contamination),
)

kNN_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.knn"), "KNN"),
    clfname="KNN",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(n_neighbors=500,
                       method="largest",
                       radius=0.9936009435553693,
                       metric="chebyshev",
                       **contamination)
)

hbos_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.hbos"), "HBOS"),
    clfname="HBOS",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_bins=10,
                       alpha=0.827288822866795,
                       tol=0.7535474454670121,
                       **contamination),
)

mcd_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.mcd"), "MCD"),
    clfname="MCD",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(support_fraction=0.7138341918585475,
                       **contamination)
)

pca_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.pca"), "PCA"),
    clfname="PCA",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_components=3,
                       weighted=0,
                       whiten=1,
                       **contamination)
)

markov_exp = Experiment(
    clf=getattr(importlib.import_module("markov.markov"), "Markov"),
    clfname="MarkovChain",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_sensors=3,
                       divisions=8,
                       resample=True,
                       sample_period=1,
                       **contamination)
)

gvfod_exp = Experiment(
    clf=getattr(importlib.import_module("gvfod"), "GVFOD"),
    clfname="GVFOD",
    use_pca=False,
    use_scaling=False,
    kwargs=save_kwargs(space=[[10, 180],  # Angle limits
                              [-1, 1],  # Torque limits
                              [0, 300]],
                       divs_per_dim=[7, 7, 2],
                       wrap_idxs=None,
                       int_idxs=None,
                       numtilings=2,
                       discount_rate=0.9587470519252619,
                       learn_rate=0.10439012333499052,
                       lamda=0.20893365640615555,
                       beta=888,
                       **contamination)
)
