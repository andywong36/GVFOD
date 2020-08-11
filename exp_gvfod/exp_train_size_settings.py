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
    kwargs=save_kwargs(n_estimators=11,
                       max_features=0.8673163424581232,
                       bootstrap=False,
                       **contamination
                       ),
)

ocsvm_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.ocsvm"), "OCSVM"),
    clfname="OCSVM",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(kernel="sigmoid",
                       nu=0.998658823680417,
                       coef0=9.225351804663987,
                       gamma=0.0015828238935596535,
                       **contamination)
)

lof_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.lof"), "LOF"),
    clfname="LOF",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(leaf_size=58,
                       metric="chebyshev",
                       **contamination),
)

abod_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.abod"), "ABOD"),
    clfname="ABOD",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_neighbors=98,
                       **contamination),
)

kNN_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.knn"), "KNN"),
    clfname="KNN",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(n_neighbors=500,
                       method="largest",
                       radius=0.0096124,
                       metric="chebyshev",
                       **contamination)
)

hbos_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.hbos"), "HBOS"),
    clfname="HBOS",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_bins=106,
                       alpha=0.9828624725446773,
                       tol=0.8225734839484481,
                       **contamination),
)

mcd_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.mcd"), "MCD"),
    clfname="MCD",
    use_pca=True,
    use_scaling=True,
    kwargs=save_kwargs(support_fraction=0.58249,
                       **contamination)
)

pca_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.pca"), "PCA"),
    clfname="PCA",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_components=49,
                       weighted=1,
                       whiten=1,
                       **contamination)
)

markov_exp = Experiment(
    clf=getattr(importlib.import_module("markov.markov"), "Markov"),
    clfname="MarkovChain",
    use_pca=False,
    use_scaling=True,
    kwargs=save_kwargs(n_sensors=3,
                       divisions=18,
                       resample=True,
                       sample_period=2,
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
                       divs_per_dim=[3, 5, 3],
                       wrap_idxs=None,
                       int_idxs=None,
                       numtilings=5,
                       discount_rate= 0.9787171325363924,
                       learn_rate=0.4821972075639546,
                       lamda=0.2573599546759918,
                       beta=334,
                       **contamination)
)
