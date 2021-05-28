import importlib
from collections import namedtuple

import hyperopt
import numpy as np
from hyperopt import hp as hp
from sklearn import decomposition as decomp

from .tuning_utils import factors

Experiment = namedtuple("ExperimentSettings",
                        ["clf", "clfname", "metric", "runs", "parameters"], )
Experiment.__new__.__defaults__ = (None, None, "f1", 400, None)

if_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.iforest"), "IForest"),
    clfname="IForest",
    parameters=hp.choice(
        'ctransform',
        [
            {"n_estimators": hyperopt.pyll.scope.int(hp.quniform("n_estimators_nopca", 10, 200, 1)),
             "contamination": 0.05,
             "max_features": hp.uniform("max_features_nopca", 0.1, 1.0),
             "n_jobs": 1,
             "bootstrap": hp.choice("bootstrap_nopca", [0, 1]),
             "transform": None},
            {"n_estimators": hyperopt.pyll.scope.int(hp.quniform("n_estimators_pca", 10, 200, 1)),
             "contamination": 0.05,
             "max_features": hp.uniform("max_features_pca", 0.1, 1.0),
             "n_jobs": 1,
             "bootstrap": hp.choice("bootstrap_pca", [0, 1]),
             "transform": decomp.PCA(n_components=20)}
        ]
    ),
)

ocsvm_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.ocsvm"), "OCSVM"),
    clfname="OCSVM",
    parameters=
    hp.choice(
        "ckernel",
        [
            {"kernel": "linear",
             "nu": hp.uniform("nu_linear", 0, 1),
             "contamination": 0.05,
             "transform": hp.choice(
                 "transform_linear",
                 [None, decomp.PCA(n_components=20)])},
            {"kernel": "rbf",
             "nu": hp.uniform("nu_rbf", 0, 1),
             "gamma": hp.choice("gamma_rbf",
                                ["auto",
                                 hp.lognormal("gamma_rbf_float", np.log(1 / 2000), 1)]),
             "contamination": 0.05,
             "transform": hp.choice(
                 "transform_rbf",
                 [None, decomp.PCA(n_components=20)])},
            {"kernel": "sigmoid",
             "nu": hp.uniform("nu_sigmoid", 0, 1),
             "coef0": hp.normal("coef0", 0, 1),
             "gamma": hp.choice("gamma_sigmoid",
                                ["auto",
                                 hp.lognormal("gamma_sigmoid_float", np.log(1 / 2000), 1)]),
             "contamination": 0.05,
             "transform": hp.choice(
                 "transform_sigmoid",
                 [None, decomp.PCA(n_components=20)])},
        ]
    )
)

lof_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.lof"), "LOF"),
    clfname="LOF",
    parameters=hp.choice(
        "ctransform",
        [
            {"n_neighbors": hyperopt.pyll.scope.int(hp.quniform("n_neighbors_nopca", 2, 500, 1)),
             "leaf_size": hyperopt.pyll.scope.int(hp.quniform("leaf_size_nopca", 10, 100, 1)),
             "metric": hp.choice("metric_nopca", ["chebyshev", "l1", "l2"]),
             "contamination": 0.05,
             "n_jobs": 1,
             "transform": None},
            {"n_neighbors": hyperopt.pyll.scope.int(hp.quniform("n_neighbors_pca", 2, 500, 1)),
             "leaf_size": hyperopt.pyll.scope.int(hp.quniform("leaf_size_pca", 10, 100, 1)),
             "metric": hp.choice("metric_pca", ["chebyshev", "l1", "l2"]),
             "contamination": 0.05,
             "n_jobs": 1,
             "transform": decomp.PCA(n_components=20)}
        ]

    )
)

abod_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.abod"), "ABOD"),
    clfname="ABOD",
    parameters=hp.choice(
        'ctransform',
        [
            {"contamination": 0.05,
             "n_neighbors": hyperopt.pyll.scope.int(hp.quniform("n_neighbors_nopca", 5, 100, 1)),
             "transform": None},
            {"contamination": 0.05,
             "n_neighbors": hyperopt.pyll.scope.int(hp.quniform("n_neighbors_pca", 5, 100, 1)),
             "transform": decomp.PCA(n_components=20)}
        ]
    )
)

kNN_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.knn"), "KNN"),
    clfname="KNN",
    parameters=hp.choice(
        "ctransform",
        [
            {
                "n_neighbors": hyperopt.pyll.scope.int(hp.quniform("n_neighbors_nopca", 2, 500, 1)),
                "method": hp.choice("method_nopca", ["largest", "mean", "median"]),
                "radius": hp.lognormal("radius_nopca", 0, 1),
                "metric": hp.choice("metric_nopca", ["chebyshev", "l1", "l2"]),
                "contamination": 0.05,
                "transform": None,
            },
            {
                "n_neighbors": hyperopt.pyll.scope.int(hp.quniform("n_neighbors_pca", 2, 500, 1)),
                "method": hp.choice("methodmetric_pca", ["largest", "mean", "median"]),
                "radius": hp.lognormal("radiusmetric_pca", 0, 1),
                "metric": hp.choice("metric_pca", ["chebyshev", "l1", "l2"]),
                "contamination": 0.05,
                "transform": decomp.PCA(n_components=20),
            },
        ]
    )
)

hbos_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.hbos"), "HBOS"),
    clfname="HBOS",
    parameters=hp.choice(
        "ctransform",
        [
            {
                "alpha": hp.uniform("alpha_nopca", 0, 1),
                "n_bins": hyperopt.pyll.scope.int(hp.quniform("n_bins_nopca", 10, 1000, 1)),
                "tol": hp.uniform("tol_nopca", 0, 1),
                "contamination": 0.05,
                "transform": None,
            },
            {
                "alpha": hp.uniform("alpha_pca", 0, 1),
                "n_bins": hyperopt.pyll.scope.int(hp.quniform("n_bins_pca", 10, 1000, 1)),
                "tol": hp.uniform("tol_pca", 0, 1),
                "contamination": 0.05,
                "transform": decomp.PCA(n_components=20),
            }
        ]
    )
)

mcd_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.mcd"), "MCD"),
    clfname="MCD",
    parameters={"support_fraction": hp.uniform("support_fraction_pca", 0.01, 0.99),
                "contamination": 0.05,
                "transform": decomp.PCA(n_components=20)}
)

pca_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.pca"), "PCA"),
    clfname="PCA",
    parameters=
    {"contamination": 0.05,
     "n_components": hyperopt.pyll.scope.int(hp.quniform("n_components", 2, 50, 1)),
     "whiten": hp.choice("whiten", [False, True]),
     "weighted": hp.choice("weighted", [False, True]),
     "transform": None
     },
)

markov_exp = Experiment(
    clf=getattr(importlib.import_module("markov.markov"), "Markov"),
    clfname="MarkovChain",
    metric="f1",
    parameters={
        "n_sensors": 3,
        "contamination": 0.05,
        "divisions": hyperopt.pyll.scope.int(hp.quniform("divisions", 5, 100, 2)),
        "resample": True,
        "sample_period": hp.choice("sample_period", factors(2000)),
        "transform": None,
    }
)

hmm_exp = Experiment(
    clf=getattr(importlib.import_module("hmm"), "HMM"),
    clfname = "HMM",
    metric="f1",
    parameters={
        "n_sensors": 3,
        "n_states": hyperopt.pyll.scope.int(hp.choice("n_states"), (4, 8, 16, 32)),
        "contamination": 0.05,
    },
    runs=8,
)

gvfod_exp = Experiment(
    clf=getattr(importlib.import_module("gvfod"), "GVFOD"),
    clfname="GVFOD",
    metric="f1",
    parameters={
        "space": [[10, 180],  # Angle limits
                  [-1, 1],  # Torque limits
                  [0, 300]],  # Tension limits
        "divs_per_dim": [
            hyperopt.pyll.scope.int(hp.quniform("division0", 2, 10, 1)),
            hyperopt.pyll.scope.int(hp.quniform("division1", 2, 10, 1)),
            hyperopt.pyll.scope.int(hp.quniform("division2", 2, 10, 1)),
        ],
        "numtilings": hyperopt.pyll.scope.int(hp.choice("numtilings", (1, 2, 4, 8, 16, 32))),
        "discount_rate": hp.uniform("discount_rate", 0, 1),
        "learn_rate": hp.uniform("learn_rate", 0, 1),
        "lamda": hp.uniform("lamda", 0, 1),
        "beta": hyperopt.pyll.scope.int(hp.quniform("beta", 2, 1000, 2)),
        "transform": None,
        "scaling": False,
        "contamination": 0.05,
    }
)
