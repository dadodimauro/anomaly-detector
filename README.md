anomaly-detector
==============================

A python anomaly detection software, used to find anomalies for multivariate timeseries data in an unsupervised scenario

Project Organization
------------

    anomaly-detector/
    |
    ├── LICENSE
    ├── Makefile                                    <- Makefile with commands like `make data` or `make train`
    ├── README.md                                   <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                               <- The final, canonical data sets for modeling.
    │   │   ├── filtered
    │   │   │   │   ├── clean_df-checkpoint.csv
    │   │   │   │   └── dirty_df-checkpoint.csv
    │   │   │   ├── clean_df.csv
    │   │   │   └── dirty_df.csv
    │   │   └── labels
    │   └── raw                                     <- The original data
    │       ├── V_GV_SYSMETRIC_INCTANCE_1.csv
    │       └── V_GV_SYSMETRIC_INCTANCE_2.csv
    |
    ├── docs                                        <- A default Sphinx project; see sphinx-doc.org for details
    │   
    ├── main.py
    ├── models                                      <- Trained and serialized models, model predictions, or model summaries
    │   └── model.pth
    ├── notebooks                                   <- Jupyter notebooks
    │   │   ├── PyOD-checkpoint.ipynb
    │   │   ├── USAD-checkpoint.ipynb
    │   │   ├── correlation-report-checkpoint.ipynb
    │   │   ├── multivariate_ad-checkpoint.ipynb
    │   │   ├── timeseries-checkpoint.ipynb
    │   │   └── univariate_ad-checkpoint.ipynb
    │   ├── PyOD.ipynb
    │   ├── USAD.ipynb
    │   ├── correlation-report.ipynb
    │   ├── multivariate_ad.ipynb
    │   ├── timeseries.ipynb
    │   └── univariate_ad.ipynb
    ├── paperspace-requirements.txt                 <- The requirements file for reproducing the analysis environment, e.g.
    │                                                  generated with `pip freeze > requirements.txt`
    ├── references                                  <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports                                     <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── anomalies-intervals                     <- File containing the timestamp of the anomalies identified.
    │   │   │   ├── anomalies-intervals-checkpoint.csv
    │   │   │   └── anomalies-intervals-checkpoint.txt
    │   │   ├── anomalies-intervals-multivariate.csv
    │   │   ├── anomalies-intervals-multivariate.txt
    │   │   ├── anomalies-intervals-univariate.csv
    │   │   ├── anomalies-intervals-univariate.txt
    │   │   ├── anomalies-intervals.csv
    │   │   └── anomalies-intervals.txt
    │   ├── correlation-analysis
    │   │   │   └── report_timeseries-checkpoint.html
    │   │   └── report_timeseries.html
    │   ├── figures                                 <- Generated graphics and figures to be used in reporting.
    │   │   ├── db_time
    │   │   │   └── static
    │   │   │       └── db_time.svg
    │   │   ├── history
    │   │   │   └── static
    │   │   │       └── history.svg
    │   │   ├── labels
    │   │   │   └── static
    │   │   │       │   └── labels-checkpoint.svg
    │   │   │       └── labels.svg
    │   │   └── thresholds
    │   │       ├── dynamic
    │   │       │   │   └── thresholds-checkpoint.html
    │   │       │   └── thresholds.html
    │   │       └── static
    │   │           └── thresholds.svg
    │   └── timeseries
    │       │   └── decomposition-checkpoint.svg
    │       ├── decomposition.svg
    │       ├── multivariate_ts.svg
    │       └── univariate_ts.svg
    ├── requirements.txt                            <- The requirements file
    ├── setup.py                                    <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                                         <- Source code for use in this project.
    │   ├── __init__.py                             <- Makes src a Python module
    │   ├── data                                    <- Scripts to preprocess or select data
    │   │   ├── __init__.py
    │   │   ├── columns.py
    │   │   ├── make_dataset.py
    │   │   └── preprocessing.py
    │   ├── features                                <- Scripts to turn raw data into features for modeling
    │   │   ├── __init__.py
    │   │   └── build_features.py
    │   ├── models                                  <- Scripts to define models and their utiliy functions
    │   │   ├── .gitkeep
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   ├── usad.py
    │   │   └── usad_utils.py
    │   ├── multivariate_ad.py                      <- Script of the multivariate anomaly detector
    │   ├── params.yaml                             <- File in which are specified the parameters of the application
    │   ├── univariate_ad.py                        <- Script of the univariate anomaly detector
    │   ├── utils                                   <- General utily functions
    │   │   ├── __init__.py
    │   │   ├── params.py                           
    │   │   ├── parser.py
    │   │   ├── thresholds.py
    │   │   └── utils.py
    │   └── visualization                           <- Scripts to create exploratory and results oriented visualizations
    │       ├── __init__.py
    │       ├── plotter.py
    │       └── visualize.py
    ├── test_environment.py
    ├── tox.ini
    └── tree-structure.txt

--------
