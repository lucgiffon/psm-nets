PSM-nets
========

Compressing Neural network architectures by replacing Conv and Dense layer with PSM layers: product of sparse matrices.

Notebook example
----------------

A notebook example is available in `code/notebooks/POC PSM layers with pyfaust library.ipynb`. This notebook should be self contained with software requirements directly specified in. If you want to understand how to make your own PSM-nets, this is the goto destination.

Reference paper
---------------

Giffon, L., Ayache, S., Kadri, H., Artières, T., & Sicre, R. (2021). PSM-nets: Compressing Neural Networks with Product of Sparse Matrices. In 2021 international joint conference on neural networks (IJCNN). IEEE.

Important Notes
---------------
* This code is provided as is. No particular effort have been made to make it easily readable and understandable for new users. Feel free to open an issue to ask any question on how to read/use our code.
It will be useful to others as well. I will answer shortly.
* Most of the code needs this other project: https://github.com/lucgiffon/qkmeans to be installed to work. (**Not the notebook example** which relies on a most modern and stable implementation of PALM4MSA, see the notebook directly in the `code/notebooks` folder)
* Base data and models can be generated using `make data` and `make models` from the root directory of the project
* To understand how to call the various utilities, you can have a look at the test directory. All the tests should pass.... in theory
* To understand the scripts for the experiments I recommend you take a look at `code/scripts/2021/01/14_15_compression_baselines.py` for the baselines, `code/scripts/2020/09/9_10_compression_sparse_facto_new.py` for the compression and `code/scripts/2020/11/13_15_finetune_sparsefacto.py` for the fine tuning.
* You can also look at the parameter file to understand how to call the scripts: for instance `parameters/2020/09/12_13_compression_baselines_magnitude_hard.txt` for the baselines, `parameters/2020/11/9_10_compression_palm_act_full_net_300.txt` for the compression and `parameters/2020/11/12_13_finetune_sparse_facto_palm_act.txt` for the finetuning.  


Project Organization
--------------------

"*" at the end of a name means the most important stuff is in there.

	├── LICENSE
	├── Makefile           <- Makefile with commands like `make data`
	├── README.md          <- The top-level README for developers using this project.
	├── data               <- Will store the datasets if a command like `make data` is used
	│	
	├── models             <- Will store the models if a command like `make models` is used
	│
	├── reports            
	│   └── figures        <- Generated graphics and figures to be used in reporting. Need to be generated by using scripts in folder vizualization
	│
	│
	├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── parameters               <- All the parameters that have been used for the various experiments. Each file correspond to one experiment. Have fun.
    |
	├── code               <- Source code for use in this project.
	│   │
	│   ├── data           <- Scripts to download or generate data
	│   │   └── make_dataset.py
    │   │
	│   ├── palmnet   <- Actual code for all techniques
	|   |   ├── core*  <- layer-replacing classes + code for creating compressed layers
	|   |   ├── experiments  <- utils for the experiments
	|   |   ├── layers*  <- special layers used in the experiments
    │   │   └── visualization <- utils for vizualization
    │   │
	│   ├── process_results   <- Scripts to process raw generated results
    │   │
    │   ├── scripts*   <- Scripts to run experiments  (check that to understand how to call our classes) 
    │   │
    │   ├── test*   <- Few tests for the core features (check that to have simple examples on how to use our code)
	│   │
	│   │
    │   ├── models         <- Scripts to download base models
	│   │   │                 
	│   │   └── make_models.py
	│   │
	│   └── visualization  <- Scripts to create exploratory and results oriented visualizations from processed results
	│
	└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
