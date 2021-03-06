.PHONY: clean lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = palmnet
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: svhn cifar100 cifar10 mnist

svhn: data/external/svhn.npz
cifar100: data/external/cifar100.npz
cifar10: data/external/cifar10.npz
mnist: data/external/mnist.npz

data/external/svhn.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py svhn data/external
data/external/cifar100.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py cifar100 data/external
data/external/cifar10.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py cifar10 data/external
data/external/mnist.npz:
	$(PYTHON_INTERPRETER) code/data/make_dataset.py mnist data/external

## Make Model

models: mnist_lenet cifar10_vgg19_4096x4096 cifar100_vgg19_4096x4096 svhn_vgg19_4096x4096 cifar10_vgg19_2048x2048 cifar100_vgg19_2048x2048 svhn_vgg19_2048x2048 mnist_500 cifar100_resnet20 cifar100_resnet50 cifar10_tensortrain_base cifar100_resnet50_new cifar100_resnet20_new

mnist_lenet: models/external/mnist_lenet_1570207294.h5 models/external/mnist_lenet_1_1586853546.h5  models/external/mnist_lenet_2_1586853549.h5 models/external/mnist_lenet_3_1586854101.h5

cifar100_resnet20: models/external/resnet_20_cifar100.h5

cifar100_resnet50: models/external/resnet_50_cifar100.h5

cifar100_resnet50_new: models/external/resnet_resnet50_cifar100_1587927534.h5 models/external/resnet_resnet50_cifar100_1_1588162548.h5 models/external/resnet_resnet50_cifar100_2_1588107732.h5 models/external/resnet_resnet50_cifar100_3_1588102661.h5

cifar100_resnet20_new: models/external/resnet_resnet20_cifar100_1588012286.h5 models/external/resnet_resnet20_cifar100_1_1588096045.h5 models/external/resnet_resnet20_cifar100_2_1588101554.h5 models/external/resnet_resnet20_cifar100_3_1588090286.h5

cifar10_vgg19_4096x4096: models/external/cifar10_vgg19_4096x4096_1570693209.h5

cifar100_vgg19_4096x4096: models/external/cifar100_vgg19_4096x4096_1570789868.h5

svhn_vgg19_4096x4096: models/external/svhn_vgg19_4096x4096_1570786657.h5

cifar10_vgg19_2048x2048: models/external/cifar10_vgg19_2048x2048_1572303047.h5 models/external/cifar10_vgg19_2048x2048_1_1586857195.h5 models/external/cifar10_vgg19_2048x2048_2_1586849939.h5 models/external/cifar10_vgg19_2048x2048_3_1586849939.h5

cifar100_vgg19_2048x2048: models/external/cifar100_vgg19_2048x2048_1572278802.h5 models/external/cifar100_vgg19_2048x2048_1_1586850015.h5 models/external/cifar100_vgg19_2048x2048_2_1586850015.h5 models/external/cifar100_vgg19_2048x2048_3_1586850015.h5

svhn_vgg19_2048x2048: models/external/svhn_vgg19_2048x2048_1572278831.h5 models/external/svhn_vgg19_2048x2048_1_1586873524.h5 models/external/svhn_vgg19_2048x2048_2_1586877914.h5 models/external/svhn_vgg19_2048x2048_3_1586878915.h5

mnist_500: models/external/mnist_500.h5

cifar10_tensortrain_base: models/external/cifar10_tensor_train_base_1585409008.h5

models/external/cifar10_tensor_train_base_1585409008.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar10_tensor_train_base models/external

models/external/resnet_20_cifar100.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet20 models/external

models/external/resnet_50_cifar100.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet50 models/external

models/external/resnet_resnet50_cifar100_1587927534.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet50_new models/external

models/external/resnet_resnet50_cifar100_1_1588162548.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet50_new-1 models/external

models/external/resnet_resnet50_cifar100_2_1588107732.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet50_new-2 models/external

models/external/resnet_resnet50_cifar100_3_1588102661.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet50_new-3 models/external

models/external/resnet_resnet20_cifar100_1588012286.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet20_new models/external

models/external/resnet_resnet20_cifar100_1_1588096045.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet20_new-1 models/external

models/external/resnet_resnet20_cifar100_2_1588101554.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet20_new-2 models/external

models/external/resnet_resnet20_cifar100_3_1588090286.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_resnet20_new-3 models/external

models/external/mnist_lenet_1570207294.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py mnist_lenet models/external

models/external/mnist_lenet_1_1586853546.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py mnist_lenet-1 models/external

models/external/mnist_lenet_2_1586853549.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py mnist_lenet-2 models/external

models/external/mnist_lenet_3_1586854101.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py mnist_lenet-3 models/external

models/external/mnist_500.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py mnist_500 models/external

# vgg19 4096x4096
models/external/cifar10_vgg19_4096x4096_1570693209.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar10_vgg19_4096x4096 models/external

models/external/cifar100_vgg19_4096x4096_1570789868.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_vgg19_4096x4096 models/external

models/external/svhn_vgg19_4096x4096_1570786657.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py svhn_vgg19_4096x4096 models/external

# vgg19 2048x2048
models/external/cifar10_vgg19_2048x2048_1572303047.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar10_vgg19_2048x2048 models/external

models/external/cifar10_vgg19_2048x2048_1_1586857195.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar10_vgg19_2048x2048-1 models/external

models/external/cifar10_vgg19_2048x2048_2_1586849939.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar10_vgg19_2048x2048-2 models/external

models/external/cifar10_vgg19_2048x2048_3_1586849939.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar10_vgg19_2048x2048-3 models/external

models/external/cifar100_vgg19_2048x2048_1572278802.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_vgg19_2048x2048 models/external

models/external/cifar100_vgg19_2048x2048_1_1586850015.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_vgg19_2048x2048-1 models/external

models/external/cifar100_vgg19_2048x2048_2_1586850015.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_vgg19_2048x2048-2 models/external

models/external/cifar100_vgg19_2048x2048_3_1586850015.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py cifar100_vgg19_2048x2048-3 models/external

models/external/svhn_vgg19_2048x2048_1572278831.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py svhn_vgg19_2048x2048 models/external

models/external/svhn_vgg19_2048x2048_1_1586873524.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py svhn_vgg19_2048x2048-1 models/external

models/external/svhn_vgg19_2048x2048_2_1586877914.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py svhn_vgg19_2048x2048-2 models/external

models/external/svhn_vgg19_2048x2048_3_1586878915.h5:
	$(PYTHON_INTERPRETER) code/models/make_models.py svhn_vgg19_2048x2048-3 models/external

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf data/external/*
	rm -rf models/external/*

## Lint using flake8
lint:
	flake8 code

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
