# PyRAT compiled version for Python 3.9
<p align="center">
<a href="https://pyrat-analyzer.com"><img src="https://pyrat-analyzer.com/assets/images/pyrat-logo.jpg" width=150></a>
</p>

## License
The licence of PyRAT is proprietary CEA. For more information and contact details, please see the PyRAT [website](https://pyrat-analyzer.com/).

## Install

Install requirements using your favourite package manager / virtual environment.

We suggest conda / [mamba](https://github.com/mamba-org/mamba). You can use the provided yaml environment. You can
install mamba and create a new environment with all required packages and activate it by running the following (tested
on Ubuntu 20.04):

### Conda / mamba

- Using the existing yaml config:

```bash
conda install mamba
mamba env create -f pyrat_env.yml
conda activate pyrat
```

If you have a timeout error with the pip install you can increase pip default timeout
(```export PIP_DEFAULT_TIMEOUT=1000```) and try again.

- In an existing environment, updating with the existing yaml config

```bash
mamba update -f pyrat_env.yml
```

- Using requirements.txt

After creating a conda environment with python 3.9 you can use requirements.txt as well if the installation failed for some
reason with the .yml.

Conda / mamba might fail when directly using requirements.txt as some packages are only available on pip. The following
commands will install all available packages through mamba and use pip only if mamba failed.

```bash
while read requirement; do mamba install -c conda-forge --yes $requirement || pip install $requirement; done < requirements.txt
```

the equivalent function for Windows shell should be:

```shell
FOR /F "delims=~" %f in (requirements.txt) DO mamba install --yes "%f"  -c conda-forge|| pip install "%f"
```

- In an existing environment, installing everything:

You can also install all the necessary package directly with the following commands:

```bash
mamba install numpy onnx~=1.8.1 pyparsing~=2.4.7 psutil tqdm~=4.59.0 onnxruntime>=1.7.0 \
keras~=2.6.0 matplotlib~=3.3.4 networkx~=2.5.1 pytest~=6.2.4 pyvis~=0.1.9 gitpython~=3.1.18 \
pandas~=1.2.4  loguru~=0.5.3 pillow~=8.4.0 tensorflow configargparse -c conda-forge
mamba install pytorch torchvision cpuonly -c pytorch
pip install opencv-python~=4.5.1.48 scipy~=1.5.4
```

If you have a cuda compatible device and want to use it with PyRAT change the pytorch install to:

```bash
mamba install pytorch torchvision cudatoolkit -c pytorch
```

Note that the versions have been relaxed compared to the yml / requirements.txt. It should still work in theory but we
have not thoroughly tested the installation on all superior versions.

### Virtual env

Alternatively you can create a local environment with virtualenv and install packages with pip, with python 3.9:

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Docker version

You can try the Dockerfile to run PyRAT in a container.

In a terminal on the root of PyRAT folder you can build the image with:

```bash
docker build . -t pyrat_image
```

Run the image interactively:

```bash
docker run -it pyrat_image bash
```

You can then launch PyRAT as you would on your own machine using the following instructions. For example, you can run all tests:

```bash
python -m pytest -s tests
```

## Run PyRAT

### Generic Usage:

```
usage: pyrat.py [-h] [--config CONFIG] [-V] [--model_path MODEL_PATH] [--property_path PROPERTY_PATH] [--acas ACAS [ACAS ...]] [--domains DOMAINS [DOMAINS ...]] [--max_size MAX_SIZE]
                [--max_noise MAX_NOISE] [--by_layer [BY_LAYER]] [--max_backsub MAX_BACKSUB] [--allow_smaller_size [ALLOW_SMALLER_SIZE]] [--one_noise [ONE_NOISE]]
                [--nb_constraints NB_CONSTRAINTS] [--nb_quad_approx NB_QUAD_APPROX] [--max_hybz MAX_HYBZ] [--iterative_hybz [ITERATIVE_HYBZ]] [--step_hybz STEP_HYBZ]
                [--hybz_relu_method HYBZ_RELU_METHOD] [--intermediate_concr [INTERMEDIATE_CONCR]] [--solver [SOLVER]] [--nb_it NB_IT] [--split [SPLIT]] [--timeout TIMEOUT]
                [--add_suffix [ADD_SUFFIX]] [--verbose [VERBOSE]] [--nb_process NB_PROCESS] [--log_intermediate [LOG_INTERMEDIATE]] [--transpose [TRANSPOSE]] [--squeeze [SQUEEZE]]     
                [--device DEVICE] [--sound [SOUND]] [--library LIBRARY] [--conv_type CONV_TYPE] [--it_optim_alpha IT_OPTIM_ALPHA] [--write [WRITE]] [--write_limit WRITE_LIMIT]
                [--check CHECK] [--attack [ATTACK ...]] [--attack_it ATTACK_IT] [--pgd_steps PGD_STEPS] [--nb_random NB_RANDOM] [--infer_label [INFER_LABEL]] [--dataset DATASET]       
                [--nb_img NB_IMG] [--shuffle [SHUFFLE]] [--epsilon EPSILON] [--image IMAGE] [--true_label TRUE_LABEL] [--total_labels TOTAL_LABELS] [--grayscale [GRAYSCALE]]
                [--force_analysis [FORCE_ANALYSIS]] [--scorer SCORER] [--polarity_shift POLARITY_SHIFT] [--coef_axis COEF_AXIS] [--initial [INITIAL]] [--coef_weight COEF_WEIGHT]       
                [--booster BOOSTER] [--refined_split REFINED_SPLIT [REFINED_SPLIT ...]] [--booster_threshold BOOSTER_THRESHOLD] [--mode MODE] [--metric METRIC]
                [--progress [PROGRESS]] [--log_dir LOG_DIR] [--log_name LOG_NAME] [--log_counter LOG_COUNTER] [--log_date [LOG_DATE]] [--log_number LOG_NUMBER] [--header HEADER]       
                [--debug_file DEBUG_FILE] [--debug [DEBUG]] [--use_relative_path [USE_RELATIVE_PATH]] [--read READ] [--parsed_folder PARSED_FOLDER]
                [--parsing_process PARSING_PROCESS]

PyRAT Analysis

optional arguments:
  -h, --help            show this help message and exit

General options:
  --config CONFIG, -c CONFIG
                        Path to a config.ini file for the analysis, optional
  -V, --version         show program's version number and exit

Path/data options:
  --model_path MODEL_PATH, -mp MODEL_PATH
                        Path to the model to be analysed
  --property_path PROPERTY_PATH, -pp PROPERTY_PATH
                        Path to the property file to be analysed
  --acas ACAS [ACAS ...]
                        Three ints i, j, p indicating network and property for ACAS Xu benchmark: i, j, p will lead to analysis of ACAS network i_j on property p. i must be in [1,     
                        5], j in [1, 9] and p in [1, 10]

Domain options:
  --domains DOMAINS [DOMAINS ...]
                        List of domains to be used. Choices: 'zonotopes', 'poly', 'symbox'
  --max_size MAX_SIZE   Modulates the maximum number of coefficient in the PyRAT analysis arrays. -1 means max_size is equal to sys.maxsize. This brings a tradeoff between memory and  
                        time. (default: -1)
  --max_noise MAX_NOISE
                        Number of additional noise to keep when using zonotope domains. If set to -1 all noises will be kept, if set to 0 no new noises will be added.
  --by_layer [BY_LAYER]
                        Activates full backsubstitution when using the deeppoly domain. (default: True)
  --max_backsub MAX_BACKSUB
                        Maximum backsubstitution to do for the poly domain. E.g. if equals to 3, it will only backsubstitute from layer by layer from n to layer n - 4 and then
                        backsub directly to the inputs. If set to -1, all backsubstitution will be done. Default to -1.
  --allow_smaller_size [ALLOW_SMALLER_SIZE]
                        Allows the added coefficients for the zonotopes to be of smaller size than the specified max_size. This allows the sources of these coefficients to be kept.    
                        Otherwise, they are by default fused together to match max_size. This should be activated when Add or Concat layers are present in the network.
  --one_noise [ONE_NOISE]
                        For the zonotope domain only. Only creates a single noise symbol for the analysis. This means all the inputs will be considered to vary together (with
                        different intensity).
  --nb_constraints NB_CONSTRAINTS
                        For ConZonotopes, the maximum number of constraints to add. Set to -1 for sys.maxsize. Defaults to all constraints.
  --nb_quad_approx NB_QUAD_APPROX
                        For PolyZonotopes, the maximum number of quadratic approximation to use. Set to -1 for sys.maxsize. Defaults to 0.
  --max_hybz MAX_HYBZ   For HybZonotopes, the maximum number of hybrid zonotopes to use. Set to -1 for sys.maxsize.Defaults to all.
  --iterative_hybz [ITERATIVE_HYBZ]
                        For HybZonotopes, whether to increment the number of HZ until it reaches max_hybz doing an analysis each time or directly using all HZ. Defauts to all HZ       
                        (False).
  --step_hybz STEP_HYBZ
                        For HybZonotopes, number of HZ abstraction to add at each step for the incremental method. Expressed in fraction of the maximum number of unstable ReLUs, e.g.  
                        step_hybz = 20 means we will increase nb_unstable // 20 at each step. Defaults to 20. Must be positive
  --hybz_relu_method HYBZ_RELU_METHOD
                        Two different abstractions are possible for the ReLU with the HybZonotope domain. They are both exact but might yield different solving time. - True is from    
                        Zhang et al.: https://ieeexplore.ieee.org/document/10163910 (and not the arxiv version) - False is from Ortiz et al.:
                        https://arxiv.org/pdf/2304.02755.pdfDefaults to False.
  --intermediate_concr [INTERMEDIATE_CONCR]
                        For HybZonotopes, to do concretisation with MILP only at the end of the analysis or at each step. Doing so might be longer but will be more precise than        
                        concretisation of HZ as CZ. Defaults to False.
  --solver [SOLVER]     For ConZonotopes and HybZonotopes, whether to use an exact solver or an iterative method for concretisation. Defaults to False. Solver is not scaling well on   
                        big problems.
  --nb_it NB_IT         Number of iterations for the iterative method. Defaults to 500.

Analysis options:
  --split [SPLIT], -s [SPLIT]
                        Recursively splits the inputs if the analysis is too imprecise to conclude.
  --timeout TIMEOUT, -t TIMEOUT
                        Timeout (default: inf)
  --add_suffix [ADD_SUFFIX]
                        Adds a layer to the network to refine properties which includes comparison between outputs.
  --verbose [VERBOSE], -v [VERBOSE]
                        Displays progress bar. (Default: True)
  --nb_process NB_PROCESS, -p NB_PROCESS
                        Number of processes to use when doing analysis with splits. (default: 1)
  --log_intermediate [LOG_INTERMEDIATE]
                        Stores all intermediate results when running analysis (e.g intermediate zonotopes and bounds). If False only store final domain and output_bounds.
  --transpose [TRANSPOSE], -tr [TRANSPOSE]
                        Transposes the weights of the network. Use it when loading nnet or onnx from converted nnet files.
  --squeeze [SQUEEZE], -sq [SQUEEZE]
                        Squeezes the network first dimension. Use it if the network has batch size 1 has first dimension. Don't use it if the first dimension is the number of
                        channels.
  --device DEVICE       Device to execute upon: cpu or cuda,only available when using torch as a library.
  --sound [SOUND]       Option to be correct w.r.t. floating point operations. If True, all operations are rounded correctly towards minus and plus infinity. Otherwise, round to       
                        nearest is used. Does not work on cuda. (default: True)
  --library LIBRARY     Which computing library to use. Options: numpy, torch. (default: numpy)
  --conv_type CONV_TYPE
                        Which convolution operation to use when library is Torch. If manual, use a handcrafted method. If torch, the convolution is done using torch.conv2d operator.   
                        Options: torch, manual (default: torch)
  --it_optim_alpha IT_OPTIM_ALPHA
                        Number of iteration for alpha crown optimization on poly domains. It is slower but more precise. If the number is negative, no alpha optimization are
                        performed. (default: -1). This option is only available for the torch library.
  --write [WRITE]       Option to write NoiseBox on disk when going over the write limit. This option allow to alleviate RAM issues by using temporary files instead.
  --write_limit WRITE_LIMIT
                        Maximum size in bytes of the NoiseBox that will be kept in RAM. NoiseBox over this write limit will be stored in a temporary file on the hard drive. Default    
                        is -1 which is converted to all the memory available at the launch of PyRAT.

Counter examples options:
  --check CHECK         Choose if and when to check for counter example to the property dataset. Must be in 'before', 'after', 'skip', 'both'. 'before' means before an analysis,       
                        'after' is only done after completing the analysis, 'skip' removes the counter example checks. If 'before' is chosen the foolbox library will be used to look   
                        for counter examples otherwise only the coefficients from the domain will be used. (default: 'skip')
  --attack [ATTACK ...]
                        Choose the adversarial attack type for the counter example check. Must be in 'fgsm', 'pgd', 'bim', 'bounds'. 'bounds' means the current box bounds will be      
                        usedaccording to the gradient sign (box.upper if > 0 and box.lower if < 0). (default: 'bounds').
  --attack_it ATTACK_IT, --ai ATTACK_IT
                        Number of iterations to use for bim or pgd attacks (default: 3).
  --pgd_steps PGD_STEPS, --ps PGD_STEPS
                        Number of steps for the pgd attack (default: 3).
  --nb_random NB_RANDOM, --nr NB_RANDOM
                        Number of random sample to get in the box to look for counter examples (default: 5).
  --infer_label [INFER_LABEL]
                        Automatically infer the expected label from a property to verify in case only the property is given.

Adversarial robustness options:
  --dataset DATASET     Put the path to a pytorch ImageFolder kind of dataset, or a name in ['mnist', 'cifar_10', 'cifar_100'] for robustness analysis on this dataset.
  --nb_img NB_IMG       Number of images to analyze if certifying robustness on an imagedataset (default: 10)
  --shuffle [SHUFFLE]   Shuffles the image dataset. Only useful if doing robustness analysis on an image set
  --epsilon EPSILON, -eps EPSILON
                        Radius of the ball to certify robust to adversarial inputs (default: 0.03)
  --image IMAGE, -img IMAGE
                        Path to an image to be analyzed for adversarial robustness. Images can be array saved by numpy np.save (must have extension .npy) or images compatible with     
                        PIL.Image.open (jpg, png etc). The shape of the image after loading should be compatible with the network input shape.
  --true_label TRUE_LABEL
                        The label of the image,for adversarial robustness certification
  --total_labels TOTAL_LABELS
                        The total number of labels of the dataset, for adversarial robustness certification (default: 10)
  --grayscale [GRAYSCALE], -gs [GRAYSCALE]
                        Loads the images in grayscale
  --force_analysis [FORCE_ANALYSIS]
                        If set the predicted label will be used as label even if the model has a wrong prediction and the robustness will be considered with regard to this
                        prediction.

Scorer options:
  --scorer SCORER       Type of scorer to use when doing analysis with split on inputs. Current options: 'width', 'coef', 'random' (default: width)
  --polarity_shift POLARITY_SHIFT
                        Shift in case polarity is not centered in 0 (for properties of type N(x) > c with c !=0)
  --coef_axis COEF_AXIS, --ca COEF_AXIS
                        Output axis to use when computing coefficient score. Can be either an int between [0, number of outputs - 1] or 'min' or 'max' or 'sum'
  --initial [INITIAL]   If set the concretization used in the coef scorer will be based on a [-1, 1] interval. If not set the concretization will use the actual input interval.        
  --coef_weight COEF_WEIGHT, --cw COEF_WEIGHT
                        Weight to use in the weighted sum of the coef scorer. Score is (1-cw)*size input + cw * coef score using cw = 0 is equivalent to using the widthscorer
                        (default: 1)

Booster options:
  --booster BOOSTER     Type of booster, must be in 'general', 'no_boost', 'always_boost'
  --refined_split REFINED_SPLIT [REFINED_SPLIT ...]
                        Splitting parameters to use if using analysis refinement
  --booster_threshold BOOSTER_THRESHOLD, --bt BOOSTER_THRESHOLD
                        The threshold above which to boost, for the general booster. Float value, or 'adapt' to use an adaptive threshold dependent on depth'
  --mode MODE           The computation to apply to the raw metrics, must be in [sum, min, max, mean]
  --metric METRIC, --m METRIC
                        The metric that the booster will use, either relu_stat or polarity (default: polarity)
  --progress [PROGRESS]
                        Uses the progress of the metric to evaluate when boosting instead of only the current step metric

Logging options:
  --log_dir LOG_DIR     Directory where to keep logs
  --log_name LOG_NAME   Directory to save the logging result of the experiment
  --log_counter LOG_COUNTER
                        The saving mode of the counter example. We save input and output to numpy file or in vnnlib format. Options: 'numpy' or 'vnnlib' (Default: 'numpy').
  --log_date [LOG_DATE]
                        Adds the date to the name of the logging folder
  --log_number LOG_NUMBER
                        Adds a number to the name of the logging folder
  --header HEADER       Header for the tree html file

Debug options:
  --debug_file DEBUG_FILE
                        File for a logger for debugging purpose
  --debug [DEBUG]       If True will print INFO and DEBUG to the console or a log file

Testing options:
  --use_relative_path [USE_RELATIVE_PATH]
                        If True will consider paths (logging dir, model path, image path...) as relative path from project root

Parsing options:
  --read READ           In 'parsing_only' or 'from_pickle' to do only the parsing or to read from pickle. Can also be 'parse_and_read' to try and load a pickle otherwise saving a new  
                        pickle and continuing the analysis.if not specified will do parsing and analysis at once without saving a pickle.
  --parsed_folder PARSED_FOLDER
                        Folder to store parsed inputs. Default: None
  --parsing_process PARSING_PROCESS
                        Number of processes to use when parsing the vnnlib file. (default: 1)

```

Analysis can be also run with a config.ini specification e.g (more examples in configs folder):

```ini
[data]
model_path = models/mnist/smallFNN.onnx
squeeze = True

[logging]
log_dir = results
log_name = results
log_date = False

[domain]
domains = [zonotopes]
max_size = -1
max_noise = -1
by_layer = True
allow_smaller_size = False

[analysis]
timeout = 20
add_suffix = True
nb_process = 1
verbose = True



[adv_robust]
dataset = mnist
nb_img = 10
shuffle = False
epsilon = 0.03
image = tests/images/npy/mnist.npy
true_label = 1
total_labels = 10
```

No argument is mandatory but some arguments are incompatible (property path and adversarial robustness or acas
etc.). In this case default behavior is:

- If there is no model path the analysis will run on the specified acas network / property
- If there is a model path and no property path:
    - if there is an image path: run analysis on the image (epsilon, true labels and total_labels required)
    - if there is no image and a dataset name: run benchmark on the dataset (nb_image, shuffle are required)

Other cases should fail as the arguments are incomplete (can't run analysis with no property nor image nor dataset
paths).
(proper checks / error raising when parsing arguments is TODO)

The same logic is applied to analysis from config files.

The section headers in the .ini are not mandatory, they just help organize arguments but can be renamed or removed
altogether.

If a config file is supplied with command line argument the CLI argument will overload the config argument. Order of
importance is :

```
CLI explicit arguments > config.ini arguments (optional) > default values.
```

running from terminal examples:

```bash
python pyrat.py --dataset mnist --model_path ./data/models/mnist/smallFNN.onnx --nb_img 100 \
--epsilon 0.05 --split False

python pyrat.py --split True --acas 3 1 3 --verbose True --domains zonotopes

python pyrat.py --model_path ./data/models/technip/test_offi.nnet \
 --property_path ./data/properties/technip/prop3.txt --split True --verbose True \
 --nb_process 4 --domains zonotopes --timeout 60
```
