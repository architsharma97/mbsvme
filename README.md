## About
The repository hosts scripts related to Mixture of Bayesian SVMs.

## Setup
```
chmod +x scripts/init.sh
./scripts/init.sh
```

## Dataset
Picked from [here](https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets/blob/master/benchmarks.mat). The dataset consists of multiple binary classification tasks. Each dataset also consists of multiple train/val splits. For more details, follow the [link](https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets).

## Script Examples
The options for datasets are: banana, breast_cancer, diabetis, flare_solar, german, heart, image, ringnorm, splice, image, titanic, waveform.

### Baseline
Treats the classification task as a ridge regression problem. Use -d/--data for choosing the dataset and -r/--reg_value to control the value of L2 regularization.

```
python baseline.py -d german -r 1.0
```

### Deep Neural Network
_needs tensorflow_
A simple 3 layer deep neural network. The network is small enough to be trained on a CPU (in a few minutes usually).
```
python dnn.py -d breast_cancer
```

### Mixture of Bayesian SVMs
The model is discussed over [here](https://architsharma97.github.io/resources/mbs_report.pdf). The code for different gating architectures is available in this repository:

#### Softmax Gating Network with Gradient Descent Learning
With the softmax gating network, the gradient descent takes a fixed number of steps along the gradient (-s/--steps for number of steps and -l/--lrate for the learning rate). Other options are: -k/--experts for number of experts, -m/--max_iters for maximum number of iterations of the outer loop, -rg/--reg_val_gate is the hyperparameter to the control L2 regularization of gating network (similarly -re/--reg_val_exp for expert weight vectors). -d option is applicable as well.

```
python mbsvme.py -d breast_cancer -s 10 --lrate 0.05 -k 4 -re 1.0 -rg 2.0
```

#### Generative Gating Networks
Applicable options (described as above): -d, -r, -k, -m

```
python mbsvme_gen.py -d breast_cancer -r 10.0 -k 2
```
#### Softmax Gating Network with Polya-Gamma Augmentation
Applicable options (described above): -d, -k, -m, -re, -rg

```
python mbsvme_pg.py -d breast_cancer -re 10.0 -rg 5.0 -k 10
```