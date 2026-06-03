# Multilayer Perceptron (MLP) in C++

A custom, from-scratch implementation of a Multilayer Perceptron neural network written in C++. This project includes its own matrix operations, activation functions, and data loading mechanisms designed to train and test the network without relying on external deep learning libraries. It is configured to work with the included Simpsons MNIST dataset.

**Note on Development Status**: 
Please note that this project is still under active development. Its functionality will continue to be evaluated and improved, as there are still several features and structural components that require further validation and testing to ensure full correctness and stability.

## Project Structure and File Descriptions

### Core Files
* **`main.cpp`**: The main entry point of the application. It orchestrates the initialization, training, and testing phases of the neural network.
* **`makefile`**: Contains the build instructions required to compile the project.

### Source and Header Files (`src/` and `include/`)
* **`Activation.h` & `Activation.cpp`**: Define and implement the activation functions (such as Sigmoid or ReLU) and their derivatives, which are used to introduce non-linearity into the network.
* **`DataLoader.h` & `DataLoader.cpp`**: Handle the reading, parsing, and preprocessing of external datasets (e.g., image data from the `simpsons-mnist` directory) into a matrix format suitable for the neural network.
* **`Matrix.h` & `Matrix.cpp`**: A custom mathematical matrix library that provides all essential linear algebra operations, including matrix multiplication, addition, scaling, and transposition required for forward propagation and backpropagation.
* **`MultiLayerPerceptron.h` & `MultiLayerPerceptron.cpp`**: Contain the core logic for the neural network architecture. This includes managing layers, weights, biases, forward propagation, and weight updates during the training phase.

### Datasets
* **`simpsons-mnist/`**: The dataset directory containing training and testing images (both grayscale and RGB) of various Simpsons characters.

## Build Instructions
To build the project, run the following command in the root directory:
```bash
make
```

To execute the compiled program:
```bash
./mlp
```
