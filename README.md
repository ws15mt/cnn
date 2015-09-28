# cnn
C++ neural network library

#### Getting started

You need the [development version of the Eigen library](https://bitbucket.org/eigen/eigen) for this software to function properly. If you use the current stable release, you will get an error like the following:

    Assertion failed: (false && "heap allocation is forbidden (EIGEN_NO_MALLOC is defined)"), function check_that_malloc_is_allowed, file /Users/cdyer/software/eigen-eigen-10219c95fe65/Eigen/src/Core/util/Memory.h, line 188.

#### Building

First you need to fetch the dependent libraries

    git submodule init
    git submodule update

In `src`, you need to first use [`cmake`](http://www.cmake.org/) to generate the makefiles

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

To compile on Windows, use the introduction in the section "Building on Windows". To compile on Linux, run the following command

    make -j 2

To see that things have built properly, you can run

    ./examples/xor

which will train a multilayer perceptron to predict the xor function.

#### Building without Eigen installed

If you don't have Eigen installed, the instructions below will fetch and compile
both `Eigen` and `cnn`.
        
    git clone https://github.com/yoavg/cnn.git
    hg clone https://bitbucket.org/eigen/eigen/

    cd cnn/
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=../eigen
    make -j 2

#### Building on Windows

For windows, you need to have prebuild Boost library downloaded from 
http://boost.teeks99.com/

##### Using Eigen as the backend
   
This would be the same procedure as above, except that we build a x64 system. The following builds a x64 system for Visual Studio 2013. 

    cmake .. -G"Visual Studio 12 Win64" -DEIGEN3_INCLUDE_DIR=/path/to/eigen

For other versions of Visual Studio, substitute 12 with corresponding version number, such as 11 for Visual Studio 2012. 

To build a win32 system, no need to include -G"Visual Studio 12 Win64". 

##### CUDA-enabled backend on Windows

On windows need to first create a symbolic link to CUDA, e.g., 

    mklink /D d:\tools\cuda "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0"
    
Make sure that CMakeLists.txt have the right cuda directories -DCUDAROOT=/path/to/cuda, and in this case is d:\tools\cuda

Build using

    cmake .. -G"Visual Studio 12 Win64" -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DCUDAROOT=/path/to/cuda

e.g., 

    cmake .. -G"Visual Studio 12 Win64" -DCUDAROOT=d:\tools\cuda -DEIGEN3_INCLUDE_DIR=d:\tools\eigen\eigen-eigen-a64c945a8fb7 -DBACKEND=cuda

Only release mode is supported for CUDA. Other modes such as Debug and RelWithDebug have compilation errors. 

#### Debugging

If you want to see the compile commands that are used, you can run

    make VERBOSE=1

#### Training Models

An illustation of how models are trained (for a simple logistic regression model) is below:

```c++
// *** First, we set up the structure of the model
// Create a model, and an SGD trainer to update its parameters.
Model mod;
SimpleSGDTrainer sgd(&mod);
// Define model parameters for a function with 3 inputs and 1 output.
Parameters& p_W = mod.add_parameters({1, 3});
// Create a "computation graph," which will define the flow of information.
CompuationGraph cg;
// Load the parameters into the computation graph. A VariableIndex identifies the
// position of a particular piece of information within the computation graph.
VariableIndex i_W = cg.add_parameters(&p_W);
// Create variables defining the input and output of the regression, and load them
// into the computation graph. Note that we don't need to set concrete values yet.
vector<cnn::real> x_values(3);
cnn::real y_value;
VariableIndex i_x = cg.add_input({3}, &x_values);
VariableIndex i_y = cg.add_input(&y_value);
// Next, set up the structure to multiply the input by the weight vector,  then run
// the output of this through a sigmoid function (logistic regression).
VariableIndex i_f = cg.add_function<MatrixMultiply>({i_W, i_x});
VariableIndex i_y_pred = cg.add_function<Softmax>({i_f});
// Finally, we create a function to calculate the loss. The model will be optimized
// to minimize the value of the final function in the computation graph.
VariableIndex i_l = cg.add_function<BinaryLogLoss>({i_y_pred, i_y});
// We are now done setting up the graph, and we can print out its structure:
cg.PrintGraphViz();

// *** Now, we perform a parameter update for a single example.
// Set the input/output to the values specified by the training data:
i_x = {0.5, 0.3, 0.7};
i_y = 1.0;
// "forward" propagates values forward through the computation graph, and returns
// the loss.
cnn::real loss = as_scalar(cg.forward());
// "backward" performs back-propagation, and accumulates the gradients of the
// parameters within the "Model" data structure.
cg.backward();
// "sgd.update" updates parameters of the model that was passed to its constructor.
// Here 1.0 is the scaling factor that allows us to control the size of the update.
sgd.update(1.0);
```

Note that this very simple example that doesn't cover things like memory initialization, reading/writing models, recurrent/LSTM networks, or adding biases to functions. The best way to get an idea of how to use cnn for real is to look in the `example` directory, particularly starting with the simplest `xor` example.

#### Others

Kaisheng Yao is mainly working on Windows machine. So this setup on this homepage should work on Windows. If you have any questions, please send email to 

kaisheny@microsoft.com

