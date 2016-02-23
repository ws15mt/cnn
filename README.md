# CNN
C++ neural network library

## Introduction

This toolkit is folked from Chris Dyer's CNN toolkit. It has however been extensively developed since then. The following features and improvements are made. 

### Features

Data parallel : Support loading multiple sentences so that training can be efficiently run. One way is to use these sentences without padding. The other way is padding shorter sentences with </s> symbols.  

GPU : Many CPU function calls have been extended to run on GPUs.

Double precision: The code can be built with double precision. 

Support CUDNN : some function calls use cuDNN implementations. These include softmax. 

#### Supported Modeling Techniques

It is very easy to develop advanced models using this toolkit. I have been experimented with Theano and CNTK and some other tools. Compared to other tools, the main advantage of this tool is its ease of developing new models. We have used this tool for attentional neural machine translation, conversation modeling, and speech processing. The second advangtage is its ease of deployment; it is easy to expose functions as a DLL so that those functions can be called from Python or C#. This is a big advantage over other tools since you are able to deploy your models for production easily. 


#### Getting started

You need the [development version of the Eigen library](https://bitbucket.org/eigen/eigen) for this software to function properly. If you use the current stable release, you will get an error like the following:

    Assertion failed: (false && "heap allocation is forbidden (EIGEN_NO_MALLOC is defined)"), function check_that_malloc_is_allowed, file /Users/cdyer/software/eigen-eigen-10219c95fe65/Eigen/src/Core/util/Memory.h, line 188.

To get the development version, simply download its most recent code from the following link

	https://bitbucket.org/eigen/eigen/downloads

#### Building

First clone cnn 
   
    git clone https://github.com/kaishengyao/cnn
    cd cnn


Then you need to fetch the dependent libraries
    
    git submodule init
    git submodule update

If you use Windows, jump to the later section for Building on Windows

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

Pick the one that matches your Visual Studio version. For example, the following is a link to Visual Studio 2013 Win64

	https://sourceforge.net/projects/boost/files/boost-binaries/1.59.0/boost_1_59_0-msvc-12.0-64.exe/download

Once Boost is installed, make two directories under cnn. One directory is for building CPU build and the other directory is for GPU build.
 
##### CPU-build : Using Eigen as the backend
   
First make a build directory:

	mkdir msbuild 

Then, cd to this directory

	cd msbuild

Then make a project using the following command. The following command is for building with Visual Studio 2013. Visual Studio 2013 is recommended. Visual Studio 2012 will not be able to build. Visual Studio 2015 is too slow. It may have an improved version for Visual Studio 2015. 

The following builds a Win64 bit. 

    cmake .. -G"Visual Studio 12 Win64" -DEIGEN3_INCLUDE_DIR=/path/to/eigen

To build a win32 system, no need to include -G"Visual Studio 12 Win64". 

##### GPU-build : CUDA-enabled backend on Windows

You need to first install CUDA. CUDA is usually installed to a directory such as "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5", which we will refer it as <your CUDA dir>.

Then, you need to create a symbolic link to CUDA, with command mklink /D <your symbolic CUDA dir> <your CUDA dir>; e.g., 

        mklink /D d:\tools\cuda "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5"
    
Then make a directory

	mkdir msbuildcuda

	cd msbuildcuda

Then download cuDNN (v.4) or later version from NVDIA webpage. Copy cudnn.h to <your cuda dir>/include and cudnn.lib to <your CUDA dir>/lib/x64. 
 

When building, make sure that CMakeLists.txt have the right cuda directories -DCUDA_ROOT=/path/to/cuda, and in this case is d:/tools/cuda

Build using

    cmake .. -G "Visual Studio 12 Win64" -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DCUDA_ROOT=/path/to/cuda -DCUDNN_ROOT=/path/to/cudnn

e.g., 

    cmake .. -G "Visual Studio 12 Win64" -DCUDA_ROOT=d:/tools/cuda -DEIGEN3_INCLUDE_DIR=d:\tools\eigen\eigen-eigen-a64c945a8fb7 -DBACKEND=cuda -DCUDNN_ROOT=d:/tools/cuda

Notice that the path to cuda needs to use '/' instead of '\'. 

Only release mode is supported for CUDA. Other modes such as Debug and RelWithDebug have compilation errors, because nvcc doesn't support some c++11 features which are in Eigen3 library.  

#### What to do after cmake

The cmake makes project file. In msbuild and msbuildcuda, you can find Visual Studio project file cnn.sln. Open the project file in Visual Studio. You should try rnnlm2 project, as that is the one currently well supported. Other projects may have compilation errors but they are easy to fix. Contact me if you see any build problems.  

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

### Examples

Examples are under examples directory. A well-tested example is rnnlm2. Its experiment is at exp/lm/rnnlm2.bat.

### Contacts:

kaisheny@microsoft.com


