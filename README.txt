To operate, run 'RamiXorBatch.m' in MatLab/Octave, it has no dependencies. It trains a neural network to approximate the logical XOR-function
and saves the weight matrices representing the neural network into files and produces and saves a few plots
describing the evolution on the weights and the error function during the training progress.

The file TF-XOR.py contains a TensorFlow/Keras implementation for comparison.

The text below is copied from the beginning of 'RamiXorBatch.m'.




## A neural network for the logical XOR function. Personal proof of concept and practice run to get the basics right in a small controllable context.
## Not meant to be scalable, so this is full of literal constants and some ad-hoc fixes. Furthermore many things that should be encapsulated
## to functions are not. Reader beware.
##
##The trained neural network is saved in the form
## of four .mat files: XOR_{W,B}{1,2}.mat, but there is currently no functionality to use these directly.
## Rami Luisto 2020


%{
General idea:

The architecture of the network is two input nodes, one hidden layer consisting of three neurons and
a single output node. The expected behaviour is contained in the Training Data matrix D; each row contains the input
(the first two elements) and the expected output (the last element).

The network is trained with backwards propagation and Gaussian descent. We train in batches of four, always
averaging over the behaviour of the network for the four basic test cases. We repeat these batches until
the maximum rounds are achieved.
 
The used sigmoid function is specified in the function phi, and it's derivative needs to be contained in the function
called phiprime. We use the same Sigmoid in all the neurons, including the output.
The error function we use is the square Euclidean, contained in function called error.

Some arrays are used to plot the evolution of various constants at the end, these are named *_Plotter.
%}
