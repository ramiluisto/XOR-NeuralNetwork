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


## Upper limit on the amount of training rounds. 10000 is almost always enough, 5000 is often.
MAX_ROUNDS = 10000;


## Gradient jump constant
EPSILON = 1;

## Plotting parameters. (Plotting an array with a million inputs is a bit
## slow, so we only use the Plotter_Range amount of points.)
Plotter_Range = 100
if (MAX_ROUNDS < Plotter_Range)
  Plotter_Range = MAX_ROUNDS;
end

Snapshot_Interval = idivide(MAX_ROUNDS,Plotter_Range);

W1_Plotter = zeros(2,3,Plotter_Range);
W2_Plotter = zeros(3,1,Plotter_Range);
B1_Plotter = zeros(1,3,Plotter_Range);
B2_Plotter = zeros(1,1,Plotter_Range);
Error_Plotter = [];


## Training data. Contains the four cases of logical XOR.
Training_Data = [
     0,0,0 ;
     1,0,1 ;
     0,1,1 ;
     1,1,0
];

## The sigmoid used in all the neurons.
function y = phi(x)
  y = 1 / (1 + exp(-x));
end

function y = phiprime(x)
  y = phi(x)*(1-phi(x));
end

## Euclidian squared error function.
function err = error(T, Y)
  err = 0.5*(T-Y)**2;
end


## Functions to calculate the weighted inputs of the
## output neuron (Z2) and the three hidden neurons (Z1).
function Z1 = GetZ1( input_values, Weights_1, Constants_1 )
  Z1 = Weights_1'*(input_values') + Constants_1';
end

function Z2 = GetZ2( Z1, Weights_2, Constants_2 )
  Z2 = Weights_2'*(arrayfun(@phi,Z1)) + Constants_2';
end



## Randomized weight matrices. W1 and W2 are the weights
## between the layers and B1 and B2 are the constant weights.
W1 = 2*(-0.5*ones(2,3) + rand(2,3));
W2 = 2*(-0.5*ones(3,1) + rand(3,1));
B1 = 2*(-0.5*ones(1,3) + rand(1,3));
B2 = 2*(-0.5*ones(1,1) + rand(1,1));



## Start the training routine. j will be the ubiquituous running index.
### Training START
for j = 1:MAX_ROUNDS

  ## Grapping Plotting Data for a sparse set
  if (rem(j,Snapshot_Interval) == 0)
    k = j/Snapshot_Interval;
    W1_Plotter(:,:, k) = W1;
    W2_Plotter(:,:, k) = W2;
    B1_Plotter(:,:, k) = B1;
    B2_Plotter(:,:, k) = B2;
  end

  ## Initializing variables to hold batch averages of derivatives.
  Delta1_Avg = zeros(2,3);
  Delta2_Avg = zeros(1,3);
  DeltaB1_Avg = zeros(1,3);
  DeltaB2_Avg = zeros(1,1);

### Batch START
  ## Batch start. We do all the four elementary cases in a single batch.
  for i = 1:4
    ## Extract the training data input point and expected output.
    X = Training_Data(i:i,1:2);
    Y = Training_Data(i:i,3:3);

    ## Calculate neuron input values.
    Z1 = GetZ1( X, W1, B1);
    Z2 = GetZ2( Z1, W2, B2);

    ## Print current progress
    Status = sprintf(
		 "|| Progress: %5.1f%% | Y = %d | Z2 = %4.2f | err = %5.3f ||",
		 100*j/MAX_ROUNDS, Y, phi(Z2) , error(phi(Z2),Y))
    
    ## Grapping error date for plotting from a sparse set
    if (rem(j,Snapshot_Interval) == 0)
      k = j/Snapshot_Interval;
      Error_Plotter(k) = error(phi(Z2), Y);
    end

    
    ## We next calculate derivatives for the weights in W1, W2, B1 and B2.

    ## This delta2-constant is used in various other calculations and
    ## represents the derivative of the error w.r.t. the input of
    ## the output neuron.
    delta2 = (phi(Z2) - Y)*phiprime(Z2);
    
    ## Derivatives of layer 2 weights in W2
    Delta2 = zeros(1,3);
    for k = 1:3
      Delta2(k) = delta2*phi(Z1(k));
    end
    
    ## Derivatives of layer 1 weights in W1
    Delta1 = zeros(2,3);
    for k = 1:3
      for m = 1:2
	Delta1(m,k) = X(m)*phiprime(Z1(k))*Delta2(k)*W2(2);
      end
    end
    
    ## Derivatives of layer 2 constant weight(s) in B2.
    DeltaB2 = delta2;
    
    ## Derivatives of layer 1 constant weight in B1
    for k = 1:3
      DeltaB1(k) = delta2*W2(k)*phiprime(Z1(k));
    end
    

    ## Add calculated errors to the averaging counters
    Delta1_Avg = Delta1_Avg + Delta1;
    Delta2_Avg = Delta2_Avg + Delta2;
    DeltaB1_Avg = DeltaB1_Avg + DeltaB1;
    DeltaB2_Avg = DeltaB2_Avg + DeltaB2;
  end
### Batch END

  
  ## Calculate averages from the averaging counters.
  Delta1_Avg = Delta1_Avg./4;
  Delta2_Avg = Delta2_Avg./4;
  DeltaB1_Avg = DeltaB1_Avg./4;
  DeltaB2_Avg = DeltaB2_Avg./4;

  
  ## Begin the updating of weights via the gradient descent.
  ## Each weight W is altered to W - EPSILON*derivative(W).
  
  ## Update layer one weights in W1
  for k = 1:3
    for m = 1:2
      W1(m,k) = W1(m,k) - EPSILON*Delta1_Avg(m,k);
    end
  end
  
  ## Update layer two weights in W2
  for i = 1:3
    W2(i) = W2(i) - EPSILON*Delta2_Avg(i);
  end
  
  ## Update constant weights in B1
  for i = 1:3
    B1(i) = B1(i) - EPSILON*DeltaB1_Avg(i);
  end

  ## Update constant weights in B2
  B2 = B2 - EPSILON*DeltaB2_Avg;

### Training END
end

## Saving the matrices to actual files.
save XOR_W1.mat W1;
save XOR_W2.mat W2;
save XOR_B1.mat B1;
save XOR_B2.mat B2;


Status = sprintf("\nTraining complete. Moving to testing. \n")

## We next test our network with the four base cases.
### Testing START
Status = "Begin testing"
for i = 1:4
  
  ## Extract the training data point.
  X = Training_Data(i:i,1:2);
  Y = Training_Data(i:i,3:3);
  Training_Data(i:i,:);
  ## Calculate neuron input values.
  Z1 = GetZ1( X, W1, B1);
  Z2 = GetZ2( Z1, W2, B2);
  sprintf("Input: %d,%d | Expected: %d | Given: %6.4f | Err: %4.2f", X(1),X(2),Y,phi(Z2),error(phi(Z2),Y))
end
### Testing END

## Plotting. All the plot data is contained in the
## matrices W1_Plotter, W2_Plotter, B1_Plotter and B2_Ploter.
## Getting it out to a plottable form was a bit nontrivial
## for some reason, the following solution works but is far
## from elegant. We essentially get all the plottable stuff
## into a single large matrix called Printable_Data and plot that.
Printable_Data = zeros(Plotter_Range, 13);

## Populating matrix.
for i = 1:3
  for j = 1:Plotter_Range
    Printable_Data(j,i) = W1_Plotter(1,i,j);
  end
end
for i = 1:3
  for j = 1:Plotter_Range
    Printable_Data(j,i+3) = W1_Plotter(2,i,j);
  end
end
for i = 1:3
  for j = 1:Plotter_Range
    Printable_Data(j,i+6) = W2_Plotter(i,1,j);
  end
end
for i = 1:3
  for j = 1:Plotter_Range
    Printable_Data(j,i+9) = B1_Plotter(1,i,j);
  end
end
for i = 1:1
  for j = 1:Plotter_Range
    Printable_Data(j,i+12) = B2_Plotter(1,i,j);
  end
end


Status = "Plotting...\n"

# Plot the weights and save the result.
figure;
Weight_Plot = plot(1:Plotter_Range, Printable_Data(:,:));
title("Evolution of various weights.")
saveas(gcf, 'Weights.png')


# Plot the error and save the result.
figure;
Error_Plot = plot(1:Plotter_Range, Error_Plotter);
title("Evolution of the error function.")
saveas(gcf, 'Error.png')

pause

