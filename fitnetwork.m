% Creates, trains, tests and validates a fitting network, returning its
% performance
function [p,net1] = fitnetwork(n, inputs, targets, trainFunction)
inputs = NormalData;
targets = NormalRainRate;
n = 30;
% Create a Fitting Network
hiddenLayerSize = n;
net = fitnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%Training function
net.trainFcn = trainFunction;  % Levenberg-Marquardt

%Performance Function
net.performFcn = 'mse';  % Mean squared error

%Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};


% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

% Recalculate Training, Validation and Test Performance, applying masks to
% data
trainTargets = targets .* tr.trainMask{1};
valTargets = targets  .* tr.valMask{1};
testTargets = targets  .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,outputs);
valPerformance = perform(net,valTargets,outputs);
testPerformance = perform(net,testTargets,outputs);

p = performance;
% View the Network
view(net)
net1=net;
