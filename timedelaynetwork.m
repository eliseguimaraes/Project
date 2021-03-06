function [net2,net3,p] = timedelaynetwork(a,b,n, input, target, trainFunction)
% Solve an Input-Output Time-Series Problem with a Time Delay Neural
% Network: prediction
%Function parameters
%   a               ...     input delay
%   b               ...     end of input window
%   n               ...     hidden layer size
%   input           ...     neural network's input data
%   target          ...     neural network's input data
%   trainFunction   ...     neural network's training method


%This orientates data in appropriate columns and rows
inputSeries = tonndata(input,true,false);
targetSeries = tonndata(target,true,false);


inputDelays = a:b; %prediction will be made with data from a to b days before
hiddenLayerSize = n;
net = timedelaynet(inputDelays,hiddenLayerSize);% Creates a Time Delay Network

% Choose Input and Output Pre/Post-Processing Functions
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation, adapting the data to the
% number of delays
[inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,targetSeries);

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every value
net.divideParam.trainRatio = 70/100; % 70% for training
net.divideParam.valRatio = 15/100; % 15% for validation
net.divideParam.testRatio = 15/100; % 15% for testing

%Select training function
net.trainFcn = trainFunction;

% Choose a Performance Function
net.performFcn = 'mse';  % Mean squared error

% Plot Functions
net.plotFcns = {'plotperform','plottrainstate','plotresponse', ...
  'ploterrcorr', 'plotinerrcorr', 'plotregression'};


% Train the Network
[net,tr] = train(net,inputs,targets,inputStates,layerStates);

% Test the Network
outputs = net(inputs,inputStates,layerStates);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(targets,tr.trainMask); %applys a mask to sort out each phases' targets
valTargets = gmultiply(targets,tr.valMask);
testTargets = gmultiply(targets,tr.testMask);
trainPerformance = perform(net,trainTargets,outputs);
valPerformance = perform(net,valTargets,outputs);
testPerformance = perform(net,testTargets,outputs);

% View the Network
view(net);

% Early Prediction Network
% In this some application, it's useful to get the prediction 'a' timesteps early.
% We will have predicted y(t+a) once x(t) is available, but before the actual y(t+a) occurs.
% This is done by simply shifting the output by 'a' timesteps.
nets = removedelay(net,a); %shifts outputs left by 'a' timesteps
[xs,xis,ais,ts] = preparets(nets,inputSeries,targetSeries); %Rearranges data
ys = nets(xs,xis,ais);
earlyPredictPerformance = perform(nets,ts,ys); %Recalculates the performance
net2 = net;
net3 = nets;
p = performance;