

inputSeries = Data;
targetSeries = Rain;

inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};


[inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{}, targetSeries);


net.divideFcn = 'dividerand'; 
net.divideMode = 'value';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainFcn = 'trainlm';

net.performFcn = 'mse';


net.plotFcns = {'plotregression'};

[net,tr] = train(net,inputs,targets,inputStates,layerStates);
view(net)

outputs = net(inputs,inputStates,layerStates);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

%  net = removedelay(net,3);
% view(net)
% [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);
% outputs = net(inputs,inputStates,layerStates);

% trainTargets = gmultiply(targets,tr.trainMask);
% valTargets = gmultiply(targets,tr.valMask);
% testTargets = gmultiply(targets,tr.testMask);
% trainPerformance = perform(net,trainTargets,outputs)
% valPerformance = perform(net,valTargets,outputs)
% testPerformance = perform(net,testTargets,outputs)
% 
% view(net)

% netc = closeloop(net);
% netc.name = [net.name ' - Closed Loop'];
% view(netc)
% [xc,xic,aic,tc] = preparets(netc,inputSeries,{},targetSeries);
% yc = netc(xc,xic,aic);
% closedLoopPerformance = perform(netc,tc,yc)
% 
% nets = removedelay(net);
% nets.name = [net.name ' - Predict One Step Ahead'];
% view(nets)
% [xs,xis,ais,ts] = preparets(nets,inputSeries,{},targetSeries);
% ys = nets(xs,xis,ais);
% earlyPredictPerformance = perform(nets,ts,ys)
