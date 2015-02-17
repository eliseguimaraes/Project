function [NormalData] = Normalisation(Data)
M = sum(Data)/length(Data);
NormalData = (Data - M)/std(Data);
end

