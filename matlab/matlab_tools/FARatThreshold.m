function [FAR,FRR] = FARatThreshold(gen,imposter,threshold)
%FARATTHRESHOLD Summary of this function goes here
%   Detailed explanation goes here
FAR=sum(imposter>=threshold)/length(imposter);
FRR=sum(gen<threshold)/length(gen);


end

