function [distane] = hamming2(X,Y,opts)
%HAMMING2 Summary of this function goes here
%   Detailed explanation goes here

distane = 1- pdist2(X,Y,'Hamming');

end

