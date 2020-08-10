function [key]=biohashingKey(w,n)
%w= number of key size
%n= number of features of the biometric data
%% dj random numbers

key=orth(rand(n,w));


end