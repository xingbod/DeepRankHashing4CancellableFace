function [randomString] =  generateRandomString(templateSize)
% This is the core function to generate the random binary string


% The purpose of the rng function is to prevent reproduction of
% same random binary string
rng('shuffle');

% After that, we will generate the random string/ key
randomString = randi(2,1,templateSize) - 1;

end