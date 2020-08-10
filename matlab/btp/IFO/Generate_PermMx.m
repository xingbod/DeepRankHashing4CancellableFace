function [ PermuteMatx ] = Generate_PermMx( InputSample,P,m)



for ii=1:m
            PermuteMat = zeros(P,size(InputSample,2));
        for i = 1:P 
                PermuteMat(i,:) = randperm(size(InputSample,2),size(InputSample,2));
        end
        
 PermuteMatx{ii}=PermuteMat;
end
end

