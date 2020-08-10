function [W2,weights]=hammingDistEfficient(B1,B2,SHparam)
% this should give the same result as hammingDistMat
% but O(n). 
% the calculation is (1+d1)*(1+d2)*(1+d3)...
% where d1 is the Hamming distance using only features on dimension 1,etc.

weights=SHparam.lambdas;weights=weights(:);
nBits=SHparam.nbits;
ndim=length(SHparam.modes(1,:));
mmodes=SHparam.modes;
W2=1;
for dd=1:ndim
    I=find(mmodes(:,dd)>0);
    dim1Mat=B1(:,I)*diag(weights(I))*B2(:,I)';
    W2=W2.*(1+dim1Mat);
end
W2=W2-1;


end

