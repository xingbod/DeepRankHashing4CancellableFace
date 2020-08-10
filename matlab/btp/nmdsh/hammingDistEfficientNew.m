function [W2]=hammingDistEfficientNew(B1,B2,SHparam)
% simply call hammingDistEfficient multiple times for each
% threshold
threshs=unique(SHparam.thresholds);
W2=0;
for tt=1:length(threshs)
    paramI=SHparam;
    I=find(SHparam.thresholds==threshs(tt));
    paramI.lambdas=SHparam.lambdas(I);
    paramI.nbits=length(I); 
    paramI.modes=SHparam.modes(I,:);
    B1I=B1(:,I);
    B2I=B2(:,I);
    W2=W2+hammingDistEfficient(B1I,B2I,paramI);
end
