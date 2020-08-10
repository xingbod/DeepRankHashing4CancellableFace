function [deltas,yy ] = findMDSHdeltas(SHparam,numBuckets)
% find a set of buckets to search for mdsh
% instead of all buckets that differ from the query by Hamming distance one
% at most, here we take the all buckets that differ from the query by delta
% and we precompute the delta (ind. of query)

nb=SHparam.nbits;
deltas=zeros(numBuckets,nb);

deltasBig=ones(nb)-2*eye(nb); % all Hamming distance one

% add Hamming distance two
for i=1:nb
    for j=(i+1):nb
        dd=ones(1,nb);
        dd(i)=-1;dd(j)=-1;
        deltasBig=[deltasBig;dd];
    end
end


% [~,iisort]=sort(SHparam.lambdas);
% % add Hamming distance three for the top 10 single dimensions
% for i=1:10
%     for j=(i+1):10
%         for k=(j+1):10
%              dd=ones(1,nb);
%             dd(iisort(i))=-1;dd(iisort(j))=-1;dd(iisort(k))=-1;
%             deltasBig=[deltasBig;dd];
%         end
%     end
% end
            
    


B1=ones(1,nb);
W2=hammingDistEfficientNew(B1,deltasBig,SHparam);
[yy,ii]=sort(W2,'descend');
deltas=deltasBig(ii(1:numBuckets),:);
yy=yy(1:numBuckets);


end

