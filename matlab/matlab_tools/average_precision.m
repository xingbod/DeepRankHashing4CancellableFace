function [ mAP] = average_precision(D, S,top_k)
%% compute performace metrics
% D: test training pairwise distance
% S: test training ground truth similarity label
% R: options for radius of hamming ball
% M: options for number of nearest neighbors 

[Ntest, Ndb] = size(D);
S = logical(S);
% top_k=2000;

% mAP and topM preision
[~, Id] = sort(D, 2);                
lid = sub2ind(size(D), repmat(1:Ntest, 1, Ndb), Id(:)');
Stmp = reshape(S(lid), Ntest, Ndb);
Scumsum = cumsum(Stmp, 2);
P =  Scumsum ./ repmat(1:Ndb, Ntest, 1);% j th precision
R =  Scumsum ./ sum(S,2) ;% j th recall R(0)=0;

P_10 = P(:, 1:top_k);
R_10 = R(:, 1:top_k);
for ri=1:top_k
    if ri==1
         R_10(:,ri)=R(:,ri) - 0;
    else
         R_10(:,ri)=R(:,ri) - R(:,ri-1);
    end
end

mAP =mean(sum(P_10.*R_10, 2));

