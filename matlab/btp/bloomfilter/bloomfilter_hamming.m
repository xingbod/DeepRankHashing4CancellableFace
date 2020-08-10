function [distane] = bloomfilter_hamming(X,Y,opts)
%BLOOMFILTER_HAMMING Summary of this function goes here
%   Detailed explanation goes here
toall=length(X);
block=toall/opts.BF_SIZE;
X= reshape(X,[block opts.BF_SIZE]);
Y= reshape(Y,[block opts.BF_SIZE]);
dist = 0;

N_BF =size(X,1);
for i=1:N_BF
    A = X(i, :);
    B = Y(i, :);
%     dist  = dist+pdist2(A,B,'Hamming');
    suma = sum(A) + sum(B);
    if (suma > 0)
        dist  = dist+ sum( double(xor(A,B))) / double(suma);

    end
end
        distane= dist / double(N_BF);

end

