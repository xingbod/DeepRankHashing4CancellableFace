function [U,X,Y] = showNdfun(X, u)
% [U,X,Y] = showNdfun(X,u)
%
% Display the ND function u(X)
% X does not need to be a regular grid

L = 512; % horizontal image size
Nbins = 32; % number of bins for the horizontal dimension

X = double(X);
D = size(X,2);
n = size(u,2);

x = X(:,1);
y = X(:,2);

mx = min(x);
Mx = max(x);
my = min(y);
My = max(y);

D = (Mx-mx)/Nbins;
[X,Y] = meshgrid(mx:D:Mx, my:D:My);
[nr,nc] = size(X);

bins = sub2ind([nr nc], fix((nr-1)*(y-my)/(My-my)+1), fix((nc-1)*(x-mx)/(Mx-mx)+1));

U = nan([nr*nc 1 n]);

for i = 1:nr*nc
    j = find(bins==i);
    if ~isempty(j)
        for m = 1:n
            U(i, 1, m) = max(u(j, m));
        end
    end
end

p = prctile(abs(U(:)), 98);
U = uint8(128+128*U/p);

U = reshape(U, [nr nc 1 n]);
U = imresize(U, [NaN L], 'nearest');

montage(U)
colormap([0 0 0; jet(255)])


