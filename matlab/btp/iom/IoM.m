function [code, model] = IoM(data, opts, model)

% Do hash encoding
K = opts.K;
L = opts.L;
N = size(data.X, 2);
if ~isempty(data.X)
    Zx = model.Wx'*data.X;
    Zx = reshape(Zx, K, L*N);
    [~, Hx] = max(Zx);
    Hx = reshape(Hx, L, N);
end


code.Hx = Hx;
