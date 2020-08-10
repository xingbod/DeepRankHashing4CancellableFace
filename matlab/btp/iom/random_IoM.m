function model = random_IoM(opts)
dx = opts.dX;
K = opts.K;
L = opts.L;
%%
Px = cell(1, L);% projection matrx tmp
%%
for n = 1 : L
    if opts.gaussian
        Wx = normc((mvnrnd(zeros(dx, 1), diag(ones(dx, 1)), K))');
    else
        Wx=laprnd(dx, K, 0, 1);
    end
    Px{n} = Wx;
end
model.Wx = cell2mat(Px);