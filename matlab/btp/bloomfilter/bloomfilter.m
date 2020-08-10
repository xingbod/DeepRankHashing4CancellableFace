function [template] = bloomfilter(feat,opts)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
feat = reshape(feat,[opts.H opts.W]);
N_BITS_BF = opts.N_BITS_BF ; % parameters for BF extraction 2^10
N_WORDS_BF = opts.N_WORDS_BF;
BF_SIZE = power(2, N_BITS_BF);
N_BF_Y = floor(opts.H/N_BITS_BF);
N_BF_X = floor(opts.W/N_WORDS_BF);
N_BLOCKS = N_BF_X * N_BF_Y;

template = zeros(N_BLOCKS, BF_SIZE); %80*3*10, 16

index = 1;
for x=1:N_BF_X %0 1 2/ 1 2 3
    for y=1:N_BF_Y
        bf =zeros(BF_SIZE,1);
        
        ini_x = (x-1) * N_WORDS_BF+1;
        fin_x = (x) * N_WORDS_BF;
        ini_y = (y-1) * N_BITS_BF+1;
        fin_y = (y ) * N_BITS_BF;
        new_hists = feat(ini_y: fin_y, ini_x: fin_x);
        
        for k=1:N_WORDS_BF
            hist = new_hists(:, k);
            location =sum(power(2,find(hist>0)-1));
            bf(location+1) = 1;
        end
        template(index,:) = bf;
        index=index +1;
    end
end


% template = reshape(template,[1 numel(template)]);

end

