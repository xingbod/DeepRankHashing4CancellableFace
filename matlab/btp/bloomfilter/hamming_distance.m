function [dist] = hamming_distance(X, Y)
 %Computes the noralised Hamming distance between two Bloom filter templates'''
    dist = 0;

    N_BLOCKS = size(X,1);
    
    for i=1:N_BLOCKS
        A = X(i, :);
        B = Y(i, :);
        suma = sum(A) + sum(B);
        if suma > 0
            dist = dist+ sum( xor(A,B)) /suma;
        end
        
    end
    dist = dist / N_BLOCKS;

    end


