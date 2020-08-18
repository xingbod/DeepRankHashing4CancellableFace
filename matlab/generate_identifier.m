function [identifiers ] = generate_identifier(m,q,num_user)

identifiers = zeros(6000,m*q);
for i=1:num_user
    tmp = dec2bin(round(rand(1,m)*(2^q-1)))-'0';
    identifiers(i,:) = reshape(tmp',1,numel(tmp));
end


end