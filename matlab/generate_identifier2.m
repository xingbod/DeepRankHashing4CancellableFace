function [identifiers ] = generate_identifier2(m,q,num_user)

identifiers = zeros(6000,m);
for i=1:num_user
    identifiers(i,:) = round(rand(1,m)*(2^q-1));
end

end