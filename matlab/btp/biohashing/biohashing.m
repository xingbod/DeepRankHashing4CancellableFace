function [transformed_data] = biohashing(biometric_data,key)

if isempty(key)
    key=biohashingKey(128,length(biometric_data(1,:)));
end

transformed_data = biometric_data*key>0;

end