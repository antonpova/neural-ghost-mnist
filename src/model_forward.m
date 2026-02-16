function [output, latent, cache] = model_forward(x, layers)
A = x;
latent = [];
cache = struct();
for i = 1:length(layers)
    cache(i).prev_A=A;
    
    Z = A*layers(i).W + layers(i).b;

    cache(i).Z = Z; 

    if size(Z, 2) == 2
        latent = Z;
    end

    if strcmp(layers(i).act, "relu")
        A = my_relu(Z);
    elseif strcmp(layers(i).act, "sigmoid")
        A = my_sigmoid(Z);
    elseif strcmp(layers(i).act, "linear")
        A = Z;
    else
        error("unknown activation function")
    end
    

end
output = A;
end