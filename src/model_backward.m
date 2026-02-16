function grads = model_backward(y_pred, y_true, layers, cache)
    % y_pred: output_img
    % y_true: input_img
    % layers: W b and so on
    % cache:  Z и prev_A
    
    num_layers = length(layers);
    m = size(y_true, 1); %batch size
    
    grads = struct();

    % formula for MSE: L = sum((Pred - True)^2) / m
    % derivitive dL/dOutput = 2 * (Pred - True) / m

    dA = 2*(y_pred-y_true)./m;

    for i = num_layers:-1:1
        Z = cache(i).Z;
        A_prev = cache(i).prev_A;
        act = layers(i).act;

        % getting through activation (dZ = dA * f'(Z))
        if strcmp(act, 'relu')
            dZ = dA .* my_relu_prime(Z);
            
        elseif strcmp(act, 'sigmoid')
            dZ = dA .* my_sigmoid_prime(Z);
            
        elseif strcmp(act, 'linear')
            dZ = dA;
            
        else
            error('unknown activation in backward');
        end


        % calculating the gradients for W and b
        % dW = transposed input * dZ
        grads(i).dW = A_prev' * dZ;
        
        % db = sum dZ for every image in bach
        grads(i).db = sum(dZ, 1);
        
        % propagate the loss to the next layer
        if i>1
            dA = dZ * layers(i).W';
        end
    end

end