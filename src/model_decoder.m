function output_img = model_decoder(latent_vector, layers)
    % latent_vector: [1 x 2] 
    
    A = latent_vector;
    
    start_index = 0;
    for i = 1:length(layers)
        
        if size(layers(i).W, 1) == 2
            start_index = i;
            break;
        end
    end
    
    if start_index == 0
        error('there is no layer with size 2');
    end
    
    for i = start_index:length(layers)
        
        Z = A * layers(i).W + layers(i).b;
        
        act = layers(i).act;
        
        if strcmp(act, 'relu')
            A = my_relu(Z);
        elseif strcmp(act, 'sigmoid')
            A = my_sigmoid(Z);
        elseif strcmp(act, 'linear')
            A = Z;
        end
    end
    
    output_img = A;
end