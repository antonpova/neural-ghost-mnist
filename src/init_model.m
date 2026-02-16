function status = init_model(filename, layer_sizes)
    % X*W+b
    % 784 -> 512 -> 128 -> 32 -> 2 -> 32 -> 128 -> 512 -> 784
    
    %layer_sizes=[784, 512, 128, 32, 2, 32, 128, 512, 784];
    
    layer = struct();
    
    for i = 1:(length(layer_sizes)-1) 
        layer(i).W = randn(layer_sizes(i),layer_sizes(i+1)) * sqrt(2/layer_sizes(i));
        layer(i).b = zeros(1, layer_sizes(i+1));
    
        layer(i).act = "relu";
        if i==(length(layer_sizes)-1)
            layer(i).act = "sigmoid";
            layer(i).W = randn(layer_sizes(i),layer_sizes(i+1)) * sqrt(1/layer_sizes(i));
        elseif layer_sizes(i+1)==2
            layer(i).act = "linear";
        end
    end
    save(filename, "layer")
    status = 'Model initialized successfully';
end