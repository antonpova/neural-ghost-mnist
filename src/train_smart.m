function [layers, history] = train_smart(X_train, Y_train, layers, cfg, run_folder, anim_data)
    num_train = size(X_train, 2);
    
    
    % folders
    evo_folder = fullfile(run_folder, 'evolution');
    anim_folder = fullfile(run_folder, 'animation');
    
    if ~exist(evo_folder, 'dir'), mkdir(evo_folder); end
    if ~exist(anim_folder, 'dir'), mkdir(anim_folder); end
    
    fixed_indices = zeros(1, 10);
    for d = 0:9
        fixed_indices(d + 1) = find(Y_train == d, 1);
    end
    fixed_batch = X_train(:, fixed_indices)';
    
     % --- trajectories ---
    traj_filename = fullfile(run_folder, 'trajectories.csv');
    fid = fopen(traj_filename, 'w');
    fprintf(fid, 'epoch,point_id,latent_x,latent_y,label\n');
    fclose(fid);
    % -----------------------------

    %start
    history = [];
    best_loss = inf; 
    

    fprintf('checking the loss ...\n');
 
    test_batch = X_train(:, 1:min(1000, num_train))';
    [y_test, ~] = model_forward(test_batch, layers);
    initial_loss = mean((y_test - test_batch).^2, 'all');
    best_loss = initial_loss;
    fprintf('Starting with loss: %.6f. Saving as best_model only if it beats this record.\n', best_loss);

    fprintf('--- start (snapshots are in %s) ---\n', run_folder);
    
    figure; 
    hLine = animatedline('Color', 'r', 'LineWidth', 1.5);
    title(['Training Loss (Started: ' char(datetime('now')) ')']); 
    grid on;
    
    global_step = 0;
    

    %ADAM V0 INIT
    m = layers;
    v = layers;
    for k = 1:length(layers)
        m(k).W(:) = 0; m(k).b(:) = 0;
        v(k).W(:) = 0; v(k).b(:) = 0;
    end

    % constans
    beta1 = 0.9;   % Inertia
    beta2 = 0.999; % Scale
    epsilon = 1e-8;
    
    %main loop
    for epoch = 1:cfg.epochs
        perm = randperm(num_train);
        X_shuffled = X_train(:, perm);
        epoch_loss = 0;
        tic; 
        
        for i = 1:cfg.batch_size:num_train
            idx_end = min(i + cfg.batch_size - 1, num_train); %to be save
            x_batch = X_shuffled(:, i:idx_end)'; %get a batch 
            
            if size(x_batch, 1) == 0, continue; end 
            
            global_step = global_step + 1;

            %predict and calculate MSE
            [y_pred, ~, cache] = model_forward(x_batch, layers);
            current_loss = mean((y_pred - x_batch).^2, 'all');
            epoch_loss = epoch_loss + current_loss;
            
            %gradient
            grads = model_backward(y_pred, x_batch, layers, cache);
            
            %correct the weights (legacy) 
            %for L = 1:length(layers)
            %    layers(L).W = layers(L).W - cfg.lr * grads(L).dW;
            %    layers(L).b = layers(L).b - cfg.lr * grads(L).db;
            %end

            %ADAM V0
            for L = 1:length(layers)
                gW = grads(L).dW;
                gb = grads(L).db;
                
                % update inertia (m)
                % "new speed = 0.9 * old + 0.1 * slope"
                m(L).W = beta1 * m(L).W + (1 - beta1) * gW;
                m(L).b = beta1 * m(L).b + (1 - beta1) * gb;

                % update scale (v)
                v(L).W = beta2 * v(L).W + (1 - beta2) * (gW .^ 2);
                v(L).b = beta2 * v(L).b + (1 - beta2) * (gb .^ 2);
                
                % bias correction
                m_hat_W = m(L).W / (1 - beta1^global_step);
                m_hat_b = m(L).b / (1 - beta1^global_step);
                
                v_hat_W = v(L).W / (1 - beta2^global_step);
                v_hat_b = v(L).b / (1 - beta2^global_step);
                
                % update weights
                layers(L).W = layers(L).W - cfg.lr * m_hat_W ./ (sqrt(v_hat_W) + epsilon);
                layers(L).b = layers(L).b - cfg.lr * m_hat_b ./ (sqrt(v_hat_b) + epsilon);
            end

            
            %draw a graph
            if mod(global_step, 50) == 0
                addpoints(hLine, global_step, current_loss);
                drawnow limitrate;
            end
        end
        
        %statistics
        num_batches = floor(num_train / cfg.batch_size);
        avg_epoch_loss = epoch_loss / num_batches;
        time = toc;
        
        fprintf('Ep %d | Loss: %.6f | Time: %.1fs ', epoch, avg_epoch_loss, time);
        
        % ---------------------------------------
        % 1. save evolution
        [evo_out, ~] = model_forward(fixed_batch, layers);
        
        combined_img = [];
        for k = 1:10
            digit = reshape(evo_out(k,:), 28, 28);
            combined_img = [combined_img, digit, ones(28, 2)];
        end
        
        if epoch>=cfg.decay_start
            if mod(epoch, cfg.decay_interval) == 0
                cfg.lr = cfg.lr * cfg.dr;
            end
        end


        f_img_name = fullfile(evo_folder, sprintf('epoch_%03d.png', epoch));
        imwrite(combined_img, f_img_name);
        % ---------------------------------------
        % 2. draw animation frame
        [~, lat_anim] = model_forward(anim_data.X, layers);
        
        fig_anim = figure('Visible', 'off');
        scatter(lat_anim(:,1), lat_anim(:,2), 10, anim_data.Y, 'filled');
        colormap(jet(10)); 
        title(sprintf('Epoch: %d', epoch));
        grid on;
        
        xlim([-15 15]); ylim([-15 15]);
        
        saveas(fig_anim, fullfile(anim_folder, sprintf('frame_%03d.png', epoch)));
        close(fig_anim);

        % 3. Save Trajectories to CSV
        [~, lat_anim] = model_forward(anim_data.X, layers);
        
        num_points = size(anim_data.X, 1);
        ids = (1:num_points)';
        ep_col = repmat(epoch, num_points, 1);
        
        data_block = [ep_col, ids, lat_anim(:,1), lat_anim(:,2), anim_data.Y];
        
        writematrix(data_block, traj_filename, 'WriteMode', 'append');
        % ---------------------------------------




        % 1. best model
        if avg_epoch_loss < best_loss
            best_loss = avg_epoch_loss;
            fprintf('[NEW BEST!] ');
            layer = layers; 
            save(fullfile('models', 'best_model.mat'), 'layer');
        end
        
        % 2. create a snapshot
        if mod(epoch, cfg.snapshot_interval) == 0
            % name: snapshots/run_ДАТА/ep005_loss0.054.mat
            f_name = fullfile(run_folder, sprintf('ep%03d.mat', epoch));
            layer = layers;
            save(f_name, 'layer');
            fprintf('[Saved]');
        end
        
        fprintf('\n');
    end
end