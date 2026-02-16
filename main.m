clear; clc; close all;
addpath('src');
% =========================================================
%                      PROJECT CONFIG
% =========================================================

% true  = continue training 
% false = start from scratch
CONTINUE_TRAINING = true; 


rng(42);
% training params
cfg = struct();
cfg.epochs = 10;             
cfg.lr = 0.001;              
cfg.batch_size = 1024;        
cfg.snapshot_interval = 50;  

cfg.decay_start = 80;
cfg.decay_interval = 1;
cfg.dr = 0.95;

paths = struct();
paths.data      = 'data';
paths.models    = 'models';
paths.results   = 'experiments';

model_filename = fullfile(paths.models, 'best_model.mat');

architecture = [784, 512, 128, 32, 2, 32, 128, 512, 784];

% =========================================================
% 1. LOAD DATA
% =========================================================
if ~exist('X_train', 'var')
    fprintf('>>> LOADING MNIST...\n');
    X_train = loadMNISTImages(fullfile(paths.data, 'train-images.idx3-ubyte'));
    Y_train = loadMNISTLabels(fullfile(paths.data, 'train-labels.idx1-ubyte'));
end

% =========================================================
% 2. MODEL FILE CONTROLLER (SAFETY SYSTEM)
% =========================================================
if CONTINUE_TRAINING
    if isfile(model_filename)
        fprintf('>>> MODE: CONTINUE TRAINING. LOADING %s\n', model_filename);
        data = load(model_filename);
        layers = data.layer;
    else
        error('Model is not found! Check if filename is correct or start from scratch CONTINUE_TRAINING = false');
    end
else
    fprintf('>>> MODE: NEW MODEL.\n');
    
    % make a backup
    if isfile(model_filename)
        t_str = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        backup_name = ['backup_', t_str, '.mat'];
        backup_path = fullfile(paths.models, backup_name);
        movefile(model_filename, backup_path);
        fprintf('!!! Old model was renamed and moved to: %s\n', backup_path);
    end
    
    % init
    init_model(model_filename, architecture);
    data = load(model_filename);
    layers = data.layer;
end

% =========================================================
% 3. TRAINING START
% =========================================================

% --- PREPARE EXPERIMENT FOLDER ---
t_str = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
current_run_folder = fullfile(paths.results, ['run_', t_str]);

if ~exist(current_run_folder, 'dir'), mkdir(current_run_folder); end

% --- PREPARE ANIMATION DATA (Fixed 2000 points) ---
num_anim = 2000;
anim_idx = randperm(size(X_train, 2), num_anim);

anim_data = struct();
anim_data.X = X_train(:, anim_idx)'; % [2000 x 784]
anim_data.Y = Y_train(anim_idx);     % [2000 x 1] (Colors)


[layers, history] = train_smart(X_train, Y_train, layers, cfg, current_run_folder, anim_data);


% =========================================================
% 4. DEMONSTRATION
% =========================================================

fprintf('\n>>> REPORT GENERATION and CSV...\n');

% --- A. Reconstruction (how it was -> how it now) ---
figure('Name', 'Reconstruction Check');
for i = 1:5
    idx = randi(size(X_train, 2));
    img = X_train(:, idx)';
    
    [out, ~] = model_forward(img, layers);
    
    subplot(2, 5, i); imshow(reshape(img, 28, 28)); title('Original');
    subplot(2, 5, i+5); imshow(reshape(out, 28, 28)); title('Autoencoder');
end


% --- B. Latent space (and export for R) ---
% we pick 2000 random images and get latent coordinates for them

num_samples = 2000;
indices = randperm(size(X_train, 2), num_samples);


export_data = zeros(num_samples, 3); % [x, y, label]

fprintf('Processing %d examples for the latent space...\n', num_samples);

for k = 1:num_samples
    idx = indices(k);
    img = X_train(:, idx)';
    label = Y_train(idx);
    
    [~, latent] = model_forward(img, layers);
    
    export_data(k, 1) = latent(1); % Coord X
    export_data(k, 2) = latent(2); % Coord Y
    export_data(k, 3) = label;     % Digit (0-9)
end

% --- C. Draw here in MATLAB ---
figure('Name', 'Latent Space');
% draw dots: X, Y, size(15), color(label)
scatter(export_data(:,1), export_data(:,2), 15, export_data(:,3), 'filled');
colormap(jet(10));
c = colorbar;      
c.Ticks = 0:9;    
title('Latent Space');
xlabel('Neuron X'); ylabel('Neuron Y');
grid on;

% --- D. Save Final CSV ---
csv_filename = fullfile(current_run_folder, 'final_latent.csv');

fid = fopen(csv_filename, 'w');
fprintf(fid, 'latent_x,latent_y,label\n');
fclose(fid);

writematrix(export_data, csv_filename, 'WriteMode', 'append');

fprintf('>>> DONE! All results saved in: %s\n', current_run_folder);