clear; clc; close all;



%find the current dir -> root -> scr
script_path = fileparts(mfilename('fullpath'));
project_root = fileparts(script_path);
addpath(fullfile(project_root, 'src'));

data_dir = fullfile(project_root, 'data');
models_dir = fullfile(project_root, 'models');

img_file = fullfile(data_dir, 'train-images.idx3-ubyte');
lbl_file = fullfile(data_dir, 'train-labels.idx1-ubyte');
model_file = fullfile(models_dir, 'best_model.mat');


% load
if ~exist('X_train', 'var')
    fprintf('Загружаю данные из: %s\n', data_dir);
    X_train = loadMNISTImages(img_file);
    Y_train = loadMNISTLabels(lbl_file);
end

% load best model
load(model_file);  % contains var layer
layers = layer;

% 2. Сначала строим карту (чтобы видеть куда тыкать)
fprintf('Генерирую карту...\n');
num_samples = 2000;
idx = randperm(size(X_train, 2), num_samples);
data_batch = X_train(:, idx)';
labels_batch = Y_train(idx);

[~, coords] = model_forward(data_batch, layers);

% Рисуем главное окно
f = figure('Name', 'Interactive Latent Space', 'Position', [100, 100, 1000, 600]);

% Слева - карта
subplot(1, 2, 1);
scatter(coords(:,1), coords(:,2), 15, labels_batch, 'filled');
colormap(jet(10));
title('Кликни левой кнопкой мыши в ДВЕ точки!');
xlabel('X'); ylabel('Y');
grid on; hold on;

% 3. ИНТЕРАКТИВ (Бесконечный цикл)
while true
    try
        % Ждем клика юзера (2 точки)
        [x, y] = ginput(2);
        
        % Рисуем линию, чтобы показать путь
        plot(x, y, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'w');
        
        % Точки начала и конца
        start_point = [x(1), y(1)];
        end_point   = [x(2), y(2)];
        
        % Генерируем морфинг
        steps = 10;
        morph_strip = [];
        
        for k = 1:steps
            t = (k-1)/(steps-1);
            % Линейная интерполяция координат
            curr_point = start_point * (1-t) + end_point * t;
            
            % Декодируем точку в картинку
            img = model_decoder(curr_point, layers);
            
            % Склеиваем в полоску
            digit = reshape(img, 28, 28);
            morph_strip = [morph_strip, digit, ones(28, 1)]; % + разделитель
        end
        
        % Показываем результат справа
        subplot(1, 2, 2);
        imshow(morph_strip);
        title(sprintf('Морфинг: [%.1f, %.1f] -> [%.1f, %.1f]', ...
            start_point(1), start_point(2), end_point(1), end_point(2)));
        
    catch
        break;
    end
end