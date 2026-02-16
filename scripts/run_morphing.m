clear; clc; close all;

% Загрузка
if ~exist('X_train', 'var')
    X_train = loadMNISTImages('train-images.idx3-ubyte');
end
load('best_model.mat'); % Загрузит переменную 'layer'
layers = layer;

% 1. Выбираем две случайные картинки
idx1 = randi(size(X_train, 2));
idx2 = randi(size(X_train, 2));

img1 = X_train(:, idx1)';
img2 = X_train(:, idx2)';

% 2. Узнаем их координаты (Энкодер)
[~, lat1] = model_forward(img1, layers);
[~, lat2] = model_forward(img2, layers);

fprintf('Морфинг между точками: [%.2f, %.2f] -> [%.2f, %.2f]\n', ...
    lat1(1), lat1(2), lat2(1), lat2(2));

% 3. Рисуем
steps = 12; % Сколько кадров в анимации
figure('Name', 'Morphing Magic', 'Color', 'w');

for k = 1:steps
    % t идет от 0.0 до 1.0
    t = (k-1) / (steps-1);
    
    % Интерполяция координат (Математика!)
    lat_current = lat1 * (1 - t) + lat2 * t;
    
    % !!! ВОТ ОН, РАЗРЕЗ МОДЕЛИ !!!
    % Мы скармливаем выдуманные координаты Декодеру
    gen_img = model_decoder(lat_current, layers);
    
    % Рисуем
    subplot(1, steps, k);
    imshow(reshape(gen_img, 28, 28));
    title(sprintf('%.1f', t));
end