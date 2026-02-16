clear; clc;

% 1. Настройки
% Используем fullfile для правильных путей
model_path = fullfile('models', 'best_model.mat');
output_dir = fullfile('web_project', 'data');

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% 2. Грузим модель
if ~isfile(model_path)
    error('Сначала обучи модель! Файл %s не найден.', model_path);
end
loaded = load(model_path);
layers = loaded.layer;

fprintf('Экспорт весов модели...\n');

% 3. Подготовка структуры для JSON
decoder_start_idx = 0;
web_layers = {};

for i = 1:length(layers)
    % --- ИСПРАВЛЕНИЕ: БЕРЕМ ДАННЫЕ НАПРЯМУЮ ---
    % У нас обычные double, никакие extractdata не нужны
    W = layers(i).W;
    b = layers(i).b;
    act = layers(i).act;
    % -------------------------------------------
    
    % Запоминаем структуру слоя
    layer_struct = struct();
    layer_struct.W = W;
    layer_struct.b = b;
    layer_struct.act = act;
    
    web_layers{i} = layer_struct;
    
    % Ищем начало декодера (вход = 2 нейрона)
    if size(W, 1) == 2
        decoder_start_idx = i - 1; % JS индексы с 0
    end
end

model_json = struct();
model_json.layers = web_layers;
model_json.decoder_start_index = decoder_start_idx;

% Сохраняем модель
json_path = fullfile(output_dir, 'model.json');
fid = fopen(json_path, 'w');
fprintf(fid, '%s', jsonencode(model_json));
fclose(fid);


% 4. Экспорт точек для графика
fprintf('Экспорт точек латентного пространства...\n');

% Подключаем src, чтобы видеть функцию model_forward
addpath('src'); 

% Загружаем данные
if ~exist('X_train', 'var')
    X_train = loadMNISTImages(fullfile('data', 'train-images.idx3-ubyte'));
    Y_train = loadMNISTLabels(fullfile('data', 'train-labels.idx1-ubyte'));
end

% Берем 2000 точек
num_samples = 2000;
idx = randperm(size(X_train, 2), num_samples);
subset_X = X_train(:, idx)';
subset_Y = Y_train(idx);

% Прогоняем
[~, latents] = model_forward(subset_X, layers); 

% Собираем структуру
points_data = struct();
points_data.x = latents(:, 1);
points_data.y = latents(:, 2);
points_data.labels = subset_Y;

points_path = fullfile(output_dir, 'points.json');
fid = fopen(points_path, 'w');
fprintf(fid, '%s', jsonencode(points_data));
fclose(fid);

fprintf('ГОТОВО! Данные лежат в папке %s\n', output_dir);