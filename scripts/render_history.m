clear; clc; close all;

% --- НАСТРОЙКИ ---
% Укажи путь к папке эксперимента, который хочешь перерисовать
run_folder = '..\experiments\run_20251122_224819'; % folder for rendering

csv_file = fullfile(run_folder, 'trajectories.csv');
output_folder = fullfile(run_folder, 'animation_hd'); % new folder

if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% --- 1. Читаем данные ---
fprintf('Читаю лог траекторий: %s ...\n', csv_file);
% Пропускаем 1 строку заголовка
data = readmatrix(csv_file, 'NumHeaderLines', 1);

% Формат CSV: [Epoch, PointID, X, Y, Label]
epochs = data(:, 1);
Xs = data(:, 3);
Ys = data(:, 4);
Labels = data(:, 5);

unique_epochs = unique(epochs);

% --- 2. Вычисляем Глобальные Границы (Камера) ---
% Мы ищем минимальные и максимальные X и Y за ВСЕ время обучения,
% чтобы камера стояла на месте, а точки двигались.
x_min = min(Xs); x_max = max(Xs);
y_min = min(Ys); y_max = max(Ys);

% Добавим отступ (margin) 10%, чтобы точки не прилипали к краям
margin_x = (x_max - x_min) * 0.1;
margin_y = (y_max - y_min) * 0.1;

global_xlim = [x_min - margin_x, x_max + margin_x];
global_ylim = [y_min - margin_y, y_max + margin_y];

fprintf('Границы сцены: X[%.1f, %.1f], Y[%.1f, %.1f]\n', ...
    global_xlim(1), global_xlim(2), global_ylim(1), global_ylim(2));

% --- 3. Рендеринг кадров ---
fig = figure('Visible', 'off', 'Position', [100, 100, 800, 800]); % Большое разрешение

fprintf('Начинаю рендеринг %d кадров...\n', length(unique_epochs));

for i = 1:length(unique_epochs)
    ep = unique_epochs(i);
    
    % Берем данные только для текущей эпохи
    mask = (epochs == ep);
    current_X = Xs(mask);
    current_Y = Ys(mask);
    current_L = Labels(mask);
    
    clf(fig); % Очищаем фигуру
    
    % Рисуем
    scatter(current_X, current_Y, 20, current_L, 'filled');
    colormap(jet(10)); 
    caxis([0 9]); % Фиксируем цвета
    
    % Красота
    grid on;
    set(gca, 'Color', [0.1 0.1 0.1]); % Темный фон (по желанию)
    xlabel('Latent X'); ylabel('Latent Y');
    title(sprintf('Epoch: %d', ep), 'Color', 'w', 'FontSize', 14);
    
    % !!! ПРИМЕНЯЕМ ГЛОБАЛЬНЫЕ ГРАНИЦЫ !!!
    xlim(global_xlim);
    ylim(global_ylim);
    
    % Сохраняем
    filename = fullfile(output_folder, sprintf('frame_%04d.png', ep));
    saveas(fig, filename);
    
    if mod(i, 5) == 0
        fprintf('Отрисовано эпох: %d / %d\n', i, length(unique_epochs));
    end
end

fprintf('Готово! Кадры лежат в: %s\n', output_folder);