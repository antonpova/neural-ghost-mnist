function labels = loadMNISTLabels(filename)
    % Открываем так же с 'b'
    fid = fopen(filename, 'r', 'b');
    
    if fid == -1
        error('Не удалось открыть файл меток!');
    end
    
    magicNumber = fread(fid, 1, 'int32');
    if magicNumber ~= 2049
        error('Это не файл меток MNIST!');
    end
    
    numLabels = fread(fid, 1, 'int32');
    
    % Просто читаем байты подряд. Каждый байт - это цифра (0, 1, .. 9)
    labels = fread(fid, inf, 'unsigned char');
    
    fclose(fid);
    
    % Конвертируем в double для удобства работы
    labels = double(labels);
end