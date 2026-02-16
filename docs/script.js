let model = null;
let pointsData = null;
let pointA = null;
let pointB = null;
let plotElement = null; // Ссылка на объект графика Plotly

const canvas = document.getElementById('digitCanvas');
const ctx = canvas.getContext('2d');
const scale = 5; 

// Границы по умолчанию (обновятся при отрисовке)
let xRange = [-15, 15]; 
let yRange = [-15, 15];

const MATLAB_JET = [
    "#0000AA", "#0000FF", "#0055FF", "##00AAFF", "##00FFFF", 
    "#55FFAA", "#AAFF55", "#FFFF00", "#FFAA00", "#FF5500"
];
//const MATLAB_JET = [
//    "#00008F", "#0000FF", "#007FFF", "#00FFFF", "#7FFF7F", 
//    "#FFFF00", "#FF7F00", "#FF0000", "#7F0000", "#400000"
//];

// === 1. МАТЕМАТИКА ===
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function relu(x) { return Math.max(0, x); }

function matMulVector(vec, mat) {
    const n = vec.length;
    const m = mat[0].length;
    const result = new Array(m).fill(0);
    for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
            sum += vec[i] * mat[i][j];
        }
        result[j] = sum;
    }
    return result;
}

function addBias(vec, bias) {
    return vec.map((val, i) => val + bias[i]);
}

function runDecoder(latentVector) {
    if (!model) return null;
    let A = latentVector;
    const startIndex = model.decoder_start_index; 
    const layers = model.layers;

    for (let i = startIndex; i < layers.length; i++) {
        const layer = layers[i];
        let Z = matMulVector(A, layer.W);
        Z = addBias(Z, layer.b);
        if (layer.act === 'relu') A = Z.map(relu);
        else if (layer.act === 'sigmoid') A = Z.map(sigmoid);
        else A = Z;
    }
    return A;
}

// === 2. ОТРИСОВКА ===
function drawDigitToContext(pixelVector, targetCtx, width, height) {
    const pxScale = width / 28; 
    targetCtx.clearRect(0, 0, width, height);
    for (let i = 0; i < 784; i++) {
        const val = pixelVector[i]; 
        const color = Math.floor(val * 255);
        // Поворот на 90 градусов (как в MATLAB reshape order)
        const col = Math.floor(i / 28);
        const row = i % 28;
        const x = col * pxScale;
        const y = row * pxScale;
        targetCtx.fillStyle = `rgb(${color}, ${color}, ${color})`;
        targetCtx.fillRect(x, y, pxScale, pxScale);
    }
}

// === 3. ИНТЕРАКТИВ (ИСПРАВЛЕН РАСЧЕТ КООРДИНАТ) ===
let lastProcessedX = 0;
let lastProcessedY = 0;

function handleMouseMove(event) {
    if (!model || !plotElement) return;

    // Получаем реальные параметры осей из Plotly
    // _fullLayout содержит вычисленные размеры (в пикселях)
    const xaxis = plotElement._fullLayout.xaxis;
    const yaxis = plotElement._fullLayout.yaxis;
    const rect = plotElement.getBoundingClientRect();

    // Координаты мыши относительно левого верхнего угла DIV-а графика
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // --- МАГИЯ ТОЧНЫХ КООРДИНАТ ---
    // xaxis._offset = отступ оси X слева в пикселях
    // xaxis._length = длина оси X в пикселях
    
    // Проверяем, находится ли мышь внутри области осей
    if (mouseX < xaxis._offset || mouseX > xaxis._offset + xaxis._length ||
        mouseY < yaxis._offset || mouseY > yaxis._offset + yaxis._length) {
        return; // Мышь на полях или легенде, игнорируем
    }

    // Переводим пиксели в проценты (0..1)
    const xPct = (mouseX - xaxis._offset) / xaxis._length;
    // Y инвертирован (экран вниз, график вверх)
    const yPct = 1 - ((mouseY - yaxis._offset) / yaxis._length);

    // Переводим проценты в значения данных (используя текущий Range)
    // xRange обновляется при зуме!
    const dataX = xRange[0] + xPct * (xRange[1] - xRange[0]);
    const dataY = yRange[0] + yPct * (yRange[1] - yRange[0]);

    // Оптимизация (не считать, если сдвиг микроскопический)
    const dist = Math.sqrt((dataX - lastProcessedX)**2 + (dataY - lastProcessedY)**2);
    // Порог зависит от масштаба (чтобы работало и при зуме)
    const threshold = (xRange[1] - xRange[0]) / 200; 
    if (dist < threshold) return;

    lastProcessedX = dataX;
    lastProcessedY = dataY;

    // Обновляем UI
    document.getElementById('coordsDisplay').innerText = `X: ${dataX.toFixed(2)}, Y: ${dataY.toFixed(2)}`;
    const pixels = runDecoder([dataX, dataY]);
    drawDigitToContext(pixels, ctx, canvas.width, canvas.height);
}

// === 4. КЛИКИ ===
function handlePlotClick(event) {
    const currentLatent = [lastProcessedX, lastProcessedY];

    switch (event.button) {
        case 0: // LEFT CLICK
            pointA = currentLatent;
            break;
        case 1: // MIDDLE CLICK
            // Сброс камеры через API (без костылей)
            Plotly.relayout(plotElement, {
                'xaxis.autorange': true,
                'yaxis.autorange': true
            });
            return;
        case 2: // RIGHT CLICK
            pointB = currentLatent;
            break;
    }

    // Обновляем всё остальное
    updateStatusText();
    updatePlotVisuals(); 

    if (pointA && pointB) {
        triggerAutoUpdate();
    }
}
function updateStatusText() {
    const el = document.getElementById('morphStatus');
    let text = "";
    if (pointA) text += `<span style="color:#00ffcc">Start: [${pointA[0].toFixed(2)}, ${pointA[1].toFixed(2)}]</span><br>`;
    else text += "Set Start (Left Click)<br>";
    
    if (pointB) text += `<span style="color:#ff0066">End: [${pointB[0].toFixed(2)}, ${pointB[1].toFixed(2)}]</span>`;
    else text += "Set End (Right Click)";
    
    el.innerHTML = text;
}

function updatePlotVisuals() {
    const update = { x: [[], [], []], y: [[], [], []] };

    // Trace 10 (Line), 11 (Start), 12 (End)
    if (pointA && pointB) {
        update.x[0] = [pointA[0], pointB[0]];
        update.y[0] = [pointA[1], pointB[1]];
    }
    if (pointA) {
        update.x[1] = [pointA[0]]; update.y[1] = [pointA[1]];
    }
    if (pointB) {
        update.x[2] = [pointB[0]]; update.y[2] = [pointB[1]];
    }
    
    Plotly.restyle('scatterPlot', update, [10, 11, 12]);
}

function triggerAutoUpdate() {
    if (!pointA || !pointB) return;
    generateMorphStrip(pointA, pointB);
    generateGif(pointA, pointB);
}

document.getElementById('gifFrames').addEventListener('change', triggerAutoUpdate);
document.getElementById('gifInterval').addEventListener('change', triggerAutoUpdate);


// === 5. МОРФИНГ (ПОЛОСКА) ===
function generateMorphStrip(start, end) {
    const container = document.getElementById('morphStrip');
    container.innerHTML = ''; 
    const steps = 10;
    for (let k = 0; k < steps; k++) {
        const t = k / (steps - 1);
        const curX = start[0] * (1 - t) + end[0] * t;
        const curY = start[1] * (1 - t) + end[1] * t;
        const pixels = runDecoder([curX, curY]);
        const stepCanvas = document.createElement('canvas');
        stepCanvas.width = 28; stepCanvas.height = 28;
        drawDigitToContext(pixels, stepCanvas.getContext('2d'), 28, 28);
        container.appendChild(stepCanvas);
    }
}

// === 6. ГЕНЕРАЦИЯ GIF ===
function generateGif(start, end) {
    const resDiv = document.getElementById('gifResult');
    resDiv.innerHTML = '<span style="color: #8892b0; font-size: 0.8rem">Generating...</span>';

    const framesCount = parseInt(document.getElementById('gifFrames').value) || 30;
    const interval = parseFloat(document.getElementById('gifInterval').value) || 0.1;
    
    const frames = [];
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 140; tempCanvas.height = 140;
    const tempCtx = tempCanvas.getContext('2d');

    for (let k = 0; k < framesCount; k++) {
        const t = k / (framesCount - 1);
        const curX = start[0] * (1 - t) + end[0] * t;
        const curY = start[1] * (1 - t) + end[1] * t;
        const pixels = runDecoder([curX, curY]);
        
        tempCtx.fillStyle = "#000"; 
        tempCtx.fillRect(0,0,140,140);
        drawDigitToContext(pixels, tempCtx, 140, 140);
        frames.push(tempCanvas.toDataURL());
    }

    gifshot.createGIF({
        images: frames,
        interval: interval,
        gifWidth: 140,
        gifHeight: 140,
        numFrames: framesCount,
    }, function(obj) {
        if(!obj.error) {
            const image = obj.image;
            const fName = `morph_[${start[0].toFixed(2)},${start[1].toFixed(2)}]_to_[${end[0].toFixed(2)},${end[1].toFixed(2)}].gif`;
            resDiv.innerHTML = `<img src="${image}" alt="Morph GIF"><a href="${image}" download="${fName}" class="download-link">Download GIF</a>`;
        } else {
            resDiv.innerHTML = "Error generating GIF";
        }
    });
}

// === 7. ИНИЦИАЛИЗАЦИЯ ===
function initPlot() {
    const traces = [];
    
    // Точки цифр 0-9
    for (let digit = 0; digit <= 9; digit++) {
        const indices = [];
        pointsData.labels.forEach((lbl, idx) => { if (lbl === digit) indices.push(idx); });
        
        traces.push({
            x: indices.map(i => pointsData.x[i]),
            y: indices.map(i => pointsData.y[i]),
            mode: 'markers',
            type: 'scatter',
            name: `Digit ${digit}`,
            marker: { size: 5, color: MATLAB_JET[digit], opacity: 0.9 },
            hoverinfo: 'none'
        });
    }

    // Служебные трейсы (Путь, Старт, Конец)
    // Trace 10
    traces.push({ x: [], y: [], mode: 'lines', name: 'Path', line: { color: 'white', width: 2, dash: 'dot' }, hoverinfo: 'none', showlegend: false });
    // Trace 11
    traces.push({ x: [], y: [], mode: 'markers', name: 'Start', marker: { size: 12, color: '#00ffcc', line: {color: 'white', width: 2} }, hoverinfo: 'none', showlegend: false });
    // Trace 12
    traces.push({ x: [], y: [], mode: 'markers', name: 'End', marker: { size: 12, color: '#ff0066', line: {color: 'white', width: 2} }, hoverinfo: 'none', showlegend: false });

    const layout = {
        paper_bgcolor: '#050a14',
        plot_bgcolor: '#050a14',
        xaxis: { showgrid: true, gridcolor: '#1f2d40', zeroline: false, fixedrange: false },
        yaxis: { showgrid: true, gridcolor: '#1f2d40', zeroline: false, fixedrange: false },
        margin: { t: 20, l: 50, r: 20, b: 30 },
        hovermode: false,
        showlegend: true,
        legend: { font: { color: '#8892b0' }, bgcolor: 'rgba(0,0,0,0)' },
        
        // Отключаем стандартный дабл-клик через layout
        doubleClick: false, 
        // Устанавливаем режим перетаскивания (зум рамкой)
        dragmode: 'zoom'
    };
    
    const config = { 
        responsive: true, 
        displayModeBar: true, 
        // Убираем лишние кнопки из панели, оставляем только нужные
        modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d'],
        // Отключаем зум колесиком (чтобы оно работало только на наш ресет)
        scrollZoom: false,
        // Отключаем встроенные подсказки Plotly
        showTips: false,
        // Явно отключаем дабл-клик
        doubleClick: false
    };
    
    Plotly.newPlot('scatterPlot', traces, layout, config).then(gd => {
        plotElement = gd; // Сохраняем ссылку глобально
        
        // Инициализация границ
        xRange = gd._fullLayout.xaxis.range;
        yRange = gd._fullLayout.yaxis.range;

        const container = document.getElementById('scatterPlotContainer');
        container.addEventListener('mousemove', handleMouseMove);
        container.addEventListener('mousedown', handlePlotClick);
        container.addEventListener('contextmenu', e => e.preventDefault());
        
        // Слушатель изменения зума/камеры (relayout)
        gd.on('plotly_relayout', function(eventdata){
            // Если оси изменились (zoom/pan/autoscale)
            if(gd._fullLayout.xaxis && gd._fullLayout.yaxis) {
                xRange = gd._fullLayout.xaxis.range;
                yRange = gd._fullLayout.yaxis.range;
                console.log("Zoom updated:", xRange, yRange);
            }
        });
    });
}

async function init() {
    try {
        const modelRes = await fetch('data/model.json');
        model = await modelRes.json();
        const pointsRes = await fetch('data/points.json');
        pointsData = await pointsRes.json();
        initPlot();
    } catch (e) {
        console.error(e);
        document.getElementById('morphStatus').innerText = "Error loading data!";
    }
}

init();