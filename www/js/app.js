function main()
{
    fetch('training_data.json')
        .then(response => response.json())
        .then(data => run(data));

}
function run(data)
{


    const canvas = document.getElementById('canvasShapes');
    const normedCanvas = document.getElementById('resizedCanvas');
    const drawer = new SimpleDrawer(canvas, normedCanvas);

    const btnAdd = document.getElementById('btnAdd');

    const labelMapping = {
        "line" : [1, 0, 0, 0], 
        "triangle" :   [0, 1, 0, 0],
        "square" :  [0, 0, 1, 0],
        "circle" :   [0, 0, 0, 1]
    }
    const reverseLabels = ["line", "triangle", "square", "circle"];

    const trainingInputs = data.trainingInputs;
    const trainingOutputs = data.trainingLabels.map((o) => labelMapping[o]);
    const trainingLabels = data.trainingLabels;
    const predElms = [];
    initTrainingList(trainingInputs, trainingLabels, 30, predElms);

    let model = null;
    btnAdd.onclick = function()
    {
        
        const r = document.querySelector('input[name="shape"]:checked');
        if (r !== null)
        {
            const inputFeatures = drawer.features;
            const label = labelMapping[r.value];
            trainingLabels.push(r.value);

            const liElm = document.createElement('li');
            document.getElementById('trainingSet').appendChild(liElm);
            liElm.classList.add('example');

            const canvas = document.createElement('canvas');
            canvas.classList.add('shape')
            liElm.appendChild(canvas);

            canvas.width = drawer.normedCanvas.width;
            canvas.height = drawer.normedCanvas.height;
            canvas.getContext('2d').drawImage(drawer.normedCanvas, 0, 0);

            const labelElm = document.createElement('span');
            liElm.appendChild(labelElm);
            labelElm.innerHTML = r.value;

            const predElm = document.createElement('span');
            liElm.appendChild(predElm);
            predElms.push(predElm);

            trainingInputs.push(inputFeatures)
            trainingOutputs.push(label)

        }
        drawer.reset();
        console.log(JSON.stringify({
            "trainingInputs" : trainingInputs,
            "trainingLabels" : trainingLabels
        }));
    }

    const progressBar = document.querySelector('.js-bar');

    const btnTrain = document.getElementById('btnTrain');
    btnTrain.onclick = async function()
    {
        model = buildModel();
        btnTrain.disabled = true;
        btnAdd.disabled = true;
        btnPredict.disabled = true;
        progressBar.style.left = '-100%';
        await model.fit(tf.tensor(trainingInputs), tf.tensor(trainingOutputs), {
            epochs: 500,
            shuffle: true,
            callbacks: {
                onEpochEnd: function(b, l) {
                    const perc = Math.ceil(b * 100 / 500);
                    progressBar.style.left = `-${100 - perc}%`;
                }
            }
        });

        btnTrain.disabled = false;
        btnAdd.disabled = false;
        btnPredict.disabled = false;

        const preds = model.predict(tf.tensor(trainingInputs));
        const argmax = preds.argMax(1).arraySync();
            
        predElms.forEach((elm, i) => {
            elm.innerHTML = reverseLabels[ argmax[i] ];
        });

        visualizeWeights(model);
    }

    const btnPredict = document.getElementById('btnPredict');
    btnPredict.onclick = function()
    {
        const inputFeatures = drawer.features;
        const preds = model.predict(tf.tensor([inputFeatures]));
        const argmax = preds.argMax(1).arraySync();
        const label = reverseLabels[argmax[0]];
        document.getElementById('prediction').innerHTML = label;
        
    }

}

function initTrainingList(trainingInputs, trainingLabels, sizePx, predElms)
{
    for (let i = 0; i < trainingInputs.length; ++i)
    {
        const input = trainingInputs[i];
        const output = trainingLabels[i];

        const liElm = document.createElement('li');
        document.getElementById('trainingSet').appendChild(liElm);
        liElm.classList.add('example');

        const canvas = document.createElement('canvas');
        canvas.classList.add('shape');
        liElm.appendChild(canvas);

        canvas.width = sizePx;
        canvas.height = sizePx;

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, sizePx, sizePx);
        const imgd = ctx.getImageData(0, 0, sizePx, sizePx);
        const pix = imgd.data;
        for (var j = 0; j < pix.length; j += 4)
        {
            const ison = input[j / 4] == 1;

            pix[j] = (ison) ? 0 : 255;
            pix[j+1] = (ison) ? 0 : 255;
            pix[j+2] = (ison) ? 0 : 255;
        }
        ctx.putImageData(imgd, 0, 0);
        
        const labelElm = document.createElement('span');
        liElm.appendChild(labelElm);
        labelElm.innerHTML = output;

        const predElm = document.createElement('span');
        liElm.appendChild(predElm);
        predElms.push(predElm);   

    }
}

function buildModel()
{
    const model = tf.sequential();
    const hiddenLayer = tf.layers.dense({ 
        inputShape: [900],
        units: 10,
        activation: 'tanh',
        useBias: true
    });
    const outputLayer = tf.layers.dense({ 
        units: 4,
        useBias: true,
        activation: 'softmax'
    });

    model.add(hiddenLayer);
    model.add(outputLayer);

    const optimizer = tf.train.adam(0.0001);
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    return model;
}

function visualizeWeights(layer)
{
    const weights = layer.getWeights();
    const pixelWeights = weights[0].transpose().reshape([10, 30, 30]);
    
    const maxVals = pixelWeights.max([1, 2]).arraySync();
    const minVals = pixelWeights.min([1, 2]).arraySync();

    const outputSize = 120;
    const pixelSize = outputSize / 30;
    
    document.getElementById('weights').innerHTML = "";

    const W = pixelWeights.arraySync();

    for (let unit = 0; unit < 5; ++unit)
    {
        const canvas = document.createElement('canvas');
        canvas.width = outputSize;
        canvas.height = outputSize;
        document.getElementById('weights').appendChild(canvas);

        const ctx = canvas.getContext('2d');

        const w = W[unit];
        for (let i = 0; i < w.length; ++i)
        {
            for (let j = 0; j < w[i].length; ++j)
            {
                const normedWeight = (w[i][j] - minVals[unit]) / (maxVals[unit] - minVals[unit]);
                ctx.fillStyle = `hsla(43, 100%, 50%, ${normedWeight})`;
                ctx.fillRect(j * pixelSize, outputSize - i * pixelSize, pixelSize, pixelSize);
            }
        }
    }
    

}

//
// Based on https://developer.mozilla.org/en-US/docs/Web/API/Element/mousemove_event
//
class SimpleDrawer
{
    constructor(canvas, normedCanvas)
    {
        this.canvas = canvas;
        this.normedCanvas = normedCanvas;

        this.ctx = canvas.getContext('2d');
        
        this.reset();

        canvas.onmousedown = (e) => {
            this.x = e.offsetX;
            this.y = e.offsetY;
            this.drawing = true;
            this._updateExtent(this.x, this.y);
        };

        canvas.onmousemove = (e) => {
            if (this.drawing) {
                this._drawLine(e.offsetX, e.offsetY);
                this.x = e.offsetX;
                this.y = e.offsetY;
                this._updateExtent(this.x, this.y);
            }
        };

        canvas.onmouseup = (e) => {
            if (this.drawing) {
                this._drawLine(e.offsetX, e.offsetY);
                this._updateExtent(e.offsetX, e.offsetY);
                this.x = 0;
                this.y = 0;
                this.drawing = false;
                this.getBWPixels(normedCanvas.width);

            }
        };
    }

    reset()
    {
        this.drawing = false;
        
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const normedCtx = this.normedCanvas.getContext('2d');
        normedCtx.fillStyle = 'white';
        normedCtx.fillRect(0, 0, this.normedCanvas.width, this.normedCanvas.height);

        this.x = 0;
        this.y = 0;

        this.minX = Infinity;
        this.maxX = -Infinity;
        this.minY = Infinity;
        this.maxY = -Infinity;
    }
    _updateExtent(x, y)
    {
        this.minX = Math.min(x, this.minX);
        this.maxX = Math.max(x, this.maxX);
        this.minY = Math.min(y, this.minY);
        this.maxY = Math.max(y, this.maxY);
    }

    _drawLine(toX, toY)
    {
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 5;
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        ctx.closePath();
    }

    getBWPixels(targetSizePx)
    {
        const ctx = this.ctx;
        const paddingPx = 7;

        const drawingSizePx = targetSizePx - paddingPx;

        const offscreenCanvas = this.normedCanvas;
        offscreenCanvas.width = targetSizePx;
        offscreenCanvas.height = targetSizePx;

        const offscreenCtx = offscreenCanvas.getContext('2d');
        offscreenCtx.fillStyle = 'white';
        offscreenCtx.fillRect(0, 0, targetSizePx, targetSizePx);
        offscreenCtx.imageSmoothingQuality = "high";

        offscreenCtx.drawImage(this.canvas, this.minX, this.minY, 
            this.maxX - this.minX, 
            this.maxY - this.minY,
            (targetSizePx-drawingSizePx)/2, 
            (targetSizePx-drawingSizePx)/2,
            drawingSizePx,
            drawingSizePx
        );

        const imgd = offscreenCtx.getImageData(0, 0, targetSizePx, targetSizePx);
        const pix = imgd.data;
        
        const output = [];
        output.length = targetSizePx * targetSizePx;

        for (var i = 0; i < pix.length; i += 4)
        {
            const pv = (pix[i] < 255 || pix[i+1] < 255 || pix[i+2] < 255) ? 1 : 0;
            output[i / 4] = pv;

            pix[i] = (pv === 1) ? 0 : 255;
            pix[i+1] = (pv === 1) ? 0 : 255;
            pix[i+2] = (pv === 1) ? 0 : 255;
        }
        
        offscreenCtx.putImageData(imgd, 0, 0);

        this.features = output;
    }


}

window.onload = main;