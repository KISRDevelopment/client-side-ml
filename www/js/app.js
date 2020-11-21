
// one hot encoding of labels
const labelMapping = {
        "line" : [1, 0, 0, 0], 
        "triangle" :   [0, 1, 0, 0],
        "square" :  [0, 0, 1, 0],
        "circle" :   [0, 0, 0, 1]
}

const reverseLabels = ["line", "triangle", "square", "circle"];

const CFG = {
    "epochs" : 100,
    "inputPx" : 30
}

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

    const trainingInputs = data.trainingInputs;
    const trainingOutputs = data.trainingLabels.map((o) => labelMapping[o]);
    const trainingLabels = data.trainingLabels;
    const predElms = [];
    
    let model = buildModel();
    initTrainingList(trainingInputs, trainingLabels);
    updatePredictions(trainingInputs, model);

    btnAdd.onclick = function()
    {
        
        const r = document.querySelector('input[name="shape"]:checked');
        if (r !== null)
        {
            const inputFeatures = drawer.features;
            const label = labelMapping[r.value];
            trainingLabels.push(r.value);

            addExample(inputFeatures, r.value);

            trainingInputs.push(inputFeatures)
            trainingOutputs.push(label)

        }
        drawer.reset();
        // console.log(JSON.stringify({
        //     "trainingInputs" : trainingInputs,
        //     "trainingLabels" : trainingLabels
        // }));
    }

    const progressBar = document.querySelector('.js-bar');
    
    const btnTrain = document.getElementById('btnTrain');
    btnTrain.onclick = async function()
    {
        model = buildModel();

        // disable buttons
        btnTrain.disabled = true;
        btnAdd.disabled = true;
        btnPredict.disabled = true;

        // init progress bar to 0
        progressBar.style.left = '-100%';

        // train the model
        await model.fit(tf.tensor(trainingInputs), tf.tensor(trainingOutputs), {
            epochs: CFG.epochs,
            shuffle: true,
            callbacks: {
                onEpochEnd: function(b, l) {
                    const perc = Math.ceil(b * 100 / CFG.epochs);
                    progressBar.style.left = `-${100 - perc}%`;
                }
            }
        });

        // re-enable buttons
        btnTrain.disabled = false;
        btnAdd.disabled = false;
        btnPredict.disabled = false;

        updatePredictions(trainingInputs, model);
        
    }

    const btnPredict = document.getElementById('btnPredict');
    const outputCanvas = document.getElementById('vectorShapes');
    const outputCtx = outputCanvas.getContext('2d');
    btnPredict.onclick = function()
    {
        const inputFeatures = drawer.features;
        const preds = model.predict(tf.tensor([inputFeatures]));
        const argmax = preds.argMax(1).arraySync();
        let label = reverseLabels[argmax[0]];
        document.getElementById('prediction').innerHTML = label;
        
        outputCtx.fillStyle = 'white';
        outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);

        const width = drawer.maxX - drawer.minX;
        const height = drawer.maxY - drawer.minY;

        outputCtx.strokeStyle = 'purple';
        outputCtx.lineWidth = 5;

        label = "triangle";

        outputCtx.beginPath();
        if (label === "square")
        {
            
            outputCtx.rect(drawer.minX, drawer.minY, width, height);
            
        }
        else if (label === "line")
        {
            outputCtx.moveTo(drawer.startx, drawer.starty);

            outputCtx.lineTo(drawer.endx,  drawer.endy);
        }
        else if (label === 'circle')
        {
            outputCtx.ellipse(drawer.minX + 0.5 * width, drawer.minY + 0.5 * height, 0.5*width, 0.5*height, 0, 0, 2*Math.PI)
        }
        else if (label === 'triangle')
        {
            outputCtx.moveTo(drawer.minX, drawer.maxY);
            outputCtx.lineTo(drawer.minX + 0.5 * width, drawer.minY);
            outputCtx.lineTo(drawer.maxX, drawer.maxY);
            outputCtx.lineTo(drawer.minX, drawer.maxY)
        }
        outputCtx.stroke();
    }

}

function initTrainingList(trainingInputs, trainingLabels)
{
    for (let i = 0; i < trainingInputs.length; ++i)
    {
        const input = trainingInputs[i];
        const output = trainingLabels[i];
        addExample(input, output);
    }
}

function updatePredictions(trainingInputs, model)
{

    const predElms = document.querySelectorAll('.js-pred');
    const preds = model.predict(tf.tensor(trainingInputs));
    const hardPreds = preds.argMax(1).arraySync();

    for (let i = 0; i < trainingInputs.length; ++i)
    {
        const input = trainingInputs[i];
        const predictedLabel = reverseLabels[hardPreds[i]];

        // we manipulate the index because we are _prepending_ to the list
        predElms[trainingInputs.length - i - 1].innerHTML = predictedLabel;
    }
}
function addExample(input, output)
{
    const sizePx = Math.sqrt(input.length);


    const trElm = document.createElement('tr');
    document.getElementById('tblExamples').prepend(trElm);

    const tdCanvas = document.createElement('td');
    trElm.appendChild(tdCanvas);

    // init canvas for drawn shape
    const canvas = document.createElement('canvas');
    canvas.classList.add('shape');
    tdCanvas.appendChild(canvas);
    canvas.width = sizePx;
    canvas.height = sizePx;

    // draw the shape
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
    
    // set label
    const tdLabel = document.createElement('td');
    trElm.appendChild(tdLabel);
    tdLabel.innerHTML = output;

    // prediction element
    const tdPrediction = document.createElement('td');
    trElm.appendChild(tdPrediction);
    tdPrediction.classList.add('js-pred');
    tdPrediction.innerHTML = "";
}

// creates a TF model
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
            this.startx = this.x;
            this.starty = this.y;
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
                this.endx = e.offsetX;
                this.endy = e.offsetY;

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