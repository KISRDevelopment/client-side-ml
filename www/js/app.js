function main()
{


    const canvas = document.getElementById('canvasShapes');
    const drawer = new SimpleDrawer(canvas);

}

//
// Based on https://developer.mozilla.org/en-US/docs/Web/API/Element/mousemove_event
//
class SimpleDrawer
{
    constructor(canvas)
    {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.drawing = false;
        
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.x = 0;
        this.y = 0;

        canvas.onmousedown = (e) => {
            this.x = e.offsetX;
            this.y = e.offsetY;
            this.drawing = true;
        };

        canvas.onmousemove = (e) => {
            if (this.drawing) {
                this._drawLine(e.offsetX, e.offsetY);
                this.x = e.offsetX;
                this.y = e.offsetY;
            }
        };

        canvas.onmouseup = (e) => {
            if (this.drawing) {
                this._drawLine(e.offsetX, e.offsetY);
                this.x = 0;
                this.y = 0;
                this.drawing = false;
                this.getBWPixels(30);
            }
        };
    }

    _drawLine(toX, toY)
    {
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        ctx.closePath();
    }

    getBWPixels(targetSizePx)
    {
        const ctx = this.ctx;

        const offscreenCanvas = document.getElementById('resizedCanvas');
        offscreenCanvas.width = targetSizePx;
        offscreenCanvas.height = targetSizePx;

        const offscreenCtx = offscreenCanvas.getContext('2d');
        offscreenCtx.imageSmoothingQuality = "high";

        offscreenCtx.drawImage(this.canvas, 0, 0, this.canvas.width, this.canvas.height, 0, 0, targetSizePx, targetSizePx);

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
    }
}

window.onload = main;