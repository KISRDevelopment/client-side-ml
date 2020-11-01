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

        this.minX = Infinity;
        this.maxX = -Infinity;
        this.minY = Infinity;
        this.maxY = -Infinity;

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
                this.getBWPixels(30);
                //this.drawExtent();

            }
        };
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

        const aspectRatio = (this.maxX - this.minX) / (this.maxY - this.minY);
        let newWidth = 0;
        let newHeight = 0;
        if (aspectRatio > 1)
        {
            newHeight = targetSizePx / aspectRatio;
            newWidth = targetSizePx;
        }
        else 
        {
            newHeight = targetSizePx;
            newWidth = targetSizePx * aspectRatio;
        }

        offscreenCtx.drawImage(this.canvas, this.minX, this.minY, this.maxX - this.minX, 
                this.maxY - this.minY, (targetSizePx - newWidth) / 2, (targetSizePx - newHeight) / 2, newWidth , newHeight);

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

    drawExtent()
    {
        const ctx = this.ctx;

        ctx.strokeStyle = 'magenta';
        console.log(this.minX, this.minY, this.maxX, this.maxY)
        ctx.rect(this.minX, this.minY, this.maxX - this.minX, this.maxY - this.minY);
        ctx.stroke();
    }
}

window.onload = main;