(() => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const outCanvas = document.getElementById('outCanvas');
    const cropImg = document.getElementById('cropImg');

    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (e) {
            console.error('camera init failed', e);
        }
    }

    function captureFrame() {
        const w = canvas.width = video.videoWidth || 480;
        const h = canvas.height = video.videoHeight || 360;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, w, h);
        return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    }

    async function postDetect() {
        const blob = await captureFrame();
        if (!blob) return;
        const form = new FormData();
        form.append('file', blob, 'frame.png');

        // call WASM-backed API endpoint on backend server (Vivaz.Api on port 5000)
        // this uses the Vivaz.WASM helper inside the API (preferred when WASM weights embedded)
        const resp = await fetch('http://localhost:5000/api/face/wasm/detectjson', { method: 'POST', body: form });
        if (!resp.ok) {
            console.error('API error', resp.statusText);
            return;
        }
        const json = await resp.json();
        drawDetections(json);

        // also fetch the crop PNG
        // request crop via WASM-backed endpoint as well
        const cropResp = await fetch('http://localhost:5000/api/face/wasm/detectcrop', { method: 'POST', body: form });
        if (cropResp.ok) {
            const blobCrop = await cropResp.blob();
            cropImg.src = URL.createObjectURL(blobCrop);
        }
    }

    function drawDetections(data) {
        const ctx = outCanvas.getContext('2d');
        outCanvas.width = video.videoWidth || 480;
        outCanvas.height = video.videoHeight || 360;
        ctx.drawImage(video, 0, 0, outCanvas.width, outCanvas.height);
        ctx.lineWidth = 2;
        // per-scale boxes
        const colors = { '32': 'red', '48': 'green', '64': 'blue' };
        if (data && data.scales) {
            for (const scaleEntry of data.scales) {
                const s = scaleEntry.scale;
                for (const det of scaleEntry.detections || []) {
                    ctx.strokeStyle = colors[String(s)] || 'magenta';
                    ctx.strokeRect(det.x, det.y, det.w, det.h);
                }
            }
        }
        // final
        if (data && data.final) {
            ctx.lineWidth = 4;
            ctx.strokeStyle = 'yellow';
            const f = data.final;
            ctx.strokeRect(f.x, f.y, f.w, f.h);
        }
    }

    document.getElementById('btnCapture').addEventListener('click', postDetect);
    initCamera();
})();
