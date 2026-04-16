(() => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const overlay = document.getElementById('overlay') || (() => { const c = document.createElement('canvas'); c.id='overlay'; document.body.insertBefore(c, video.nextSibling); return c; })();
  const imgA = document.getElementById('imgA');
  const imgB = document.getElementById('imgB');
  const resultEl = document.getElementById('result');
  let blobA = null, blobB = null;

  async function initCamera(){
    try{ const s = await navigator.mediaDevices.getUserMedia({ video: true }); video.srcObject = s; await new Promise(r=>video.onloadedmetadata = r); }
    catch(e){ console.error('Não foi possível aceder à câmara', e); throw e; }
  }

  function drawEllipse(){
    overlay.width = video.videoWidth || 320;
    overlay.height = video.videoHeight || 240;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0,0,overlay.width,overlay.height);
    // translucent dark background
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(0,0,overlay.width,overlay.height);
    // cut out ellipse
    const cx = overlay.width/2, cy = overlay.height/2;
    const rx = overlay.width * 0.28, ry = overlay.height * 0.36;
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI*2); ctx.fill();
    ctx.restore();
    // stroke ellipse
    ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI*2); ctx.strokeStyle='white'; ctx.lineWidth=3; ctx.stroke();
    // small hint
    ctx.fillStyle='white'; ctx.font='14px sans-serif'; ctx.textAlign='center'; ctx.fillText('Encaixe o rosto na elipse e capture', cx, overlay.height - 10);
  }

  function capture(){ const w = canvas.width = video.videoWidth || 320; const h = canvas.height = video.videoHeight || 240; const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0,w,h); return new Promise(res=>canvas.toBlob(res,'image/png')); }

  function isCentered(det){
    if(!det) return false;
    const vw = video.videoWidth || 320, vh = video.videoHeight || 240;
    const cx = vw/2, cy = vh/2; const rx = vw*0.28, ry = vh*0.36;
    const bx = det.x + det.w/2, by = det.y + det.h/2;
    const dx = (bx - cx)/rx, dy = (by - cy)/ry;
    const inside = (dx*dx + dy*dy) <= 1.0;
    // require face size reasonable
    const minSize = Math.min(vw, vh) * 0.12; const maxSize = Math.min(vw, vh) * 0.8;
    const sizeOk = det.w >= minSize && det.h >= minSize && det.w <= maxSize && det.h <= maxSize;
    return inside && sizeOk;
  }

  async function detectBlob(blob){ const form = new FormData(); form.append('image', blob, 'img.png'); form.append('file', blob, 'img.png'); try{ const r = await fetch('/api/face/detectjson', { method:'POST', body: form }); if(!r.ok) return null; return await r.json(); }catch(e){ console.error('detect error', e); return null; } }

  async function postCompare(aBlob,bBlob){ const form = new FormData(); form.append('a', aBlob, 'a.png'); form.append('b', bBlob, 'b.png'); try{ const r = await fetch('/api/face/compare', { method:'POST', body: form }); if(!r.ok) return null; return await r.json(); }catch(e){ console.error('compare error', e); return null; } }

  document.getElementById('capA').addEventListener('click', async ()=>{
    try{ if(!video.srcObject) await initCamera(); }
    catch{ alert('Não foi possível aceder à câmara.'); return; }
    const blob = await capture(); const detResp = await detectBlob(blob);
    if(!detResp || !detResp.final){ alert('Rosto não detectado. Ajuste e tente novamente.'); return; }
    const det = { x: detResp.final.x, y: detResp.final.y, w: detResp.final.w, h: detResp.final.h };
    if(!isCentered(det)){ alert('Por favor, encaixe o rosto dentro da elipse antes de capturar.'); return; }
    blobA = blob; imgA.src = URL.createObjectURL(blobA);
  });

  document.getElementById('capB').addEventListener('click', async ()=>{
    try{ if(!video.srcObject) await initCamera(); }
    catch{ alert('Não foi possível aceder à câmara.'); return; }
    const blob = await capture(); const detResp = await detectBlob(blob);
    if(!detResp || !detResp.final){ alert('Rosto não detectado. Ajuste e tente novamente.'); return; }
    const det = { x: detResp.final.x, y: detResp.final.y, w: detResp.final.w, h: detResp.final.h };
    if(!isCentered(det)){ alert('Por favor, encaixe o rosto dentro da elipse antes de capturar.'); return; }
    blobB = blob; imgB.src = URL.createObjectURL(blobB);
  });

  document.getElementById('compare').addEventListener('click', async ()=>{
    if(!blobA || !blobB){ alert('Capture as duas imagens primeiro.'); return; }
    resultEl.innerText = 'Comparando...';
    const res = await postCompare(blobA, blobB);
    if(!res){ resultEl.innerText = 'Erro na comparação'; return; }
    resultEl.innerText = JSON.stringify(res);
  });

  // animate overlay
  function loop(){ if(video.readyState >= 2){ drawEllipse(); } requestAnimationFrame(loop); }
  loop();
})();
