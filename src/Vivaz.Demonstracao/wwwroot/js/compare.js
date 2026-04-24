// Shared compare logic for both API and WASM demos.
// Expects window.demoCompareConfig to define: embedEndpoint, compareEndpoint, thresholdPercent
(async function(){
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const imgA = document.getElementById('imgA');
  const imgB = document.getElementById('imgB');
  const resultEl = document.getElementById('result');
  const overlay = document.getElementById('overlay');
  let blobA = null, blobB = null;
  let detecting = false;

  function resizeOverlay(){
    if(!overlay || !video) return;
    const w = video.videoWidth || video.width || 640;
    const h = video.videoHeight || video.height || 480;
    if(overlay.width !== w || overlay.height !== h){
      overlay.width = w; overlay.height = h;
    }
    // Sincroniza o tamanho visual do canvas com o elemento video
    overlay.style.width = video.clientWidth + 'px';
    overlay.style.height = video.clientHeight + 'px';
  }

  async function initCamera(){
    try{
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = s;
      return new Promise(res => {
        video.onloadedmetadata = () => { 
          video.play(); 
          resizeOverlay();
          res();
        };
      });
    }catch(e){ console.error('camera init failed', e); throw e; }
  }

  function capture(){
    const w = canvas.width = video.videoWidth || video.width || 640;
    const h = canvas.height = video.videoHeight || video.height || 480;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0,w,h);
    return new Promise(res=>canvas.toBlob(res,'image/png'));
  }

  async function detectOnce(blob){
    if(window.vivazWasm){
      if(window.vivazWasm.ready) await window.vivazWasm.ready;
      return await window.vivazWasm.detect(blob);
    }
    
    const buf = await blob.arrayBuffer();
    if(window.VivazClientWASM && window.VivazClientWASM.detectFromArrayBuffer){
      return await window.VivazClientWASM.detectFromArrayBuffer(new Uint8Array(buf));
    }
    
    console.warn('No client WASM available for detect; server fallback disabled for compare_wasm.html');
    return null;
  }

  async function detectLoop(intervalMs=1000){
    console.log(`[compare] Iniciando loop de detecção (${intervalMs}ms)...`);
    while(detecting){
      try{
        const blob = await capture();
        const resp = await detectOnce(blob);
        if(overlay){ 
          resizeOverlay(); 
          const ctx = overlay.getContext('2d'); 
          ctx.clearRect(0,0,overlay.width, overlay.height); 
          
          if(resp && resp.found){
            const face = resp.final || resp;
            ctx.strokeStyle='#00ff00'; 
            ctx.lineWidth=4; 
            ctx.strokeRect(face.x, face.y, face.w, face.h);
            
            ctx.fillStyle = '#00ff00';
            ctx.font = '16px Arial';
            ctx.fillText('Face Detectada', face.x, face.y > 20 ? face.y - 5 : face.y + 20);
          }
        }
      }catch(e){ console.warn('detect loop error', e); if(overlay){ const ctx = overlay.getContext('2d'); ctx.clearRect(0,0,overlay.width, overlay.height); } }
      await new Promise(r=>setTimeout(r, intervalMs));
    }
    if(overlay){ const ctx = overlay.getContext('2d'); ctx.clearRect(0,0,overlay.width, overlay.height); }
  }

  async function postEmbed(blob){
    const form = new FormData(); form.append('file', blob, 'img.png');
    try{
      const resp = await fetch(window.demoCompareConfig.embedEndpoint, { method: 'POST', body: form });
      if(!resp.ok) return null; return await resp.json();
    }catch(e){ console.error('embed error', e); return null; }
  }

  async function postCompare(blobA_, blobB_){
    const form = new FormData(); form.append('a', blobA_, 'a.png'); form.append('b', blobB_, 'b.png');
    try{ const resp = await fetch(window.demoCompareConfig.compareEndpoint, { method: 'POST', body: form }); if(!resp.ok) return null; return await resp.json(); }
    catch(e){ console.error('compare error', e); return null; }
  }

  document.getElementById('capA').addEventListener('click', async ()=>{ blobA = await capture(); imgA.src = URL.createObjectURL(blobA); });
  document.getElementById('capB').addEventListener('click', async ()=>{ blobB = await capture(); imgB.src = URL.createObjectURL(blobB); });

  document.getElementById('compare').addEventListener('click', async ()=>{
    if(!blobA || !blobB){ alert('capture both images'); return; }
    resultEl.innerText = 'Processando...';
    try{
      if (window.vivazWasm){
        const aJson = await window.vivazWasm.embedFromBlob(blobA);
        const bJson = await window.vivazWasm.embedFromBlob(blobB);
        if (aJson && bJson && aJson.embedding && bJson.embedding){
          const a = aJson.embedding, b = bJson.embedding; let dot=0, na=0, nb=0; for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
          na=Math.sqrt(na); nb=Math.sqrt(nb); const cos = dot/(Math.max(1e-12, na*nb)); const percent = Math.max(0, cos)*100;
          const same = percent >= (window.demoCompareConfig.thresholdPercent||70);
          resultEl.innerText = `Resultado: ${same ? 'SIM' : 'NÃO'} (${percent.toFixed(2)}%)`; 
          return;
        }
      }
    }catch(e){ console.warn('client WASM embed failed, falling back', e); }

    const aJson = await postEmbed(blobA);
    const bJson = await postEmbed(blobB);
    if(aJson && aJson.embedding && bJson && bJson.embedding){
      const a = aJson.embedding, b = bJson.embedding; let dot=0, na=0, nb=0; for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
      na=Math.sqrt(na); nb=Math.sqrt(nb); const cos = dot/(Math.max(1e-12, na*nb)); const percent = Math.max(0, cos)*100; const same = percent >= (window.demoCompareConfig.thresholdPercent||70);
      resultEl.innerText = `Resultado (API): ${same ? 'SIM' : 'NÃO'} (${percent.toFixed(2)}%)`; 
      return;
    }

    const compareJson = await postCompare(blobA, blobB);
    if(compareJson){ 
      const res = compareJson.result || compareJson;
      resultEl.innerText = `Resultado (Server): ${res.same ? 'SIM' : 'NÃO'} (${res.percent}%)`; 
      return; 
    }
    resultEl.innerText = 'Comparison failed';
  });

  document.getElementById('startDetect').addEventListener('click', async (ev)=>{
    if(!video.srcObject){ try{ await initCamera(); } catch(e){ console.warn('camera init on demand failed', e); } }
    detecting = !detecting; 
    ev.target.innerText = detecting ? 'Parar Detecção' : 'Iniciar Detecção'; 
    if(detecting) detectLoop(1000);
  });

  // Try to start camera and detection automatically (may require user gesture in some browsers)
  (async ()=>{ try{ await initCamera(); detecting = true; const btn = document.getElementById('startDetect'); if(btn) btn.innerText = 'Parar Detecção'; detectLoop(1000); } catch(e){ /* ignore */ } })();

})();
