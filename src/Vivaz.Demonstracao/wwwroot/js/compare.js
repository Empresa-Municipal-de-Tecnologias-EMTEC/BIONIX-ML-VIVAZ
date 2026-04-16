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
    const w = video.videoWidth || video.width || 320;
    const h = video.videoHeight || video.height || 240;
    if(overlay.width !== w || overlay.height !== h){
      overlay.width = w; overlay.height = h;
      overlay.style.width = w + 'px'; overlay.style.height = h + 'px';
    }
  }

  async function initCamera(){
    try{
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      video.srcObject = s;
      video.onloadedmetadata = ()=>{ video.play(); resizeOverlay(); };
    }catch(e){ console.error('camera init failed', e); throw e; }
  }

  function capture(){
    const w = canvas.width = video.videoWidth || video.width || 320;
    const h = canvas.height = video.videoHeight || video.height || 240;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0,w,h);
    return new Promise(res=>canvas.toBlob(res,'image/png'));
  }

  async function detectOnce(blob){
    const buf = await blob.arrayBuffer();
    if(window.vivazWasm && window.vivazWasm.ready) await window.vivazWasm.ready;
    if(window.VivazClientWASM && window.VivazClientWASM.detectFromArrayBuffer){
      return await window.VivazClientWASM.detectFromArrayBuffer(new Uint8Array(buf));
    }
    if(window.vivazWasm && window.vivazWasm._impl && window.vivazWasm._impl.detectFromArrayBuffer){
      return await window.vivazWasm._impl.detectFromArrayBuffer(new Uint8Array(buf));
    }
    const form = new FormData(); form.append('file', blob, 'img.png');
    const r = await fetch('/api/face/wasm/detectjson', { method: 'POST', body: form });
    if(!r.ok) return null; return await r.json();
  }

  async function detectLoop(intervalMs=250){
    while(detecting){
      try{
        const blob = await capture();
        const resp = await detectOnce(blob);
        if(overlay){ resizeOverlay(); const ctx = overlay.getContext('2d'); ctx.clearRect(0,0,overlay.width, overlay.height); }
        if(resp && resp.found && overlay){ const ctx = overlay.getContext('2d'); ctx.strokeStyle='lime'; ctx.lineWidth=3; ctx.strokeRect(resp.x, resp.y, resp.w, resp.h); }
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
    try{
      if (window.vivazWasm && window.vivazWasm.ready) await window.vivazWasm.ready;
      if (window.vivazWasm && window.vivazWasm.embedFromBlob){
        const aJson = await window.vivazWasm.embedFromBlob(blobA);
        const bJson = await window.vivazWasm.embedFromBlob(blobB);
        if (aJson && bJson && aJson.embedding && bJson.embedding){
          const a = aJson.embedding, b = bJson.embedding; let dot=0, na=0, nb=0; for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
          na=Math.sqrt(na); nb=Math.sqrt(nb); const cos = dot/(Math.max(1e-12, na*nb)); const percent = Math.max(0, cos)*100;
          const same = percent >= (window.demoCompareConfig.thresholdPercent||70);
          resultEl.innerText = JSON.stringify({ percent, same, method: 'embed_local_wasm' }); return;
        }
      }
    }catch(e){ console.warn('client WASM embed failed, falling back', e); }

    const aJson = await postEmbed(blobA);
    const bJson = await postEmbed(blobB);
    if(aJson && aJson.embedding && bJson && bJson.embedding){
      const a = aJson.embedding, b = bJson.embedding; let dot=0, na=0, nb=0; for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
      na=Math.sqrt(na); nb=Math.sqrt(nb); const cos = dot/(Math.max(1e-12, na*nb)); const percent = Math.max(0, cos)*100; const same = percent >= (window.demoCompareConfig.thresholdPercent||70);
      resultEl.innerText = JSON.stringify({ percent, same, method: 'embed_local' }); return;
    }

    const compareJson = await postCompare(blobA, blobB);
    if(compareJson){ resultEl.innerText = JSON.stringify({ result: compareJson, method: 'server_compare' }); return; }
    resultEl.innerText = 'Comparison failed';
  });

  document.getElementById('startDetect').addEventListener('click', async (ev)=>{
    if(!video.srcObject){ try{ await initCamera(); } catch(e){ console.warn('camera init on demand failed', e); } }
    detecting = !detecting; ev.target.innerText = detecting ? 'Stop Detect' : 'Start Detect'; if(detecting) detectLoop(250);
  });

  // Try to start camera and detection automatically (may require user gesture in some browsers)
  (async ()=>{ try{ await initCamera(); detecting = true; const btn = document.getElementById('startDetect'); if(btn) btn.innerText = 'Stop Detect'; detectLoop(250); } catch(e){ /* ignore */ } })();

})();
