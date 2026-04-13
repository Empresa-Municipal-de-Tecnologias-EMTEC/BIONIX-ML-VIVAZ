(async ()=>{
  // Shared compare logic for both API and WASM demos.
  // Expects window.demoCompareConfig to define: embedEndpoint, compareEndpoint, thresholdPercent
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const imgA = document.getElementById('imgA');
  const imgB = document.getElementById('imgB');
  const resultEl = document.getElementById('result');
  let blobA = null, blobB = null;

  async function initCamera(){
    try{ const s = await navigator.mediaDevices.getUserMedia({video:true}); video.srcObject = s; }
    catch(e){ console.error('camera init failed', e); }
  }

  function capture(){ const w = canvas.width = video.videoWidth || 320; const h = canvas.height = video.videoHeight || 240; const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0,w,h); return new Promise(res=>canvas.toBlob(res,'image/png')); }

  async function postEmbed(blob){ const form = new FormData(); form.append('file', blob, 'img.png');
    try{
      const resp = await fetch(window.demoCompareConfig.embedEndpoint, { method: 'POST', body: form });
      if(!resp.ok) return null;
      return await resp.json();
    }catch(e){ console.error('embed error', e); return null; }
  }

  async function postCompare(blobA_, blobB_){ const form = new FormData(); form.append('a', blobA_, 'a.png'); form.append('b', blobB_, 'b.png');
    try{
      const resp = await fetch(window.demoCompareConfig.compareEndpoint, { method: 'POST', body: form });
      if(!resp.ok) return null;
      return await resp.json();
    }catch(e){ console.error('compare error', e); return null; }
  }

  document.getElementById('capA').addEventListener('click', async ()=>{ blobA = await capture(); imgA.src = URL.createObjectURL(blobA); });
  document.getElementById('capB').addEventListener('click', async ()=>{ blobB = await capture(); imgB.src = URL.createObjectURL(blobB); });
  document.getElementById('compare').addEventListener('click', async ()=>{
    if(!blobA || !blobB){ alert('capture both images'); return; }

    // prefer embed endpoints: fetch embeddings then compute locally for responsiveness
    const aJson = await postEmbed(blobA);
    const bJson = await postEmbed(blobB);
    if(aJson && aJson.embedding && bJson && bJson.embedding){
      const a = aJson.embedding, b = bJson.embedding;
      let dot=0, na=0, nb=0; for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
      na=Math.sqrt(na); nb=Math.sqrt(nb); const cos = dot/(Math.max(1e-12, na*nb)); const percent = Math.max(0, cos)*100;
      const same = percent >= (window.demoCompareConfig.thresholdPercent||70);
      resultEl.innerText = JSON.stringify({ percent, same, method: 'embed_local' });
      return;
    }

    // fallback to server compare endpoint
    const compareJson = await postCompare(blobA, blobB);
    if(compareJson){ resultEl.innerText = JSON.stringify({ result: compareJson, method: 'server_compare' }); return; }

    resultEl.innerText = 'Comparison failed';
  });

  initCamera();
})();
