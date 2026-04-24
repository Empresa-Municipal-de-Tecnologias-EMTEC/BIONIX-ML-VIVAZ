// compare.js - Lógica de demonstração para Comparação Facial
// Gerencia o Detector (contínuo para feedback visual) e o Identificador (sob demanda para comparação)
(async function(){
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const imgA = document.getElementById('imgA');
  const imgB = document.getElementById('imgB');
  const resultEl = document.getElementById('result');
  const overlay = document.getElementById('overlay');
  
  let blobA = null, blobB = null;
  let detecting = false;

  // 1. Configuração da Câmera e Overlay
  function resizeOverlay(){
    if(!overlay || !video) return;
    const w = video.videoWidth || video.width || 640;
    const h = video.videoHeight || video.height || 480;
    if(overlay.width !== w || overlay.height !== h){
      overlay.width = w; 
      overlay.height = h;
    }
    overlay.style.width = video.clientWidth + 'px';
    overlay.style.height = video.clientHeight + 'px';
  }

  async function initCamera(){
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = s;
      return new Promise(res => {
        video.onloadedmetadata = () => { 
          video.play(); 
          resizeOverlay();
          res();
        };
      });
    } catch(e) { 
      console.error('[compare] Falha ao iniciar câmera:', e); 
      throw e; 
    }
  }

  function capture(){
    const w = canvas.width = video.videoWidth || video.width || 640;
    const h = canvas.height = video.videoHeight || video.height || 480;
    const ctx = canvas.getContext('2d'); 
    ctx.drawImage(video, 0, 0, w, h);
    return new Promise(res => canvas.toBlob(res, 'image/png'));
  }

  // 2. Lógica do DETECTOR (Rodando 1x por segundo)
  async function detectOnce(blob){
    if(window.vivazWasm){
      if(window.vivazWasm.ready) await window.vivazWasm.ready;
      // Chama o método DetectJson do VivazClient.cs via WASM
      return await window.vivazWasm.detect(blob);
    }
    return null;
  }

  async function detectLoop(intervalMs = 1000){
    console.log(`[compare] Detector iniciado: 1 ciclo a cada ${intervalMs}ms`);
    while(detecting){
      try {
        const blob = await capture();
        const resp = await detectOnce(blob);
        
        if(overlay){ 
          resizeOverlay(); 
          const ctx = overlay.getContext('2d'); 
          ctx.clearRect(0, 0, overlay.width, overlay.height); 
          
          if(resp && resp.found){
            const face = resp.final || resp;
            // Desenha o retângulo sobre o rosto detectado
            ctx.strokeStyle = '#00ff00'; 
            ctx.lineWidth = 4; 
            ctx.strokeRect(face.x, face.y, face.w, face.h);
            
            // Label visual
            ctx.fillStyle = '#00ff00';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('FACE DETECTADA', face.x, face.y > 20 ? face.y - 10 : face.y + 25);
          }
        }
      } catch(e) { 
        console.warn('[compare] Erro no ciclo de detecção:', e); 
      }
      await new Promise(r => setTimeout(r, intervalMs));
    }
    // Limpa o overlay ao parar
    if(overlay){ 
      const ctx = overlay.getContext('2d'); 
      ctx.clearRect(0, 0, overlay.width, overlay.height); 
    }
  }

  // 3. Lógica do IDENTIFICADOR (Sob demanda)
  async function getEmbedding(blob){
    if(window.vivazWasm){
      if(window.vivazWasm.ready) await window.vivazWasm.ready;
      // Chama o método EmbedJson do VivazClient.cs via WASM
      return await window.vivazWasm.embedFromBlob(blob);
    }
    // Fallback para API remota se WASM falhar
    const form = new FormData(); 
    form.append('file', blob, 'img.png');
    const endpoint = (window.demoCompareConfig && window.demoCompareConfig.embedEndpoint) || '/api/face/wasm/embed';
    const resp = await fetch(endpoint, { method: 'POST', body: form });
    return await resp.json();
  }

  // 4. Eventos de Interface
  document.getElementById('capA').addEventListener('click', async ()=>{ 
    blobA = await capture(); 
    imgA.src = URL.createObjectURL(blobA); 
    console.log("[compare] Face A capturada para identificação.");
  });
  
  document.getElementById('capB').addEventListener('click', async ()=>{ 
    blobB = await capture(); 
    imgB.src = URL.createObjectURL(blobB); 
    console.log("[compare] Face B capturada para identificação.");
  });

  document.getElementById('compare').addEventListener('click', async ()=>{
    if(!blobA || !blobB){ alert('Capture as duas faces (A e B) antes de comparar.'); return; }
    
    resultEl.className = "alert alert-info text-center";
    resultEl.innerText = 'Extraindo assinaturas faciais (Identificador)...';
    
    try {
      // Executa o Identificador para as duas imagens
      const aJson = await getEmbedding(blobA);
      const bJson = await getEmbedding(blobB);
      
      if (aJson && bJson && aJson.embedding && bJson.embedding){
        const a = aJson.embedding, b = bJson.embedding; 
        let dot=0, na=0, nb=0; 
        for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
        na=Math.sqrt(na); nb=Math.sqrt(nb); 
        const cos = dot/(Math.max(1e-12, na*nb)); 
        const percent = Math.max(0, cos)*100;
        const same = percent >= (window.demoCompareConfig.thresholdPercent||70);
        
        resultEl.className = `alert ${same ? 'alert-success' : 'alert-danger'} text-center`;
        resultEl.innerHTML = `<strong>Resultado:</strong> ${same ? 'MESMA PESSOA' : 'PESSOAS DIFERENTES'}<br><small>Similaridade: ${percent.toFixed(2)}%</small>`; 
      } else {
        throw new Error("Não foi possível gerar embeddings.");
      }
    } catch(e) { 
      console.error('[compare] Erro na identificação:', e);
      resultEl.className = "alert alert-warning text-center";
      resultEl.innerText = 'Falha na identificação local. Verifique os pesos do Identificador.';
    }
  });

  document.getElementById('startDetect').addEventListener('click', async (ev)=>{
    if(!video.srcObject){ await initCamera(); }
    detecting = !detecting; 
    ev.target.innerText = detecting ? 'Parar Detecção' : 'Iniciar Detecção'; 
    ev.target.className = detecting ? 'btn btn-danger' : 'btn btn-outline-secondary';
    if(detecting) detectLoop(1000); // Frequência solicitada: 1x por segundo
  });

  // Inicialização Automática
  (async ()=>{ 
    try { 
      await initCamera(); 
      detecting = true; 
      const btn = document.getElementById('startDetect'); 
      if(btn) {
        btn.innerText = 'Parar Detecção';
        btn.className = 'btn btn-danger';
      }
      detectLoop(1000); 
    } catch(e) { console.warn('[compare] Auto-start da câmera falhou.'); } 
  })();
})();
