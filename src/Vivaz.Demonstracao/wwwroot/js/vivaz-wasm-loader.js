// vivaz-wasm-loader.js
// Lightweight loader that attempts to load client-side WASM artifacts from /wasm/
// and expose a minimal `window.vivazWasm` API. If no client WASM is available
// it falls back to proxying requests to the server endpoints defined in
// window.demoCompareConfig.

(function(){
  if (window.vivazWasm) return;

  const api = {
    ready: Promise.resolve(),
    _impl: null,
    async embedFromBlob(blob){
      if (this._impl && this._impl.embedFromArrayBuffer) return this._impl.embedFromArrayBuffer(await blob.arrayBuffer());
      // if server fallback is disabled, throw
      if (window.demoCompareConfig && window.demoCompareConfig.allowServerFallback === false) throw new Error('No client WASM available and server fallback disabled');
      // fallback to server if configured
      const form = new FormData(); form.append('file', blob, 'img.png');
      const resp = await fetch(window.demoCompareConfig && window.demoCompareConfig.embedEndpoint ? window.demoCompareConfig.embedEndpoint : '/api/face/wasm/embed', { method: 'POST', body: form });
      if(!resp.ok) throw new Error('embed failed');
      return await resp.json();
    },
    async compareBlobs(aBlob, bBlob){
      if (this._impl && this._impl.compareFromArrayBuffer) return this._impl.compareFromArrayBuffer(await aBlob.arrayBuffer(), await bBlob.arrayBuffer());
      if (window.demoCompareConfig && window.demoCompareConfig.allowServerFallback === false) throw new Error('No client WASM available and server fallback disabled');
      const form = new FormData(); form.append('a', aBlob, 'a.png'); form.append('b', bBlob, 'b.png');
      const resp = await fetch(window.demoCompareConfig && window.demoCompareConfig.compareEndpoint ? window.demoCompareConfig.compareEndpoint : '/api/face/wasm/compare', { method: 'POST', body: form });
      if(!resp.ok) throw new Error('compare failed');
      return await resp.json();
    }
  };

  // loader: prefer a JS glue file at /wasm/vivaz.js which should expose a global
  // `VivazClientWASM` with methods `embedFromArrayBuffer` and `compareFromArrayBuffer`.
  api.ready = (async ()=>{
    try{
      // try to load a glue script
      const glueUrl = '/wasm/vivaz.js';
      const r = await fetch(glueUrl, { method: 'HEAD' });
      if (r.ok){
        await new Promise((res, rej)=>{
          const s = document.createElement('script'); s.src = glueUrl; s.onload = res; s.onerror = rej; document.head.appendChild(s);
        });
        if (window.VivazClientWASM && (window.VivazClientWASM.embedFromArrayBuffer || window.VivazClientWASM.compareFromArrayBuffer)){
          api._impl = window.VivazClientWASM;
          return api;
        }
      }
    }catch(e){ /* ignore and fallback */ }
    // no client wasm available — leave _impl null so methods will proxy to server
    return api;
  })();

  window.vivazWasm = api;
})();
