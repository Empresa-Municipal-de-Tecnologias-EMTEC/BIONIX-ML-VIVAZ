(async function(){
  if (window.VivazClientWASM) return;
  // load dotnet runtime from published vivaz-wasm folder
  await new Promise((res, rej)=>{ const s=document.createElement('script'); s.src='/vivaz-wasm/dotnet.js'; s.onload=res; s.onerror=rej; document.head.appendChild(s); });

  // initialize runtime, direct boot resources to /vivaz-wasm
  const runtime = await createDotnetRuntime({
    loadBootResource: (type, name, defaultUri) => {
      return '/vivaz-wasm/' + name;
    }
  });

  // bind managed static methods to JS functions using mono_bind_static_method
  const bind = (sig) => {
    try { return Module.mono_bind_static_method(sig); } catch (e) { return null; }
  };

  const detectFn = bind('[Vivaz.WASM] Vivaz.WASM.VivazClient:DetectJson');
  const embedFn = bind('[Vivaz.WASM] Vivaz.WASM.VivazClient:EmbedJson');
  const compareFn = bind('[Vivaz.WASM] Vivaz.WASM.VivazClient:CompareJson');

  window.VivazClientWASM = {
    async embedFromArrayBuffer(buf){
      const arr = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
      if (embedFn) {
        const r = embedFn(arr);
        try { return JSON.parse(r); } catch(e){ return r; }
      }
      return null;
    },
    async compareFromArrayBuffer(aBuf, bBuf, threshold=0.7){
      const aa = aBuf instanceof Uint8Array ? aBuf : new Uint8Array(aBuf);
      const bb = bBuf instanceof Uint8Array ? bBuf : new Uint8Array(bBuf);
      if (compareFn) {
        const r = compareFn(aa, bb, threshold);
        try { return JSON.parse(r); } catch(e){ return r; }
      }
      return null;
    },
    async detectFromArrayBuffer(buf){
      const arr = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
      if (detectFn) {
        const r = detectFn(arr);
        try { return JSON.parse(r); } catch(e){ return r; }
      }
      return null;
    }
  };
})();
