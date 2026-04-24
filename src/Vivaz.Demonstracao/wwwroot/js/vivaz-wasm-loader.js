// vivaz-wasm-loader.js
// Lightweight loader that attempts to load client-side WASM artifacts from /vivaz-wasm/
// and expose a minimal `window.vivazWasm` API. If no client WASM is available
// it falls back to proxying requests to the server endpoints defined in
// window.demoCompareConfig.

(function(){
  if (window.vivazWasm) return;

  const api = {
    ready: null,
    _impl: null,
    _exports: null,

    async init() {
      if (this._exports) return this._exports;
      
      try {
        console.log("Iniciando Vivaz.WASM Runtime...");
        // Importa o helper gerado pelo script de publicação
        const module = await import('/vivaz-wasm/vivaz-loader-helper.js');
        this._exports = await module.initVivaz();
        console.log("Vivaz.WASM pronto!", this._exports.Vivaz.WASM.VivazClient.GetInfo());
        return this._exports;
      } catch (e) {
        console.warn("Falha ao inicializar Vivaz.WASM cliente, caindo para proxy no servidor:", e);
        throw e;
      }
    },

    async embedFromBlob(blob){
      try {
        const exports = await this.init();
        const buffer = new Uint8Array(await blob.arrayBuffer());
        const jsonResult = exports.Vivaz.WASM.VivazClient.EmbedJson(buffer);
        return JSON.parse(jsonResult);
      } catch (e) {
        // if server fallback is disabled, throw
        if (window.demoCompareConfig && window.demoCompareConfig.allowServerFallback === false) throw new Error('No client WASM available and server fallback disabled');
        // fallback to server if configured
        const form = new FormData(); form.append('file', blob, 'img.png');
        const resp = await fetch(window.demoCompareConfig && window.demoCompareConfig.embedEndpoint ? window.demoCompareConfig.embedEndpoint : '/api/face/wasm/embed', { method: 'POST', body: form });
        if(!resp.ok) throw new Error('embed failed');
        return await resp.json();
      }
    },

    async compareBlobs(aBlob, bBlob){
      try {
        const exports = await this.init();
        const bufferA = new Uint8Array(await aBlob.arrayBuffer());
        const bufferB = new Uint8Array(await bBlob.arrayBuffer());
        const jsonResult = exports.Vivaz.WASM.VivazClient.CompareJson(bufferA, bufferB, 0.7);
        return JSON.parse(jsonResult);
      } catch (e) {
        if (window.demoCompareConfig && window.demoCompareConfig.allowServerFallback === false) throw new Error('No client WASM available and server fallback disabled');
        const form = new FormData(); form.append('a', aBlob, 'a.png'); form.append('b', bBlob, 'b.png');
        const resp = await fetch(window.demoCompareConfig && window.demoCompareConfig.compareEndpoint ? window.demoCompareConfig.compareEndpoint : '/api/face/wasm/compare', { method: 'POST', body: form });
        if(!resp.ok) throw new Error('compare failed');
        return await resp.json();
      }
    },

    // Novos métodos para o Detector
    async detect(blob) {
      try {
        const exports = await this.init();
        const buffer = new Uint8Array(await blob.arrayBuffer());
        const jsonResult = exports.Vivaz.WASM.VivazClient.DetectJson(buffer);
        return JSON.parse(jsonResult);
      } catch (e) {
        const form = new FormData(); form.append('file', blob, 'img.png');
        const resp = await fetch('/api/face/wasm/detectjson', { method: 'POST', body: form });
        if(!resp.ok) throw new Error('detect failed');
        return await resp.json();
      }
    },

    async detectCrop(blob) {
      try {
        const exports = await this.init();
        const buffer = new Uint8Array(await blob.arrayBuffer());
        const croppedBuffer = exports.Vivaz.WASM.VivazClient.DetectCrop(buffer);
        if (!croppedBuffer) return null;
        return new Blob([croppedBuffer], { type: 'image/png' });
      } catch (e) {
        const form = new FormData(); form.append('file', blob, 'img.png');
        const resp = await fetch('/api/face/wasm/detectcrop', { method: 'POST', body: form });
        if(!resp.ok) throw new Error('detect crop failed');
        return await resp.blob();
      }
    }
  };

  api.ready = api.init().catch(e => console.log("WASM client load failed, using server fallback"));
  window.vivazWasm = api;
})();
