// vivaz.js - lightweight glue to initialize .NET WASM runtime and expose a
// minimal `window.VivazClientWASM` with methods used by the demo.

(function(){
  async function init(){
    try{
      const mod = await import('/vivaz-wasm/dotnet.js');
      const create = mod.default || mod.createDotnetRuntime || mod.createDotnetRuntime || window.createDotnetRuntime;
      if (!create) throw new Error('createDotnetRuntime not found in /vivaz-wasm/dotnet.js');

      const runtime = await create({
        config: {
          // allow the managed code to download pesos from the same origin
          environmentVariables: { VIVAZ_API_URL: (typeof window !== 'undefined' && window.location ? window.location.origin : '') }
        }
      });

      const exports = await runtime.getAssemblyExports('Vivaz.WASM.dll');

      function findExport(obj, name){
        if (!obj) return null;
        for (const k of Object.keys(obj)){
          try{
            const v = obj[k];
            if (typeof v === 'function' && k.toLowerCase() === name.toLowerCase()) return v;
            if (typeof v === 'object'){
              const r = findExport(v, name);
              if (r) return r;
            }
          }catch(e){/* ignore */}
        }
        return null;
      }

      const detectJsonFn = findExport(exports, 'DetectJson') || findExport(exports, 'DetectFromRgb');
      const embedJsonFn = findExport(exports, 'EmbedJson') || findExport(exports, 'EmbedFromRgb');
      const compareJsonFn = findExport(exports, 'CompareJson') || findExport(exports, 'CompareFromRgb');

      window.VivazClientWASM = {
        _runtime: runtime,
        async embedFromArrayBuffer(ab){
          if (!embedJsonFn) throw new Error('Embed function not found in Vivaz.WASM exports');
          const arr = (ab instanceof Uint8Array) ? ab : new Uint8Array(ab);
          const res = await embedJsonFn(arr);
          try { return JSON.parse(res); } catch { return res; }
        },
        async compareFromArrayBuffer(aBuf, bBuf){
          if (!compareJsonFn) throw new Error('Compare function not found in Vivaz.WASM exports');
          const a = (aBuf instanceof Uint8Array) ? aBuf : new Uint8Array(aBuf);
          const b = (bBuf instanceof Uint8Array) ? bBuf : new Uint8Array(bBuf);
          const res = await compareJsonFn(a, b, 0.7);
          try { return JSON.parse(res); } catch { return res; }
        },
        async detectFromArrayBuffer(ab){
          if (!detectJsonFn) throw new Error('Detect function not found in Vivaz.WASM exports');
          const arr = (ab instanceof Uint8Array) ? ab : new Uint8Array(ab);
          const res = await detectJsonFn(arr);
          try { return JSON.parse(res); } catch { return res; }
        }
      };

      console.info('Vivaz WASM runtime initialized, VivazClientWASM available');
    }catch(err){
      console.error('vivaz.js init error:', err);
    }
  }

  // start initialization in background
  init();
})();
