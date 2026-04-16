// vivaz.js - lightweight glue to initialize .NET WASM runtime and expose a
// minimal `window.VivazClientWASM` with methods used by the demo.

(function(){
  async function init(){
    try{
      const mod = await import('/vivaz-wasm/dotnet.js');
      // Prefer the new .NET 8+ API: `dotnet.withConfig(...).create()` when available.
      let runtime;
      try {
        if (mod && mod.dotnet && typeof mod.dotnet.withConfig === 'function') {
          const cfg = {
            configSrc: '/vivaz-wasm/blazor.boot.json',
            loadBootResource: (type, name, defaultUri, integrity) => {
              try { console.debug('[vivaz] loadBootResource', { type, name, defaultUri, integrity }); } catch (e) {}
              return defaultUri;
            },
            environmentVariables: {
              VIVAZ_API_URL: (typeof window !== 'undefined' && window.location ? window.location.origin : ''),
              MONO_LOG_LEVEL: 'debug',
              MONO_LOG_MASK: 'all'
            }
          };
          runtime = await mod.dotnet.withConfig(cfg).create();
        }
      } catch(e) {
        console.debug('[vivaz] dotnet.withConfig failed, falling back', e);
      }

      if (!runtime) {
        const create = mod.default || mod.createDotnetRuntime || window.createDotnetRuntime;
        if (!create) throw new Error('createDotnetRuntime not found in /vivaz-wasm/dotnet.js');
        runtime = await create(() => ({
          configSrc: '/vivaz-wasm/blazor.boot.json',
          loadBootResource: (type, name, defaultUri, integrity) => {
            try { console.debug('[vivaz] loadBootResource', { type, name, defaultUri, integrity }); } catch (e) {}
            return defaultUri;
          },
          config: {
            environmentVariables: {
              VIVAZ_API_URL: (typeof window !== 'undefined' && window.location ? window.location.origin : ''),
              MONO_LOG_LEVEL: 'debug',
              MONO_LOG_MASK: 'all'
            }
          }
        }));
      }

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
