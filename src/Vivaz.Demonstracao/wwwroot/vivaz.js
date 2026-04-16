// vivaz.js - lightweight glue to initialize .NET WASM runtime and expose a
// minimal `window.VivazClientWASM` with methods used by the demo.

(function(){
  async function init(){
    // singleton guard: if runtime already initialized, reuse it; if init in progress, await it
    if (window.VivazClientWASM && window.VivazClientWASM._runtime) return window.VivazClientWASM;
    if (window._vivazInitPromise) return window._vivazInitPromise;

    window._vivazInitPromise = (async () => {
      try{
        const mod = await import('/vivaz-wasm/dotnet.js');

        // Prefer the new .NET 8+ API: `dotnet.withConfig(...).create()` when available.
        let runtime;
        // Prevent concurrent callers from invoking `create()` simultaneously by using
        // a global create promise (`__vivaz_createPromise`). Other callers will await it
        // and reuse the same runtime instance when it's ready.
        try {
          if (mod && mod.dotnet && typeof mod.dotnet.withConfig === 'function') {
            const cfg = {
              configSrc: '/vivaz-wasm/Vivaz.WASM.deps.json',
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

            if (!window.__vivaz_createPromise) {
              window.__vivaz_createPromise = (async () => {
                try {
                  return await mod.dotnet.withConfig(cfg).create();
                } catch (e) {
                  // clear so others can retry if this failed in a non-deterministic way
                  window.__vivaz_createPromise = null;
                  throw e;
                }
              })();
            }

            try {
              runtime = await window.__vivaz_createPromise;
            } catch (errCreate) {
              const msg = errCreate && errCreate.message ? errCreate.message : '';
              if (msg.includes('Runtime module already loaded')) {
                console.warn('[vivaz] runtime already loaded by another script; awaiting existing instance');
                const waitForExisting = async (timeoutMs = 10000) => {
                  const start = Date.now();
                  while (Date.now() - start < timeoutMs) {
                    if (window.VivazClientWASM && window.VivazClientWASM._runtime) return window.VivazClientWASM._runtime;
                    // if another create is in progress, await it
                    if (window.__vivaz_createPromise) {
                      try { const r = await window.__vivaz_createPromise; if (r) return r; } catch(e){}
                    }
                    await new Promise(r => setTimeout(r, 200));
                  }
                  return null;
                };
                const existing = await waitForExisting(10000);
                if (existing) {
                  runtime = existing;
                } else {
                  console.debug('[vivaz] no existing runtime found after wait; rethrowing create error', errCreate);
                  throw errCreate;
                }
              } else {
                throw errCreate;
              }
            }
          }
        } catch(e) {
          console.debug('[vivaz] dotnet.withConfig failed, falling back', e);
        }

        if (!runtime) {
          const create = mod.default || mod.createDotnetRuntime || window.createDotnetRuntime;
          if (!create) throw new Error('createDotnetRuntime not found in /vivaz-wasm/dotnet.js');
          // fallback (non .withConfig) create path: also use the global create promise
          try {
            if (!window.__vivaz_createPromise) {
              window.__vivaz_createPromise = (async () => {
                try {
                  return await create(() => ({
                    configSrc: '/vivaz-wasm/Vivaz.WASM.deps.json',
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
                } catch (e) {
                  window.__vivaz_createPromise = null;
                  throw e;
                }
              })();
            }
            runtime = await window.__vivaz_createPromise;
          } catch(errCreate) {
            const msg = errCreate && errCreate.message ? errCreate.message : '';
            if (msg.includes('Runtime module already loaded')) {
              console.warn('[vivaz] runtime already loaded by another script; awaiting existing instance (fallback create)');
              const waitForExisting = async (timeoutMs = 10000) => {
                const start = Date.now();
                while (Date.now() - start < timeoutMs) {
                  if (window.VivazClientWASM && window.VivazClientWASM._runtime) return window.VivazClientWASM._runtime;
                  if (window.__vivaz_createPromise) {
                    try { const r = await window.__vivaz_createPromise; if (r) return r; } catch(e){}
                  }
                  await new Promise(r => setTimeout(r, 200));
                }
                return null;
              };
              const existing = await waitForExisting(10000);
              if (existing) {
                runtime = existing;
              } else {
                console.debug('[vivaz] no existing runtime found after wait; rethrowing create error', errCreate);
                throw errCreate;
              }
            } else {
              throw errCreate;
            }
          }
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

        return window.VivazClientWASM;
      }catch(err){
        // clear promise on failure so subsequent attempts can retry
        console.error('vivaz.js init error:', err);
        window._vivazInitPromise = null;
        throw err;
      }
    })();

    return window._vivazInitPromise;
  }

  // start initialization in background (idempotent)
  init().catch(()=>{});
})();
