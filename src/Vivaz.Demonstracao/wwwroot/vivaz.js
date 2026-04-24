// vivaz.js - Singleton Loader for Vivaz.WASM (.NET 8)
(function() {
    if (window.vivazWasm && window.vivazWasm._runtime) return;

    const vivazWasm = {
        ready: null,
        _exports: null,
        _runtime: null,

        async init() {
            if (this._exports) return this._exports;
            if (this.ready) return this.ready;

            this.ready = (async () => {
                try {
                    console.log("[vivaz] Inicializando Runtime .NET 8 WASM...");
                    
                    // Importa o módulo nativo do .NET
                    const { dotnet } = await import('/vivaz-wasm/_framework/dotnet.js');
                    
                    const { getAssemblyExports, getConfig } = await dotnet
                        .withDiagnosticTracing(false)
                        .withApplicationArgumentsFromQuery()
                        .create();

                    const config = getConfig();
                    this._exports = await getAssemblyExports(config.mainAssemblyName);
                    this._runtime = dotnet;
                    
                    console.log("[vivaz] Vivaz.WASM carregado com sucesso!");
                    return this._exports;
                } catch (e) {
                    if (e.message && e.message.includes("already loaded")) {
                        console.warn("[vivaz] Runtime já carregado, tentando recuperar exportações...");
                        // Se já carregou, o objeto dotnet deve estar disponível ou em cache
                        return this._exports; 
                    }
                    console.error("[vivaz] Erro fatal ao carregar o runtime WASM:", e);
                    this.ready = null; 
                    throw e;
                }
            })();

            return this.ready;
        },

        async _call(method, ...args) {
            const exports = await this.init();
            const client = exports.Vivaz.WASM.VivazClient;
            if (!client[method]) throw new Error(`Método ${method} não encontrado no VivazClient`);
            return client[method](...args);
        },

        async detect(blob) {
            try {
                const buffer = new Uint8Array(await blob.arrayBuffer());
                const res = await this._call('DetectJson', buffer);
                return typeof res === 'string' ? JSON.parse(res) : res;
            } catch (e) {
                console.warn("[vivaz] Fallback para API em detect:", e);
                return this._fallback('/api/face/wasm/detectjson', blob);
            }
        },

        async detectCrop(blob) {
            try {
                const buffer = new Uint8Array(await blob.arrayBuffer());
                const res = await this._call('DetectCrop', buffer);
                if (!res) return null;
                return new Blob([res], { type: 'image/png' });
            } catch (e) {
                console.warn("[vivaz] Fallback para API em detectCrop:", e);
                return this._fallback('/api/face/wasm/detectcrop', blob, true);
            }
        },

        async embedFromBlob(blob) {
            try {
                const buffer = new Uint8Array(await blob.arrayBuffer());
                const res = await this._call('EmbedJson', buffer);
                return typeof res === 'string' ? JSON.parse(res) : res;
            } catch (e) {
                console.warn("[vivaz] Fallback para API em embed:", e);
                return this._fallback('/api/face/wasm/embed', blob);
            }
        },

        async compareBlobs(aBlob, bBlob) {
            try {
                const bufferA = new Uint8Array(await aBlob.arrayBuffer());
                const bufferB = new Uint8Array(await bBlob.arrayBuffer());
                const res = await this._call('CompareJson', bufferA, bufferB, 0.7);
                return typeof res === 'string' ? JSON.parse(res) : res;
            } catch (e) {
                console.warn("[vivaz] Fallback para API em compare:", e);
                const form = new FormData();
                form.append('a', aBlob, 'a.png');
                form.append('b', bBlob, 'b.png');
                const resp = await fetch('/api/face/wasm/compare', { method: 'POST', body: form });
                return resp.json();
            }
        },

        async _fallback(url, blob, isBlob = false) {
            if (window.demoCompareConfig && window.demoCompareConfig.allowServerFallback === false) {
                throw new Error("WASM indisponível e fallback de servidor desativado.");
            }
            const form = new FormData();
            form.append('file', blob, 'image.png');
            const resp = await fetch(url, { method: 'POST', body: form });
            return isBlob ? resp.blob() : resp.json();
        }
    };

    // Exporta APIs para compatibilidade global
    window.vivazWasm = vivazWasm;
    window.VivazClientWASM = {
        _runtime: true,
        detectFromArrayBuffer: async (ab) => vivazWasm.detect(new Blob([ab])),
        embedFromArrayBuffer: async (ab) => vivazWasm.embedFromBlob(new Blob([ab])),
        compareFromArrayBuffer: async (a, b) => vivazWasm.compareBlobs(new Blob([a]), new Blob([b]))
    };

    // Inicialização imediata
    vivazWasm.init().catch(() => {});
})();
