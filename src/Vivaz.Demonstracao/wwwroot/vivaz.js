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
                const tryPaths = [
                    '/vivaz-wasm/_framework/dotnet.js',
                    '/_framework/dotnet.js',
                    '/vivaz-wasm/dotnet.js',
                    '/dotnet.js'
                ];

                let lastError = null;
                for (const path of tryPaths) {
                    try {
                        console.log(`[vivaz] Tentando carregar runtime de: ${path}`);
                        const { dotnet } = await import(path);
                        
                        const { getAssemblyExports, getConfig } = await dotnet
                            .withDiagnosticTracing(false)
                            .withApplicationArgumentsFromQuery()
                            .create();

                        const config = getConfig();
                        this._exports = await getAssemblyExports(config.mainAssemblyName);
                        this._runtime = dotnet;
                        
                        console.log(`[vivaz] Vivaz.WASM carregado com sucesso via ${path}!`);
                        return this._exports;
                    } catch (e) {
                        lastError = e;
                        console.warn(`[vivaz] Falha ao carregar de ${path}:`, e.message);
                    }
                }

                console.error("[vivaz] Erro fatal: Não foi possível carregar o runtime de nenhum caminho conhecido.", lastError);
                this.ready = null; 
                throw lastError;
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
                console.warn("[vivaz] Erro em detect:", e);
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
                console.warn("[vivaz] Erro em detectCrop:", e);
                return this._fallback('/api/face/wasm/detectcrop', blob, true);
            }
        },

        async embedFromBlob(blob) {
            try {
                const buffer = new Uint8Array(await blob.arrayBuffer());
                const res = await this._call('EmbedJson', buffer);
                return typeof res === 'string' ? JSON.parse(res) : res;
            } catch (e) {
                console.warn("[vivaz] Erro em embed:", e);
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
                console.warn("[vivaz] Erro em compare:", e);
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

    window.vivazWasm = vivazWasm;
    window.VivazClientWASM = {
        _runtime: true,
        detectFromArrayBuffer: async (ab) => vivazWasm.detect(new Blob([ab])),
        embedFromArrayBuffer: async (ab) => vivazWasm.embedFromBlob(new Blob([ab])),
        compareFromArrayBuffer: async (a, b) => vivazWasm.compareBlobs(new Blob([a]), new Blob([b]))
    };

    vivazWasm.init().catch(() => {});
})();
