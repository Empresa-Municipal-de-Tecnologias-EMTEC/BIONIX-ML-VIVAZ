// Helper para carregar o Vivaz.WASM usando a nova API do .NET 8
import { dotnet } from './_framework/dotnet.js';

let isRuntimeReady = false;
let exports = null;

async function fetchBootManifest() {
    try {
        const res = await fetch('/vivaz-wasm/blazor.boot.json', { cache: 'no-store' });
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.warn('vivaz-loader-helper: failed to fetch blazor.boot.json', e);
        return null;
    }
}

function mkdirp(FS, path) {
    const parts = path.split('/').filter(p => p.length);
    let cur = '/';
    for (const p of parts) {
        cur = cur.endsWith('/') ? cur + p : cur + '/' + p;
        try { FS.stat(cur); } catch (e) { try { FS.mkdir(cur); } catch (e2) { /* ignore */ } }
    }
}

async function waitForFS(timeoutMs = 30000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        const M = globalThis.Module || window.Module || globalThis.module || null;
        const FS = (M && M.FS) || globalThis.FS || null;
        if (FS && typeof FS.writeFile === 'function' && typeof FS.mkdir === 'function') return FS;
        await new Promise(r => setTimeout(r, 200));
    }
    return null;
}

async function mountPesosIntoFS(FS, assets) {
    if (!FS) return false;
    const pesosAssets = assets.filter(a => a && a.name && a.name.startsWith('PESOS/'));
    console.info(`vivaz-loader-helper: will attempt to mount ${pesosAssets.length} PESOS assets into FS`);
    for (const asset of pesosAssets) {
        const url = asset.name.startsWith('/') ? asset.name : `/${asset.name}`;
        try {
            const res = await fetch(url, { cache: 'no-store' });
            if (!res.ok) { console.warn('vivaz-loader-helper: asset not found', url); continue; }
            const buf = new Uint8Array(await res.arrayBuffer());
            const dir = asset.name.split('/').slice(0, -1).join('/') || '/';
            try { mkdirp(FS, dir); } catch (e) { /* ignore */ }
            try {
                FS.writeFile('/' + asset.name, buf, { flags: 'w' });
                console.info('vivaz-loader-helper: mounted', asset.name);
            } catch (e) {
                console.warn('vivaz-loader-helper: failed to write', asset.name, e);
            }
        } catch (e) {
            console.warn('vivaz-loader-helper: fetch error for', url, e);
        }
    }
    return true;
}

export async function initVivaz() {
    if (isRuntimeReady) return exports;

    // Fetch manifest early to know which PESOS assets to mount
    const manifest = await fetchBootManifest();
    let assets = [];
    if (manifest && manifest.assets) {
        if (Array.isArray(manifest.assets)) {
            assets = manifest.assets;
        } else {
            assets = Object.keys(manifest.assets).map(n => ({ name: n }));
        }
    }

    const { getAssemblyExports, getConfig } = await dotnet
        .withDiagnosticTracing(false)
        .withApplicationArgumentsFromQuery()
        .create();

    // Wait for Emscripten-like FS to appear (dotnet runtime may attach it)
    const FS = await waitForFS(60000);
    if (FS) {
        try {
            await mountPesosIntoFS(FS, assets);
            // Quick verification: check for root PESOS dir and the preferred detector dir
            try {
                const statPesos = FS.stat('/PESOS');
                console.info('vivaz-loader-helper: FS.stat /PESOS ->', statPesos);
            } catch (e) {
                console.warn('vivaz-loader-helper: FS.stat /PESOS failed', e);
            }
            try {
                const statDet = FS.stat('/PESOS/CLASSIFICADOR_DETECTOR_LEVE_B');
                console.info('vivaz-loader-helper: FS.stat /PESOS/CLASSIFICADOR_DETECTOR_LEVE_B ->', statDet);
            } catch (e) {
                console.warn('vivaz-loader-helper: FS.stat /PESOS/CLASSIFICADOR_DETECTOR_LEVE_B failed', e);
            }
        } catch (e) {
            console.warn('vivaz-loader-helper: mounting PESOS failed', e);
        }
    } else {
        console.warn('vivaz-loader-helper: Module.FS did not appear within timeout; proceeding without mounting');
    }

    const config = getConfig();
    exports = await getAssemblyExports(config.mainAssemblyName);
    // Ask managed runtime to ensure PESOS files are available (will fetch and write into virtual FS)
    try {
        const pesosDir = '/PESOS/CLASSIFICADOR_DETECTOR_LEVE_B';
        if (exports && exports.Vivaz && exports.Vivaz.WASM && exports.Vivaz.WASM.VivazClient && typeof exports.Vivaz.WASM.VivazClient.EnsurePesosAvailable === 'function') {
            try {
                console.info('vivaz-loader-helper: requesting EnsurePesosAvailable for', pesosDir);
                await exports.Vivaz.WASM.VivazClient.EnsurePesosAvailable(pesosDir);
                console.info('vivaz-loader-helper: EnsurePesosAvailable completed');
            } catch (e) {
                console.warn('vivaz-loader-helper: EnsurePesosAvailable failed', e);
            }
        }
    } catch (e) { console.warn('vivaz-loader-helper: EnsurePesosAvailable check failed', e); }
    isRuntimeReady = true;
    return exports;
}
