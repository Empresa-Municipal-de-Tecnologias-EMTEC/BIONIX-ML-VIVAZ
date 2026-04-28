using System;
using System.IO;
using System.Text.Json;
using System.Net.Http;
using System.Threading.Tasks;
using System.Runtime.InteropServices.JavaScript;
using Bionix.ML;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.computacao;
using DetectorLeveBModel;
using IdentificadorLeveModel;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Vivaz.WASM
{
    public partial class VivazClient
    {
        [JSExport]
        public static string GetInfo()
        {
            try 
            {
                var m = new Model();
                return $"Vivaz.WASM rodando Bionix.ML: {m.Info()}";
            }
            catch (Exception ex)
            {
                return $"Erro ao obter info: {ex.Message}";
            }
        }

        [JSExport]
        public static async System.Threading.Tasks.Task<string> EnsurePesosAvailable(string dirOrUrl)
        {
            try
            {
                Console.WriteLine($"[VivazClient] EnsurePesosAvailable called with: {dirOrUrl}");
                // Determine if argument is an absolute URL or a local path
                string webBase;
                string localBaseDir;
                if (Uri.TryCreate(dirOrUrl, UriKind.Absolute, out var parsed) && (parsed.Scheme == "http" || parsed.Scheme == "https"))
                {
                    // dirOrUrl is an absolute URL; use its origin as web base and derive local target from its absolute path
                    webBase = parsed.GetLeftPart(UriPartial.Authority) + parsed.AbsolutePath.TrimEnd('/');
                    localBaseDir = Path.Combine(Directory.GetCurrentDirectory(), parsed.AbsolutePath.TrimStart('/'));
                    Console.WriteLine($"[VivazClient] Parsed absolute URL. webBase={webBase}, localBaseDir={localBaseDir}");
                }
                else
                {
                        // Treat as local (possibly absolute or relative) path and build webBase as relative path from root
                        localBaseDir = dirOrUrl;
                        var webPath = localBaseDir.Replace('\\', '/').TrimEnd('/');
                        if (!webPath.StartsWith('/')) webPath = "/" + webPath;
                        // Use relative web path; HttpClient in WASM requires absolute URIs, so prefix with origin using JS -> loader should pass absolute URL when possible.
                        webBase = webPath; // may be relative; caller can pass absolute URL to avoid this
                        Console.WriteLine($"[VivazClient] Using relative webBase={webBase}, localBaseDir={localBaseDir}");
                }

                // If local dir already has files, skip
                try { if (Directory.Exists(localBaseDir) && Directory.GetFiles(localBaseDir).Length > 0) return "ok"; } catch { }

                var files = new[] { "w1.bin", "b1.bin", "w2.bin", "b2.bin", "convW.bin", "convB.bin", "meta.json", "opt_meta.json" };
                using var client = new HttpClient();

                foreach (var f in files)
                {
                    string url;
                    if (webBase.StartsWith("http://") || webBase.StartsWith("https://")) url = webBase.TrimEnd('/') + "/" + f;
                    else
                    {
                        // If webBase is relative, try to construct absolute using window origin is not available here; log and continue
                        Console.WriteLine($"[VivazClient] Skipping GET for relative web path: {webBase}/{f} (no origin available)");
                        continue;
                    }

                    try
                    {
                        var bytes = await client.GetByteArrayAsync(url);
                        var targetPath = Path.Combine(localBaseDir, f);
                        var targetDir = Path.GetDirectoryName(targetPath) ?? ".";
                        if (!Directory.Exists(targetDir)) Directory.CreateDirectory(targetDir);
                        File.WriteAllBytes(targetPath, bytes);
                        Console.WriteLine($"[VivazClient] Wrote {targetPath} ({bytes.Length} bytes)");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[VivazClient] Failed GET {url}: {ex.Message}");
                    }
                }

                return "done";
            }
            catch (Exception ex)
            {
                Console.WriteLine("[VivazClient] EnsurePesosAvailable error: " + ex.ToString());
                return "error";
            }
        }

        private static string GetPesosDir(string name)
        {
            // No WASM, os pesos devem estar na pasta virtual /PESOS
            var basePesos = Path.Combine(Directory.GetCurrentDirectory(), "PESOS");

            // Prefer the newer _B classifier if available
            var preferredB = Path.Combine(basePesos, "CLASSIFICADOR_DETECTOR_LEVE_B");
            // Directory.Exists can be unreliable inside WASM virtual FS; also check for known files
            if (Directory.Exists(preferredB) || File.Exists(Path.Combine(preferredB, "w1.bin")) || File.Exists(Path.Combine(preferredB, "meta.json")))
                return preferredB;

            // Otherwise try the requested name
            var pesosDir = Path.Combine(basePesos, name);
            if (Directory.Exists(pesosDir) || File.Exists(Path.Combine(pesosDir, "w1.bin")) || File.Exists(Path.Combine(pesosDir, "meta.json")))
                return pesosDir;

            // Backwards-compat fallback: if requesting CLASSIFICADOR_DETECTOR_LEVE and it's missing,
            // still allow using CLASSIFICADOR_DETECTOR_LEVE_B when present (already checked above)
            return pesosDir;
        }

        [JSExport]
        public static string DetectJson(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0) 
                return JsonSerializer.Serialize(new { found = false, error = "Dados de imagem vazios" });

            try
            {
                using var ms = new MemoryStream(imageBytes);
                var img = Image.Load<Rgba32>(ms);
                var bmp = BMP.FromImage(img);
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
                
                var pesosDir = GetPesosDir("CLASSIFICADOR_DETECTOR_LEVE_B");
                Console.WriteLine($"[VivazClient] DetectJson: ctx={ctx?.GetType().Name}, pesosDir={pesosDir}");
                try
                {
                    if (Directory.Exists(pesosDir))
                    {
                        var fls = Directory.GetFiles(pesosDir);
                        Console.WriteLine($"[VivazClient] DetectJson: pesosDir contains {fls.Length} files");
                        foreach (var f in fls) Console.WriteLine($"[VivazClient]   - {Path.GetFileName(f)}");
                    }
                    else Console.WriteLine($"[VivazClient] DetectJson: pesosDir does not exist: {pesosDir}");
                }
                catch (Exception ex) { Console.WriteLine("[VivazClient] Inspect pesosDir failed: " + ex.ToString()); }

                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                try
                {
                    foreach (var np in det.GetNamedParameters())
                    {
                        try { Console.WriteLine($"[VivazClient] Param {np.name}: type={np.tensor?.GetType().Name}, shape=[{string.Join(',', np.tensor?.Shape ?? new int[0])}], size={np.tensor?.Size}"); } catch { }
                    }
                }
                catch (Exception ex) { Console.WriteLine("[VivazClient] Failed to enumerate parameters: " + ex.ToString()); }
                var all = det.DetectTopPerScale(bmp, ctx, 0.7, null, null, 5);
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);
                
                return JsonSerializer.Serialize(new { 
                    found = cons.found, 
                    final = cons.found ? new { x = cons.x, y = cons.y, w = cons.w, h = cons.h } : null
                });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { found = false, error = ex.ToString() });
            }
        }

        [JSExport]
        public static string DetectFromRgb(int width, int height, byte[] rgbBytes)
        {
            if (rgbBytes == null || rgbBytes.Length == 0) return JsonSerializer.Serialize(new { found = false });
            try
            {
                var bmp = new Bionix.ML.dados.imagem.bmp.BMP(width, height, 3, rgbBytes);
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
                var pesosDir = GetPesosDir("CLASSIFICADOR_DETECTOR_LEVE_B");
                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                var all = det.DetectTopPerScale(bmp, ctx, 0.5, null, null, 5);
                try
                {
                    if (all == null)
                    {
                        Console.WriteLine("[VivazClient] DetectTopPerScale returned null");
                    }
                    else
                    {
                        Console.WriteLine($"[VivazClient] total candidates: {all.Count}");
                        for (int k = 0; k < Math.Min(10, all.Count); k++)
                        {
                            try
                            {
                                Console.WriteLine($"[VivazClient] candidate[{k}]: {all[k]}");
                            }
                            catch { }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("[VivazClient] Error while enumerating candidates: " + ex.ToString());
                }
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);
                
                return JsonSerializer.Serialize(new { 
                    found = cons.found, 
                    x = cons.x, y = cons.y, w = cons.w, h = cons.h 
                });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { found = false, error = ex.ToString() });
            }
        }

        [JSExport]
        public static byte[]? DetectCrop(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0) return null;
            try
            {
                using var ms = new MemoryStream(imageBytes);
                var img = Image.Load<Rgba32>(ms);
                var bmp = BMP.FromImage(img);
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
                
                var pesosDir = GetPesosDir("CLASSIFICADOR_DETECTOR_LEVE_B");
                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                var all = det.DetectTopPerScale(bmp, ctx, 0.5, null, null, 5);
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);
                
                if (!cons.found) return null;

                int cx = Math.Max(0, cons.x); 
                int cy = Math.Max(0, cons.y);
                int cw = Math.Min(cons.w, bmp.Width - cx); 
                int ch = Math.Min(cons.h, bmp.Height - cy);

                var outImg = new Image<Rgba32>(cw, ch);
                for (int y = 0; y < ch; y++)
                {
                    for (int x = 0; x < cw; x++)
                    {
                        int sx = cx + x, sy = cy + y;
                        int srcIdx = (sy * bmp.Width + sx) * bmp.QuantidadeCanais;
                        byte r = bmp.Armazenamento[srcIdx + 0];
                        byte g = bmp.Armazenamento[srcIdx + 1];
                        byte b = bmp.Armazenamento[srcIdx + 2];
                        outImg[x, y] = new Rgba32(r, g, b, 255);
                    }
                }

                using var outMs = new MemoryStream();
                outImg.SaveAsPng(outMs);
                return outMs.ToArray();
            }
            catch { return null; }
        }

        [JSExport]
        public static byte[]? DetectCropPng(byte[] imageBytes)
        {
            return DetectCrop(imageBytes);
        }

        [JSExport]
        public static byte[]? DetectCropBmp(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0)
            {
                Console.WriteLine("[VivazClient] DetectCropBmp: empty imageBytes");
                return null;
            }

            try
            {
                Console.WriteLine($"[VivazClient] DetectCropBmp called: {imageBytes.Length} bytes");
                using var ms = new MemoryStream(imageBytes);
                var fmt = Image.DetectFormat(ms);
                ms.Position = 0;
                Console.WriteLine($"[VivazClient] Detected image format: {fmt?.Name ?? "unknown"}");
                var img = Image.Load<Rgba32>(ms);
                Console.WriteLine($"[VivazClient] Loaded image: {img.Width}x{img.Height}");
                var bmp = BMP.FromImage(img);
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();

                var pesosDir = GetPesosDir("CLASSIFICADOR_DETECTOR_LEVE_B");
                Console.WriteLine($"[VivazClient] Using pesosDir: {pesosDir}");
                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                var all = det.DetectTopPerScale(bmp, ctx, 0.5, null, null, 5);
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);

                Console.WriteLine($"[VivazClient] Detection consensus: found={cons.found}, x={cons.x}, y={cons.y}, w={cons.w}, h={cons.h}");

                if (!cons.found)
                {
                    Console.WriteLine("[VivazClient] DetectCropBmp: no face found");
                    return null;
                }

                int cx = Math.Max(0, cons.x);
                int cy = Math.Max(0, cons.y);
                int cw = Math.Min(cons.w, bmp.Width - cx);
                int ch = Math.Min(cons.h, bmp.Height - cy);

                var outImg = new Image<Rgba32>(cw, ch);
                for (int y = 0; y < ch; y++)
                {
                    for (int x = 0; x < cw; x++)
                    {
                        int sx = cx + x, sy = cy + y;
                        int srcIdx = (sy * bmp.Width + sx) * bmp.QuantidadeCanais;
                        byte r = bmp.Armazenamento[srcIdx + 0];
                        byte g = bmp.Armazenamento[srcIdx + 1];
                        byte b = bmp.Armazenamento[srcIdx + 2];
                        outImg[x, y] = new Rgba32(r, g, b, 255);
                    }
                }

                using var outMs = new MemoryStream();
                outImg.SaveAsBmp(outMs);
                var outBytes = outMs.ToArray();
                Console.WriteLine($"[VivazClient] DetectCropBmp: returning BMP {outBytes.Length} bytes");
                return outBytes;
            }
            catch (Exception ex)
            {
                Console.WriteLine("[VivazClient] DetectCropBmp error: " + ex.ToString());
                return null;
            }
        }

        // New: decode the provided image bytes using the same pipeline and return the full BMP bytes (no detection)
        [JSExport]
        public static byte[]? DetectDecodeBmp(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0)
            {
                Console.WriteLine("[VivazClient] DetectDecodeBmp: empty imageBytes");
                return null;
            }

            try
            {
                Console.WriteLine($"[VivazClient] DetectDecodeBmp called: {imageBytes.Length} bytes");
                using var ms = new MemoryStream(imageBytes);
                var fmt = Image.DetectFormat(ms);
                ms.Position = 0;
                Console.WriteLine($"[VivazClient] DetectDecodeBmp detected format: {fmt?.Name ?? "unknown"}");
                var img = Image.Load<Rgba32>(ms);
                Console.WriteLine($"[VivazClient] DetectDecodeBmp loaded image: {img.Width}x{img.Height}");
                var bmp = BMP.FromImage(img);

                using var outMs = new MemoryStream();
                outMs.Position = 0;
                // Save as BMP using ImageSharp to preserve exact decoded pixels
                img.SaveAsBmp(outMs);
                var bytes = outMs.ToArray();
                Console.WriteLine($"[VivazClient] DetectDecodeBmp: returning BMP {bytes.Length} bytes");
                return bytes;
            }
            catch (Exception ex)
            {
                Console.WriteLine("[VivazClient] DetectDecodeBmp error: " + ex.ToString());
                return null;
            }
        }

        [JSExport]
        public static string EmbedJson(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0) 
                return JsonSerializer.Serialize(new { error = "no data" });
            try
            {
                using var ms = new MemoryStream(imageBytes);
                var img = Image.Load<Rgba32>(ms);
                var bmp = BMP.FromImage(img);
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
                
                var pesosDir = GetPesosDir("IDENTIFICADOR_LEVE");
                var ident = IdentificadorLeve.GetInstance(ctx, pesosDir);
                var embT = ident.EmbedFromBmp(bmp, ctx);
                var arr = ident.EmbeddingToArray(embT, l2Normalize: true);
                
                return JsonSerializer.Serialize(new { embedding = arr });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { error = ex.ToString() });
            }
        }

        [JSExport]
        public static string CompareJson(byte[] aBytes, byte[] bBytes, double threshold = 0.7)
        {
            if (aBytes == null || bBytes == null) 
                return JsonSerializer.Serialize(new { error = "two images required" });
            try
            {
                using var msa = new MemoryStream(aBytes);
                using var msb = new MemoryStream(bBytes);
                var imga = Image.Load<Rgba32>(msa);
                var imgb = Image.Load<Rgba32>(msb);
                var bmpa = BMP.FromImage(imga);
                var bmpb = BMP.FromImage(imgb);
                
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
                var pesosDir = GetPesosDir("IDENTIFICADOR_LEVE");
                var ident = IdentificadorLeve.GetInstance(ctx, pesosDir);
                
                var embA = ident.EmbedFromBmp(bmpa, ctx);
                var embB = ident.EmbedFromBmp(bmpb, ctx);
                var arrA = ident.EmbeddingToArray(embA, true);
                var arrB = ident.EmbeddingToArray(embB, true);
                
                var (percent, same) = IdentificadorLeve.CompareEmbeddings(arrA, arrB, threshold);
                return JsonSerializer.Serialize(new { percent = percent, same = same });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { error = ex.Message });
            }
        }
    }
}
