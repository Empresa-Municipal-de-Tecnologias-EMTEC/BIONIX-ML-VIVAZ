using System;
using System.IO;
using System.Text.Json;
using Bionix.ML;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.computacao;
using DetectorLeveModel;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Vivaz.WASM
{
    public class VivazClient
    {
        public string UseModel()
        {
            var m = new Model();
            return $"Vivaz usando: {m.Info()}";
        }

        // Detect and return consensus box as JSON string: { found:bool, x,y,w,h }
        private static string? ExtractEmbeddedWeightsIfPresent(out string? tempPesosDir)
        {
            tempPesosDir = null;
            try
            {
                var asm = typeof(VivazClient).Assembly;
                // resource path convention: Vivaz.WASM.pesos.CLASSIFICADOR_DETECTOR_LEVE.zip
                foreach (var name in asm.GetManifestResourceNames())
                {
                    if (name.ToLower().Contains("classificador_detector_leve"))
                    {
                        var outDir = Path.Combine(Path.GetTempPath(), "vivaz_pesos");
                        Directory.CreateDirectory(outDir);
                        var resFile = Path.Combine(outDir, "embedded_pesos.bin");
                        using var s = asm.GetManifestResourceStream(name);
                        if (s == null) continue;
                        using var fs = File.Create(resFile);
                        s.CopyTo(fs);
                        // If the resource is a zip/dir expected by GetInstance, try to unzip
                        try
                        {
                            // attempt to unzip (best-effort)
                            System.IO.Compression.ZipFile.ExtractToDirectory(resFile, Path.Combine(outDir, "CLASSIFICADOR_DETECTOR_LEVE"), true);
                            tempPesosDir = Path.Combine(outDir, "CLASSIFICADOR_DETECTOR_LEVE");
                            return tempPesosDir;
                        }
                        catch
                        {
                            // not a zip — leave single file; detector may support single-file weights
                            tempPesosDir = outDir;
                            return tempPesosDir;
                        }
                    }
                }
            }
            catch { }
            return null;
        }

        public static string DetectJson(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0) return JsonSerializer.Serialize(new { found = false });
            try
            {
                using var ms = new MemoryStream(imageBytes);
                var img = Image.Load<Rgba32>(ms);
                var bmp = BMP.FromImage(img);
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
                // prefer embedded weights if present
                string? tempPesos;
                var embedded = ExtractEmbeddedWeightsIfPresent(out tempPesos);
                var pesosDir = embedded ?? Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "CLASSIFICADOR_DETECTOR_LEVE");
                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                var all = det.DetectTopPerScale(bmp, ctx, 0.5, null, null, 5);
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);
                if (!cons.found) return JsonSerializer.Serialize(new { found = false });
                return JsonSerializer.Serialize(new { found = true, x = cons.x, y = cons.y, w = cons.w, h = cons.h });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { found = false, error = ex.Message });
            }
        }

        // Detect and return PNG bytes of the consensus crop. Returns null on failure.
        public static byte[] DetectCropPng(byte[] imageBytes)
        {
            if (imageBytes == null || imageBytes.Length == 0) return null;
            try
            {
                using var ms = new MemoryStream(imageBytes);
                var img = Image.Load<Rgba32>(ms);
                var bmp = BMP.FromImage(img);
                ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
                // prefer embedded weights if present
                string? tempPesos;
                var embedded = ExtractEmbeddedWeightsIfPresent(out tempPesos);
                var pesosDir = embedded ?? Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "CLASSIFICADOR_DETECTOR_LEVE");
                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                var all = det.DetectTopPerScale(bmp, ctx, 0.5, null, null, 5);
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);
                if (!cons.found) return null;
                int cx = cons.x, cy = cons.y, cw = cons.w, ch = cons.h;
                var outImg = new Image<Rgba32>(cw, ch);
                outImg.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < ch; y++)
                    {
                        var row = accessor.GetRowSpan(y);
                        for (int x = 0; x < cw; x++)
                        {
                            int sx = cx + x, sy = cy + y;
                            int srcIdx = (sy * bmp.Width + sx) * bmp.QuantidadeCanais;
                            byte r = bmp.Armazenamento[srcIdx + 0];
                            byte g = bmp.Armazenamento[srcIdx + 1];
                            byte b = bmp.Armazenamento[srcIdx + 2];
                            row[x] = new Rgba32(r, g, b, 255);
                        }
                    }
                });
                using var outMs = new MemoryStream();
                outImg.SaveAsPng(outMs);
                return outMs.ToArray();
            }
            catch { return null; }
        }
    }
}
