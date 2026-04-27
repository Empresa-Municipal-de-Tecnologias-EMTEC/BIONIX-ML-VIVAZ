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
                
                var pesosDir = GetPesosDir("CLASSIFICADOR_DETECTOR_LEVE");
                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                var all = det.DetectTopPerScale(bmp, ctx, 0.5, null, null, 5);
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);
                
                return JsonSerializer.Serialize(new { 
                    found = cons.found, 
                    final = cons.found ? new { x = cons.x, y = cons.y, w = cons.w, h = cons.h } : null
                });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { found = false, error = ex.Message });
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
                var pesosDir = GetPesosDir("CLASSIFICADOR_DETECTOR_LEVE");
                var det = DetectorLeve.GetInstance(ctx, pesosDir);
                var all = det.DetectTopPerScale(bmp, ctx, 0.5, null, null, 5);
                var cons = det.AggregateConsensus(all, bmp.Width, bmp.Height, 0.4);
                
                return JsonSerializer.Serialize(new { 
                    found = cons.found, 
                    x = cons.x, y = cons.y, w = cons.w, h = cons.h 
                });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new { found = false, error = ex.Message });
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
                
                var pesosDir = GetPesosDir("CLASSIFICADOR_DETECTOR_LEVE");
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
                return JsonSerializer.Serialize(new { error = ex.Message });
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
