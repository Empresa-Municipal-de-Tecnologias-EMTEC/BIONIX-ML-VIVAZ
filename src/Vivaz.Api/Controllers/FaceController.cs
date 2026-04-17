using Microsoft.AspNetCore.Mvc;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.dados.imagem;
using Bionix.ML.computacao;
using DetectorLeveBModel;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Linq;
// avoid importing System.IO to prevent name conflicts with ControllerBase.File
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace Vivaz.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class FaceController : ControllerBase
{
    private readonly ILogger<FaceController> _log;
    private readonly DetectorLeve _detector;
    public FaceController(ILogger<FaceController> log, DetectorLeve detector)
    {
        _log = log;
        _detector = detector;
    }
    [HttpPost("detect")]
    public async Task<IActionResult> Detect([FromForm] IFormFile image)
    {
        if (image == null) return BadRequest("image file required");
        _log.LogDebug("Detect: received image length {len}", image.Length);
        using var ms = new MemoryStream();
        await image.CopyToAsync(ms);
        ms.Seek(0, SeekOrigin.Begin);
        // load image into ImageSharp and convert to BMP
        Image<Rgba32> img;
        try { img = Image.Load<Rgba32>(ms); }
        catch (Exception ex) { _log.LogWarning(ex, "invalid image payload"); return BadRequest("invalid image"); }
        var bmp = BMP.FromImage(img);

        // create a matching CPU compute context per-request and use the application-level detector
        ComputacaoContexto ctx = new ComputacaoCPUContexto();
        _log.LogDebug("Detect: using per-request CPU context and application-level detector");

        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // Delegate resizing + detection to model helper to avoid sliding-window on huge images
            var best = _detector.DetectBestResized(bmp, ctx, detectCutoff: 0.3, scales: null, steps: null, maxDim: 800);
            sw.Stop();
            _log.LogInformation("Detect: DetectBestResized finished in {ms}ms, score={score}", sw.ElapsedMilliseconds, best.score);

            if (double.IsNegativeInfinity(best.score) || double.IsNaN(best.score))
            {
                return Ok(new { detections = Enumerable.Empty<object>() });
            }

            int cx = Math.Max(0, best.x);
            int cy = Math.Max(0, best.y);
            int cw = Math.Max(1, best.w);
            int ch = Math.Max(1, best.h);

            using var outImg = img.Clone(c => c.Crop(new SixLabors.ImageSharp.Rectangle(cx, cy, cw, ch)));
            using var outMs = new MemoryStream();
            outImg.SaveAsJpeg(outMs);
            outMs.Seek(0, SeekOrigin.Begin);
            return base.File(outMs.ToArray(), "image/jpeg", "crop.jpg");
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Detect: exception during detection/crop");
            return StatusCode(500, new { error = "internal error" });
        }
    }

    [HttpPost("detectjson")]
    public async Task<IActionResult> DetectJson([FromForm] IFormFile image)
    {
        if (image == null) return BadRequest("image file required");
        _log.LogDebug("DetectJson: received image length {len}", image.Length);
        using var ms = new MemoryStream();
        await image.CopyToAsync(ms);
        ms.Seek(0, SeekOrigin.Begin);
        Image<Rgba32> img;
        try { img = Image.Load<Rgba32>(ms); }
        catch (Exception ex) { _log.LogWarning(ex, "invalid image payload"); return BadRequest("invalid image"); }
        var bmp = BMP.FromImage(img);
        // Use CPU context to avoid tensor type mismatches; use application-level detector
        ComputacaoContexto ctx = new ComputacaoCPUContexto();
        _log.LogDebug("DetectJson: using application-level DetectorLeve singleton");
        var swTotal = System.Diagnostics.Stopwatch.StartNew();
        try {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            _log.LogDebug("DetectJson: starting DetectTopPerScale");
            var allDet = _detector.DetectTopPerScale(bmp, ctx, detectCutoff: 0.3, scales: null, steps: null, topK: 5);
            _log.LogDebug("DetectJson: DetectTopPerScale finished in {ms}ms", sw.ElapsedMilliseconds);

            // Save `allDet` as JSON for debugging/inspection in SAIDA/debug_images
            try
            {
                var outDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA", "debug_images");
                Directory.CreateDirectory(outDir);
                var simple = allDet.Select(d => new { d.score, d.x, d.y, d.w, d.h, d.work }).ToList();
                var fname = Path.Combine(outDir, $"allDet_{DateTime.Now:yyyyMMdd_HHmmssfff}.json");
                var json = JsonSerializer.Serialize(simple, new JsonSerializerOptions { WriteIndented = true });
                System.IO.File.WriteAllText(fname, json);
                _log.LogInformation("DetectJson: saved allDet JSON to {file}", fname);
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "DetectJson: failed saving allDet JSON");
            }
            var consensus = _detector.AggregateConsensus(allDet, bmp.Width, bmp.Height, 0.4);
            _log.LogDebug("DetectJson: AggregateConsensus finished in {ms}ms", sw.ElapsedMilliseconds);
            swTotal.Stop();
            _log.LogInformation("DetectJson: total processing time {ms}ms", swTotal.ElapsedMilliseconds);

            var result = new
            {
                detections = allDet.Select(d => new { d.score, d.x, d.y, d.w, d.h, d.work }),
                final = consensus.found ? new { consensus.x, consensus.y, consensus.w, consensus.h } : null
            };
            return Ok(result);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "DetectJson: exception during detection");
            return StatusCode(500, new { error = "internal error" });
        }
        
    }

    [HttpPost("detect_scalesweep")]
    public async Task<IActionResult> DetectScaleSweep([FromForm] IFormFile image, [FromForm] string scales = null)
    {
        if (image == null) return BadRequest("image file required");
        using var ms = new MemoryStream();
        await image.CopyToAsync(ms);
        ms.Seek(0, SeekOrigin.Begin);
        Image<Rgba32> img;
        try { img = Image.Load<Rgba32>(ms); }
        catch (Exception ex) { _log.LogWarning(ex, "invalid image payload"); return BadRequest("invalid image"); }
        var bmp = BMP.FromImage(img);
        ComputacaoContexto ctx = new ComputacaoCPUContexto();

        // default candidate scales to test (including 20px anchor)
        int[] candidate = new int[] { 12, 16, 20, 24, 28, 32, 40, 48, 64 };
        if (!string.IsNullOrEmpty(scales))
        {
            try
            {
                var parts = scales.Split(new char[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var list = new System.Collections.Generic.List<int>();
                foreach (var p in parts) if (int.TryParse(p.Trim(), out var v)) list.Add(Math.Max(1, v));
                if (list.Count > 0) candidate = list.ToArray();
            }
            catch { }
        }

        var outDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA", "debug_images"); Directory.CreateDirectory(outDir);
        var report = new System.Collections.Generic.List<object>();

        foreach (var s in candidate)
        {
            try
            {
                // run detection using a single-scale sweep so windows of size `s` map to 20px when resized
                var dets = _detector.DetectTopPerScale(bmp, ctx, detectCutoff: 0.3, scales: new int[] { s }, steps: new int[] { Math.Max(1, s / 4) }, topK: 20);
                var simple = dets.Select(d => new { d.score, d.x, d.y, d.w, d.h, d.work }).ToList();
                var fname = Path.Combine(outDir, $"scale_{s}_allDet_{DateTime.Now:yyyyMMdd_HHmmssfff}.json");
                var json = JsonSerializer.Serialize(simple, new JsonSerializerOptions { WriteIndented = true });
                System.IO.File.WriteAllText(fname, json);
                _log.LogInformation("DetectScaleSweep: saved scale {scale} results to {file}", s, fname);
                report.Add(new { scale = s, count = simple.Count, file = fname, examples = simple.Take(5) });
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "DetectScaleSweep: scale {scale} failed", s);
                report.Add(new { scale = s, error = ex.Message });
            }
        }

        return Ok(new { tested = candidate, results = report });
    }

    [HttpPost("compare")]
    public async Task<IActionResult> Compare([FromForm] IFormFile a, [FromForm] IFormFile b)
    {
        if (a == null || b == null) return BadRequest("two image files required");
        using var msa = new MemoryStream(); await a.CopyToAsync(msa);
        using var msb = new MemoryStream(); await b.CopyToAsync(msb);

        try
        {
            msa.Seek(0, SeekOrigin.Begin); msb.Seek(0, SeekOrigin.Begin);
            Image<Rgba32> imga = Image.Load<Rgba32>(msa);
            Image<Rgba32> imgb = Image.Load<Rgba32>(msb);
            var bmpa = BMP.FromImage(imga);
            var bmpb = BMP.FromImage(imgb);
            ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
            var pesosDir = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "IDENTIFICADOR_LEVE");
            var ident = IdentificadorLeveModel.IdentificadorLeve.GetInstance(ctx, pesosDir);
            var embA = ident.EmbedFromBmp(bmpa, ctx);
            var embB = ident.EmbedFromBmp(bmpb, ctx);
            var arrA = ident.EmbeddingToArray(embA, true);
            var arrB = ident.EmbeddingToArray(embB, true);
            var (percent, same) = IdentificadorLeveModel.IdentificadorLeve.CompareEmbeddings(arrA, arrB, 0.7);
            return Ok(new { same = same, percent = percent });
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Error comparing images");
            return StatusCode(500, new { error = "internal error" });
        }
    }
}
