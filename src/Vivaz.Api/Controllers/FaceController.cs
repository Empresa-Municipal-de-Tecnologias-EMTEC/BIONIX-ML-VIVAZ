using Microsoft.AspNetCore.Mvc;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.dados.imagem;
using Bionix.ML.computacao;
using DetectorLeveBModel;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Vivaz.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class FaceController : ControllerBase
{
    [HttpPost("detect")]
    public async Task<IActionResult> Detect([FromForm] IFormFile image)
    {
        if (image == null) return BadRequest("image file required");
        using var ms = new MemoryStream();
        await image.CopyToAsync(ms);
        ms.Seek(0, SeekOrigin.Begin);

        // load image into ImageSharp and convert to BMP
        Image<Rgba32> img;
        try { img = Image.Load<Rgba32>(ms); }
        catch { return BadRequest("invalid image"); }
        var bmp = BMP.FromImage(img);

        // initialize compute context and detector (weights loaded from PESOS if present)
        ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
        var pesosDir = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "CLASSIFICADOR_DETECTOR_LEVE");
        var det = DetectorLeve.GetInstance(ctx, pesosDir);

        // run multi-scale top-K detection and aggregate consensus
        var allDet = det.DetectTopPerScale(bmp, ctx, detectCutoff: 0.5, scales: null, steps: null, topK: 5);
        var consensus = det.AggregateConsensus(allDet, bmp.Width, bmp.Height, 0.4);

        if (!consensus.found)
        {
            // return JSON with detections if nothing to crop
            var result = new { detections = allDet.Select(d => new { d.score, d.x, d.y, d.w, d.h, d.work }) };
            return Ok(result);
        }

        // create cropped image from BMP and return as PNG
        int cx = consensus.x, cy = consensus.y, cw = consensus.w, ch = consensus.h;
        // construct an Image<Rgba32> from BMP crop
        var cropBuf = new byte[cw * ch * bmp.QuantidadeCanais];
        for (int y = 0; y < ch; y++)
        {
            for (int x = 0; x < cw; x++)
            {
                int srcX = cx + x, srcY = cy + y;
                int srcIdx = (srcY * bmp.Width + srcX) * bmp.QuantidadeCanais;
                int dstIdx = (y * cw + x) * bmp.QuantidadeCanais;
                for (int c = 0; c < bmp.QuantidadeCanais; c++) cropBuf[dstIdx + c] = bmp.Armazenamento[srcIdx + c];
            }
        }
        // create Image<Rgba32>
        var outImg = new Image<Rgba32>(cw, ch);
        outImg.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < ch; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < cw; x++)
                {
                    int idx = (y * cw + x) * bmp.QuantidadeCanais;
                    byte r = cropBuf[idx + 0];
                    byte g = cropBuf[idx + 1];
                    byte b = cropBuf[idx + 2];
                    row[x] = new Rgba32(r, g, b, 255);
                }
            }
        });

        using var outMs = new MemoryStream();
        outImg.SaveAsPng(outMs);
        outMs.Seek(0, SeekOrigin.Begin);
        return File(outMs.ToArray(), "image/png", "crop.png");
    }

    [HttpPost("detectjson")]
    public async Task<IActionResult> DetectJson([FromForm] IFormFile image)
    {
        if (image == null) return BadRequest("image file required");
        using var ms = new MemoryStream();
        await image.CopyToAsync(ms);
        ms.Seek(0, SeekOrigin.Begin);
        Image<Rgba32> img;
        try { img = Image.Load<Rgba32>(ms); }
        catch { return BadRequest("invalid image"); }
        var bmp = BMP.FromImage(img);
        ComputacaoContexto ctx = new ComputacaoCPUSIMDContexto();
        var pesosDir = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "CLASSIFICADOR_DETECTOR_LEVE");
        var det = DetectorLeve.GetInstance(ctx, pesosDir);
        var allDet = det.DetectTopPerScale(bmp, ctx, detectCutoff: 0.5, scales: null, steps: null, topK: 5);
        var consensus = det.AggregateConsensus(allDet, bmp.Width, bmp.Height, 0.4);
        var result = new
        {
            detections = allDet.Select(d => new { d.score, d.x, d.y, d.w, d.h, d.work }),
            final = consensus.found ? new { consensus.x, consensus.y, consensus.w, consensus.h } : null
        };
        return Ok(result);
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
            return BadRequest(new { error = ex.Message });
        }
    }
}
