using Microsoft.AspNetCore.Mvc;

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
        var bytes = ms.ToArray();

        // TODO: load detector model and run inference via Bionix.ML
        // For now return a stubbed detection
        var result = new
        {
            detections = new[] {
                new { x=10, y=20, w=100, h=120, score=0.98 }
            }
        };
        return Ok(result);
    }

    [HttpPost("compare")]
    public async Task<IActionResult> Compare([FromForm] IFormFile a, [FromForm] IFormFile b)
    {
        if (a == null || b == null) return BadRequest("two image files required");
        using var msa = new MemoryStream(); await a.CopyToAsync(msa);
        using var msb = new MemoryStream(); await b.CopyToAsync(msb);

        // TODO: run recognition and compute similarity via Bionix.ML
        var response = new { same = false, score = 0.42 };
        return Ok(response);
    }
}
