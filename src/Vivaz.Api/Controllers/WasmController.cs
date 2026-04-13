using Microsoft.AspNetCore.Mvc;
using System.IO;
using System.Threading.Tasks;
using Vivaz.WASM;

namespace Vivaz.Api.Controllers
{
    [ApiController]
    [Route("api/face/[controller]")]
    public class WasmController : ControllerBase
    {
        [HttpPost("detectjson")]
        public async Task<IActionResult> DetectJson()
        {
            var file = Request.Form.Files.Count > 0 ? Request.Form.Files[0] : null;
            if (file == null) return BadRequest("no file");
            using var ms = new MemoryStream();
            await file.CopyToAsync(ms);
            var bytes = ms.ToArray();
            var json = VivazClient.DetectJson(bytes);
            return Content(json, "application/json");
        }

        [HttpPost("detectcrop")]
        public async Task<IActionResult> DetectCrop()
        {
            var file = Request.Form.Files.Count > 0 ? Request.Form.Files[0] : null;
            if (file == null) return BadRequest("no file");
            using var ms = new MemoryStream();
            await file.CopyToAsync(ms);
            var bytes = ms.ToArray();
            var png = VivazClient.DetectCropPng(bytes);
            if (png == null) return NotFound();
            return File(png, "image/png");
        }
    }
}
