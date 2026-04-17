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
            return base.File(png, "image/png");
        }

        [HttpPost("embed")]
        public async Task<IActionResult> Embed()
        {
            var file = Request.Form.Files.Count > 0 ? Request.Form.Files[0] : null;
            if (file == null) return BadRequest("no file");
            using var ms = new MemoryStream();
            await file.CopyToAsync(ms);
            var bytes = ms.ToArray();
            var json = VivazClient.EmbedJson(bytes);
            return Content(json, "application/json");
        }

        [HttpPost("compare")]
        public async Task<IActionResult> Compare()
        {
            if (Request.Form.Files.Count < 2) return BadRequest("two files required");
            var fa = Request.Form.Files[0]; var fb = Request.Form.Files[1];
            using var msa = new MemoryStream(); await fa.CopyToAsync(msa);
            using var msb = new MemoryStream(); await fb.CopyToAsync(msb);
            var json = VivazClient.CompareJson(msa.ToArray(), msb.ToArray());
            return Content(json, "application/json");
        }
    }
}
