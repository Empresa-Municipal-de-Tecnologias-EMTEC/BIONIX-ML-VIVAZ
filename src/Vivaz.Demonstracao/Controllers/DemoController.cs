using Microsoft.AspNetCore.Mvc;

namespace Vivaz.Demonstracao.Controllers;

public class DemoController : Controller
{
    public IActionResult Index()
    {
        return View();
    }

    public IActionResult Api()
    {
        return View();
    }

    public IActionResult Wasm()
    {
        return View();
    }
}
