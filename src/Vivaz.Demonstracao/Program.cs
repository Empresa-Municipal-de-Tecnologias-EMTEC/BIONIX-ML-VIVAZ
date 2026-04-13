using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.FileProviders;
using System.IO;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddControllersWithViews();

var app = builder.Build();
if (!app.Environment.IsDevelopment()) app.UseExceptionHandler("/Home/Error");
app.UseStaticFiles();
// serve demo static files
var demoStatic = Path.Combine(builder.Environment.ContentRootPath, "wwwroot");
if (Directory.Exists(demoStatic))
{
	app.UseStaticFiles(new StaticFileOptions
	{
		FileProvider = new PhysicalFileProvider(demoStatic),
		RequestPath = ""
	});
}
app.UseRouting();
app.MapControllerRoute(name: "default", pattern: "{controller=Demo}/{action=Index}/{id?}");
app.Run();
