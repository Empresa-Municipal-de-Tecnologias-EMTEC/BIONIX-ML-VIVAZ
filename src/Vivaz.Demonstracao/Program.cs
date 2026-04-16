using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.FileProviders;
using Microsoft.AspNetCore.StaticFiles;
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
	var provider = new PhysicalFileProvider(demoStatic);
	var contentTypeProvider = new FileExtensionContentTypeProvider();
	// Ensure .dll/.pdb (and any other unknown extensions) are served as binary
	contentTypeProvider.Mappings[".dll"] = "application/octet-stream";
	contentTypeProvider.Mappings[".pdb"] = "application/octet-stream";
	app.UseStaticFiles(new StaticFileOptions
	{
		FileProvider = provider,
		RequestPath = "",
		ContentTypeProvider = contentTypeProvider
	});
}
app.UseRouting();
app.MapControllerRoute(name: "default", pattern: "{controller=Demo}/{action=Index}/{id?}");
app.Run();
