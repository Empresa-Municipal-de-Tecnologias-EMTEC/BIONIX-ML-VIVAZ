using DetectorLeveBModel;
using Bionix.ML.computacao;
var builder = WebApplication.CreateBuilder(args);

// Enable detailed logging for local debugging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Debug);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddControllers();

// Register a singleton DetectorLeve instance at application level so all
// requests reuse the same model instance (weights loaded once).
builder.Services.AddSingleton(provider =>
{
    var env = provider.GetRequiredService<Microsoft.Extensions.Hosting.IHostEnvironment>();
    var pesosDir = Path.Combine(env.ContentRootPath ?? Directory.GetCurrentDirectory(), "PESOS", "CLASSIFICADOR_DETECTOR_LEVE");
    // Use CPU context here to avoid tensor type mismatches.
    ComputacaoContexto ctx = new ComputacaoCPUContexto();
    var det = DetectorLeve.GetInstance(ctx, pesosDir);
    return det;
});
// Allow all CORS for local demo usage (permite todos)
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

app.MapGet("/weatherforecast", () =>
{
    var forecast =  Enumerable.Range(1, 5).Select(index =>
        new WeatherForecast
        (
            DateOnly.FromDateTime(DateTime.Now.AddDays(index)),
            Random.Shared.Next(-20, 55),
            summaries[Random.Shared.Next(summaries.Length)]
        ))
        .ToArray();
    return forecast;
})
.WithName("GetWeatherForecast")
.WithOpenApi();

app.UseRouting();
// Simple request logging middleware to trace when requests arrive and finish.
app.Use(async (context, next) =>
{
    var logger = app.Services.GetRequiredService<Microsoft.Extensions.Logging.ILoggerFactory>().CreateLogger("RequestLogger");
    try
    {
        var req = context.Request;
        logger.LogInformation("[RequestLogger] Incoming {method} {path} from {remote} Content-Length={len}", req.Method, req.Path, context.Connection.RemoteIpAddress, req.ContentLength);
    }
    catch { }
    var sw = System.Diagnostics.Stopwatch.StartNew();
    await next();
    sw.Stop();
    try
    {
        var resp = context.Response;
        var logger2 = app.Services.GetRequiredService<Microsoft.Extensions.Logging.ILoggerFactory>().CreateLogger("RequestLogger");
        logger2.LogInformation("[RequestLogger] Finished {path} with {status} in {ms}ms", context.Request.Path, resp.StatusCode, sw.ElapsedMilliseconds);
    }
    catch { }
});
// Enable CORS policy to allow requests from the demo (and other origins)
app.UseCors("AllowAll");
app.UseEndpoints(endpoints =>
{
    endpoints.MapControllers();
    endpoints.MapFallback(() => Results.NotFound());
});

app.Run();

record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}
