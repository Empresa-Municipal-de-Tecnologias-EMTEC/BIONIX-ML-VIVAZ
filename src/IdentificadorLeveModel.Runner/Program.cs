using System;
using Bionix.ML.computacao;
using IdentificadorLeveModel;

namespace IdentificadorLeveModel.Runner
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            var computeEnv = Environment.GetEnvironmentVariable("COMPUTE") ?? "SIMD";
            ComputacaoContexto ctx = computeEnv.Equals("CPU", StringComparison.OrdinalIgnoreCase) ? (ComputacaoContexto)new ComputacaoCPUContexto() : new ComputacaoCPUSIMDContexto();
            try
            {
                Console.WriteLine("IdentificadorLeveModel Runner started");
                var model = new IdentificadorLeve(ctx);
                model.Initialize();
                Console.WriteLine("Initialized lightweight identifier (placeholder)");
            }
            finally { if (ctx is IDisposable d) d.Dispose(); }
        }
    }
}
