using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.imagem;
using IdentificadorModel.modelo;
using Bionix.ML.nucleo.funcoesPerda;
using Bionix.ML.dados.serializacao;
using Bionix.ML.nucleo.otimizadores;

namespace IdentificadorModel.Runner
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            if (args == null || args.Length == 0)
            {
                Console.WriteLine("Usage: train <identities_root_folder> [--epochs N] [--lr LR]");
                return;
            }
            var cmd = args[0].ToLowerInvariant();
            if (cmd == "train" && args.Length >= 2)
            {
                var folder = args[1];
                int epochs = 5;
                double lr = 1e-3;
                for (int i = 2; i < args.Length; i++)
                {
                    if (args[i] == "--epochs" && i + 1 < args.Length) { int.TryParse(args[i + 1], out epochs); i++; }
                    if (args[i] == "--lr" && i + 1 < args.Length) { double.TryParse(args[i + 1], out lr); i++; }
                }
                var hp = new IdentificadorModel.ExecutarTreinamento.HyperParameters { NumEpochs = epochs, InitialLearningRate = lr };
                // create compute context similar to other runners
                var computeEnv = Environment.GetEnvironmentVariable("COMPUTE") ?? "SIMD";
                ComputacaoContexto ctx = computeEnv.Equals("CPU", StringComparison.OrdinalIgnoreCase) ? (ComputacaoContexto)new ComputacaoCPUContexto() : new ComputacaoCPUSIMDContexto();
                try
                {
                    IdentificadorModel.ExecutarTreinamento.treinar(hp, new string[] { folder }, ctx);
                }
                finally
                {
                    if (ctx is IDisposable d) d.Dispose();
                }
                return;
            }
            Console.WriteLine("Unknown command");
        }

        // Note: training is delegated to IdentificadorModel.ExecutarTreinamento.treinar
    }
}
