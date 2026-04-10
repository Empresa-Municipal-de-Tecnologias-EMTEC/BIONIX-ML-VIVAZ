using System;
using DetectorModel;

namespace Bionix.ML.Vivaz
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Bionix.ML.Vivaz runner starting: invoking DetectorModel ExecutarTreinamento...");
            try
            {
                ExecutarTreinamento.Main(args);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao executar treinamento: {ex.Message}");
            }
            Console.WriteLine("Runner finished.");
        }
    }
}
