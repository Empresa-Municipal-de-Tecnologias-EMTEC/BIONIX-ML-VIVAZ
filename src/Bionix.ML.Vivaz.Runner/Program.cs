using System;

namespace Bionix.ML.Vivaz.Runner
{
    public static class Program
    {
        public static int Main(string[] args)
        {
            Console.WriteLine("Bionix.ML.Vivaz.Runner starting...");
            try
            {
                // Delegate to library entry (keeps training implementation inside DetectorModel/ExecutarTreinamento)
                Bionix.ML.Vivaz.Program.Main(args);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Runner error: {ex.Message}");
                return 1;
            }
            Console.WriteLine("Runner finished.");
            return 0;
        }
    }
}
