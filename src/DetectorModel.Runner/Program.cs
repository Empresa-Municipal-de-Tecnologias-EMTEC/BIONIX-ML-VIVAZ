using System;

namespace DetectorModel.Runner
{
    public static class Program
    {
        public static int Main(string[] args)
        {
            Console.WriteLine("DetectorModel.Runner starting...");
            try
            {
                DetectorModel.ExecutarTreinamento.Main(args);
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
