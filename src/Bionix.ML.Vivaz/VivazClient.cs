using System;
using Bionix.ML;

namespace Bionix.ML.Vivaz
{
    public class VivazClient
    {
        public string UseModel()
        {
            var m = new Model();
            return $"Vivaz usando: {m.Info()}";
        }
    }
}
