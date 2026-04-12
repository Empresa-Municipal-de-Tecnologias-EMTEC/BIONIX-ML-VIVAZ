using System;
using Bionix.ML;

namespace Vivaz.WASM
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
