using System;
using Bionix.ML.computacao;

namespace IdentificadorLeveModel
{
    public class IdentificadorLeve
    {
        private readonly ComputacaoContexto _ctx;
        public IdentificadorLeve(ComputacaoContexto ctx)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
        }

        public void Initialize() { }

        public double[] Embed(object input)
        {
            // placeholder: return zero embedding
            return new double[128];
        }
    }
}
