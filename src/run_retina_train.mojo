import model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import retina.retina_trainer as trainer

fn main() raises -> None:
    print("Iniciando treino Retina minimal...")
    var ctx = contexto_defs.criar_contexto_padrao("cpu")
    # build detector CNN block (height,width,num_filters,kernel) - reduced sizes to avoid OOM
    var bloco = model_pkg.criar_bloco_detector(320, 320, 6, 3, 3, ctx)
    var result = trainer.treinar_retina_minimal(bloco, "DATASET", 320, 320, 32, 3, 0.0001, 2)
    print("Treinador finalizou:", result)
