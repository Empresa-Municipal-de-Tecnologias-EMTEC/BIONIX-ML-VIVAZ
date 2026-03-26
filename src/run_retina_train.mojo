import model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import retina.retina_trainer as trainer

fn main() raises -> None:
    print("Iniciando treino Retina minimal...")
    var ctx = contexto_defs.criar_contexto_padrao("cpu")
    # build detector CNN block (height,width,num_filters,kernel)
    var bloco = model_pkg.criar_bloco_detector(640, 640, 6, 3, 3, ctx)
    var result = trainer.treinar_retina_minimal(bloco, "DATASET", 640, 640, 64, 5, 0.0001)
    print("Treinador finalizou:", result)
