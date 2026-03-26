import retina.retina_infer as infer_pkg
import retina.retina_model as model_utils
import bionix_ml.dados as dados_pkg
import bionix_ml.graficos as graficos_pkg
import bionix_ml.uteis as uteis
import os

fn main() raises -> None:
    print("Executando inferência Retina minimal para 10 primeiras classes...")
    var model_root = os.path.join("MODELO", "retina_modelo")
    var out_dir = os.path.join(model_root, "validacao_inferencia")
    try:
        infer_pkg.validar_10_classes(model_root, "DATASET", out_dir, 640, 8, 10)
    except _:
        print("[RUN] exceção ao executar validar_10_classes")
    print("Inferência de validação concluída. Saída:", out_dir)
