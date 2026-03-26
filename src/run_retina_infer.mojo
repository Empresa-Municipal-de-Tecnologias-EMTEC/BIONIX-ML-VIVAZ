import retina_infer as infer_pkg

fn main() -> None:
    print("Executando inferência Retina minimal...")
    var boxes = infer_pkg.inferir_retina("MODELO", 640, 16)
    print("Caixas detectadas:", boxes)
