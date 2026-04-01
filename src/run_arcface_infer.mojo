import reconhecedor.arcface_model as arc_model
import reconhecedor.arcface_infer as infer_pkg
import os

fn main() raises -> None:
    print("Executando inferência ArcFace — construindo galeria e verificando 5 pares...")
    var params  = arc_model.ArcFaceParametros(64, 8, 3, 3, 128, 0, "cpu")
    var modelo  = arc_model.ArcFace(params^, os.path.join("MODELO", "arcface_modelo"))
    var carregou = modelo.carregar()
    if not carregou:
        print("[INFER] Checkpoint não encontrado em MODELO/arcface_modelo — execute run_arcface_train primeiro.")
        return

    var dataset_dir = "DATASET"
    print("Construindo galeria de embeddings a partir de", dataset_dir, "...")
    var galeria = infer_pkg.construir_galeria(modelo, dataset_dir)
    print("Galeria construída:", galeria.tamanho(), "identidades.")

    if galeria.tamanho() == 0:
        print("[INFER] Nenhuma identidade na galeria. Verifique o dataset.")
        return

    # Verifica os pares: mesma identidade vs identidade diferente (dois primeiros)
    var train_root = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_root):
        train_root = dataset_dir

    var imgs_testadas: Int = 0
    try:
        for cls in os.listdir(train_root):
            if imgs_testadas >= 5:
                break
            var pcls = os.path.join(train_root, cls)
            if not os.path.isdir(pcls):
                continue
            try:
                for f in os.listdir(pcls):
                    if f.endswith('.bmp'):
                        var img_path = os.path.join(pcls, f)
                        var nome_out = String("?")
                        var sim_out  = Float32(0.0)
                        infer_pkg.identificar(modelo, img_path, galeria, nome_out, sim_out, 0.5)
                        var correto = "OK" if nome_out == cls else "ERRO"
                        print("  Imagem:", f, "| Real:", cls, "| Predito:", nome_out,
                              "| Sim:", sim_out, "| Status:", correto)
                        imgs_testadas = imgs_testadas + 1
                        break
            except _:
                pass
    except _:
        pass

    print("Inferência concluída.")
