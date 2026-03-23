# Ponto de entrada de treino (Mojo) - produção POC
# Uso: mojo -I ../../BIONIX-ML src/treino.mojo


import config as cfg
import io_modelo as io
import amostrador as amostrador_pkg
import pre_processamento as preprocess_pkg
import extrair_embeddings as embeddings_pkg

def main():
    # Configurações básicas

    var model_dir = cfg.MODEL_DIR
    var meta_path = model_dir + "/metadata.txt"
    var weights_path = model_dir + "/weights.bin"

    print("Iniciando treino POC (só um esqueleto)...")

    # Loop de época (stub)
    for epoch in range(1, cfg.EPOCHS + 1):
        print("Epoch " + String(epoch) + " / " + String(cfg.EPOCHS))
        var batch = amostrador_pkg.sample_epoch(cfg.DATASET_ROOT, cfg.IDENTITIES_PER_EPOCH, cfg.IMAGES_PER_IDENTITY)
        var train_similarities = List[Float32]()
        samples = 0
        for pair in batch:
            identity = pair[0]
            img_paths = pair[1]
            # simula processamento de imagens por identidade
            embs = List[List[Float32]]()
            for p in img_paths:
                img = preprocess_pkg.load_and_preprocess(p, cfg.INPUT_SHORT_SIDE)
                emb = embeddings_pkg.extract_embedding(p)
                embs.append(emb)
                samples += 1
            # calcula similaridade média intra-identity como proxy de "treino"
            if len(embs) >= 2:
                ssum = Float32(0.0)
                pairs = Int(0)
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        # cosine
                        a = embs[i]
                        b = embs[j]
                        var num = Float32(0.0)
                        var sa = Float32(0.0)
                        var sb = Float32(0.0)
                        for k in range(len(a)):
                            var x = a[k]
                            var y = b[k]
                            num = num + Float32(x) * Float32(y)
                            sa = sa + Float32(x) * Float32(x)
                            sb = sb + Float32(y) * Float32(y)
                        var denom = Float32(1.0)
                        if sa > Float32(0.0) and sb > Float32(0.0):
                            denom = (sa**0.5) * (sb**0.5)
                        ssum = ssum + (num / denom)
                        pairs = pairs + 1
                if pairs>0:
                    train_similarities.append(Float32(ssum / Float32(pairs)))

        # proxy metrics
        var avg_train_sim = Float32(0.0)
        if len(train_similarities) > 0:
            var total_sim = Float32(0.0)
            for v in train_similarities:
                total_sim = total_sim + v
            avg_train_sim = total_sim / Float32(len(train_similarities))
        print("  Samples this epoch: " + String(samples))
        print("  Avg intra-identity similarity (proxy): " + String(avg_train_sim))

        # Validação simples: para cada identidade em val, compare duas imagens se existirem
        import os
        var val_root = cfg.DATASET_ROOT + "/val"
        var val_idents_py = List[String]()
        if os.path.exists(val_root):
            var nomes = os.listdir(val_root)
            for n in nomes:
                var p = val_root + "/" + n
                if os.path.isdir(p):
                    val_idents_py.append(p)
        var tp = 0
        var tn = 0
        var fp = 0
        var fnn = 0
        var val_samples = 0
        for ident in val_idents_py:
            var files = List[String]()
            var nomes_f = os.listdir(ident)
            for fname in nomes_f:
                var p = ident + "/" + fname
                if os.path.isfile(p):
                    files.append(p)
            if len(files) >= 2:
                var db = files[0]
                var cam = files[1]
                emb_db = embeddings_pkg.extract_embedding(db)
                emb_cam = embeddings_pkg.extract_embedding(cam)
                # compute score (use Float32 accumulators to avoid dtype mixing)
                var num = Float32(0.0)
                var sa = Float32(0.0)
                var sb = Float32(0.0)
                for k in range(len(emb_db)):
                    var x = Float32(emb_db[k])
                    var y = Float32(emb_cam[k])
                    num = num + x * y
                    sa = sa + x * x
                    sb = sb + y * y
                var denom = Float32(1.0)
                if sa > Float32(0.0) and sb > Float32(0.0):
                    denom = (sa**Float32(0.5)) * (sb**Float32(0.5))
                score = num/denom
                is_match = score >= cfg.MATCH_THRESHOLD
                # positive example (same identity)
                if is_match:
                    tp += 1
                else:
                    fnn += 1
                val_samples += 1
                # negative check: compare db to a different identity first file
                if len(val_idents_py) > 1:
                    var other_choices_py = List[String]()
                    for p in val_idents_py:
                        if p != ident:
                            other_choices_py.append(p)
                    if len(other_choices_py) > 0:
                        # deterministic pick (first different identity)
                        var other = other_choices_py[0]
                        var other_files_py = List[String]()
                        var nomes_of = os.listdir(other)
                        for fname2 in nomes_of:
                            var p2 = other + "/" + fname2
                            if os.path.isfile(p2):
                                other_files_py.append(p2)
                        var emb_other = List[Float32]()
                        if len(other_files_py) > 0:
                            var other_db = other_files_py[0]
                            emb_other = embeddings_pkg.extract_embedding(other_db)
                        var num2 = Float32(0.0)
                        var sa2 = Float32(0.0)
                        var sb2 = Float32(0.0)
                        if len(emb_other) > 0:
                            for k2 in range(len(emb_db)):
                                var x = Float32(emb_db[k2])
                                var y = Float32(emb_other[k2])
                                num2 = num2 + x * y
                                sa2 = sa2 + x * x
                                sb2 = sb2 + y * y
                        var denom2 = Float32(1.0)
                        if len(emb_other) > 0:
                            if sa2 > Float32(0.0) and sb2 > Float32(0.0):
                                denom2 = (sa2**Float32(0.5)) * (sb2**Float32(0.5))
                            score2 = num2/denom2
                            is_match2 = score2 >= cfg.MATCH_THRESHOLD
                            if is_match2:
                                fp += 1
                            else:
                                tn += 1
                        # negative comparison accounted for above when `emb_other` exists

        var val_acc = Float32(0.0)
        var denom_cnt = tp + tn + fp + fnn
        if denom_cnt > 0:
            val_acc = Float32(tp + tn) / Float32(denom_cnt)
        print("  Validation samples: " + String(val_samples) + " TP=" + String(tp) + " FP=" + String(fp) + " TN=" + String(tn) + " FN=" + String(fnn))
        print("  Validation accuracy (proxy): " + String(val_acc))

    # Salva metadados e pesos dummy
    # write simple metadata without using Python-style dict literals
    io._ensure_parent_dir(meta_path)
    var mf = open(meta_path, 'w')
    mf.write("model: vivaz-poc\n")
    mf.write("input_short_side: " + String(cfg.INPUT_SHORT_SIDE) + "\n")
    mf.write("embedding_size: 512\n")
    mf.close()
    #io.save_weights(weights_path, b"VIVAZ-POC-WEIGHTS")

    print("Treino POC concluído (stub). Substituir por loop de otimização do BIONIX-ML.")
