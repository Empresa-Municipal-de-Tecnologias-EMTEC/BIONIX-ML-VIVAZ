# Amostrador variado: estratégia simples para não carregar todas as imagens na memória
# Funções compatíveis com o estilo adotado pelo restante do código (sem classes)

import os
import random

def _list_identities(dataset_root: String) -> List[String]:
    var train = dataset_root + "/train"
    if not os.path.exists(train):
        return List[String]()
    var nomes = os.listdir(train)
    var idents = List[String]()
    for n in nomes:
        var p = train + "/" + n
        if os.path.isdir(p):
            idents.append(p)
    return idents

def sample_epoch(dataset_root: String, identities_per_epoch: Int = 50, images_per_identity: Int = 5) -> List[Tuple[String, List[String]]]:
    var identities = _list_identities(dataset_root)
    if len(identities) == 0:
        return List[Tuple[String, List[String]]]()
    var count = identities_per_epoch
    if identities_per_epoch > len(identities):
        count = len(identities)
    var chosen = List[String]()
    for i in range(count):
        chosen.append(identities[i])
    var batch = List[Tuple[String, List[String]]]()
    for ident in chosen:
        var files = List[String]()
        var nomes_f = os.listdir(ident)
        for fname in nomes_f:
            var p = ident + "/" + fname
            if os.path.isfile(p):
                files.append(p)
        if len(files) == 0:
            continue
        var pick_count = images_per_identity
        if images_per_identity > len(files):
            pick_count = len(files)
        var pick = List[String]()
        for k in range(pick_count):
            pick.append(files[k])
        # identity name (basename) and picked file paths
        var ident_name = String(os.path.basename(ident))
        batch.append((ident_name, pick))
    return batch
