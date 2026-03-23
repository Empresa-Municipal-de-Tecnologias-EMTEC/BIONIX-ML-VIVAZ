# Inspeciona o dataset VIVAZ: conta identities, imagens BMP e coleta dimensões amostrais

import os
import bionix_ml.dados as dados_pkg


def _is_bmp(name: String) -> Bool:
    lower = name.lower()
    return lower.endswith('.bmp')


def inspecionar(dataroot: String = "src/DATASET", sample_idents: Int = 5):
    print("Inspecionando dataset em:", dataroot)
    splits = ["train", "val", "original"]
    overall = {}
    for s in splits:
        path = os.path.join(dataroot, s)
        if not os.path.exists(path):
            continue
        idents = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        idents = sorted(idents)
        counts = List[Int]()
        sample_dims = List[List[String]]()
        for i, ident in enumerate(idents):
            ipath = os.path.join(path, ident)
            files = [f for f in os.listdir(ipath) if os.path.isfile(os.path.join(ipath, f)) and _is_bmp(f)]
            counts.append(len(files))
            if i < sample_idents and len(files) > 0:
                f0 = os.path.join(ipath, files[0])
                try:
                    info = dados_pkg.carregar_bmp_rgb(f0)
                    sample_dims.append([ident, files[0], str(info.width), str(info.height)])
                except Exception as e:
                    sample_dims.append([ident, files[0], str(-1), str(-1)])

        total_idents = len(idents)
        total_imgs = 0
        if counts:
            for c in counts:
                total_imgs = total_imgs + c
            avg_per_ident = (total_imgs / total_idents) if total_idents > 0 else 0
            min_imgs = counts[0]
            max_imgs = counts[0]
            for c in counts:
                if c < min_imgs:
                    min_imgs = c
                if c > max_imgs:
                    max_imgs = c
        else:
            avg_per_ident = 0
            min_imgs = 0
            max_imgs = 0

        overall[s] = {
            'id_count': total_idents,
            'img_count': total_imgs,
            'avg_per_ident': avg_per_ident,
            'min_imgs': min_imgs,
            'max_imgs': max_imgs,
            'samples': sample_dims,
        }

    # Print summary
    for s, v in overall.items():
        print("--- Split:", s)
        print("  Identities: " + str(v['id_count']) + "  Images: " + str(v['img_count']) + "  Avg/imgs/ident: " + str(round(v['avg_per_ident'], 2)))
        print("  Images per identity: min=" + str(v['min_imgs']) + " max=" + str(v['max_imgs']))
        if v['samples']:
            print("  Exemplos amostrados (identity, file, width, height):")
            for tup in v['samples']:
                print("   -", tup[0], tup[1], tup[2] + "x" + tup[3])


def main():
    inspecionar()
