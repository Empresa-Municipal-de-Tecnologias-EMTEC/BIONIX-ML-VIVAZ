import bionix_ml.graficos.bmp as bmp_pkg
import bionix_ml.graficos as graficos_pkg

fn main() -> None:
    print("Starting crop test")
    # First, simulate a large image resize as in the trainer to reproduce peak allocations
    var largura = 544
    var altura = 622
    var channels = 3
    var flat = List[Float32]()
    for y in range(altura):
        for x in range(largura):
            flat.append(Float32((x + y) % 256) / 255.0)
            flat.append(Float32((x * 2 + y) % 256) / 255.0)
            flat.append(Float32((x + y * 2) % 256) / 255.0)

    print("large flat_len", len(flat))
    # Perform a full resize to a working image matrix (this may allocate many nested lists)
    var img_matrix = bmp_pkg.redimensionar_matriz_rgb_nearest_from_flat(flat, largura, altura, channels, 320, 320)^
    print("img_matrix dims -> h", len(img_matrix), "w", len(img_matrix[0]) if len(img_matrix) > 0 else 0)

    # Now call crop_and_resize_from_flat as trainer does
    var out = bmp_pkg.crop_and_resize_from_flat(flat, largura, altura, channels, 32, 32, 287, 287, 32, 32)^
    print("crop out len", len(out))

    return
