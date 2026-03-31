import retina.retina_anchor_generator as ag
import math

fn main():
    # Lightweight harness to reproduce and inspect anchors from gerar_anchors
    var sizes = [320, 640]
    for s in sizes:
        try:
            print("[HARN] gerar_anchors for input_size=", s)
            var anchors = ag.gerar_anchors(s)
            var total = len(anchors)
            var bad = 0
            var first_bad_idx = -1
            for i in range(total):
                var a = anchors[i].copy()
                # each anchor is [cx, cy, w, h]
                var ok = True
                for v in a:
                    # NaN check
                    if v != v:
                        ok = False
                        break
                    # giant/absurd values
                    if v < -1e9 or v > 1e9:
                        ok = False
                        break
                if not ok:
                    bad = bad + 1
                    if first_bad_idx == -1:
                        first_bad_idx = i
            print("[HARN] input_size=", s, "anchors=", total, "bad=", bad, "first_bad_idx=", first_bad_idx)
            if total > 0:
                var sa = anchors[0].copy()
                print("[HARN] sample anchor[0]:", sa[0], sa[1], sa[2], sa[3])
        except _:
            print("[HARN] failed to run gerar_anchors for size", s)
