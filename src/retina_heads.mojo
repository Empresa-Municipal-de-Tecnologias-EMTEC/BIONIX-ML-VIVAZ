import bionix_ml.nucleo.Tensor as tensor_defs
import math

type RetinaHeads = struct {
    peso_cls: tensor_defs.Tensor
    bias_cls: tensor_defs.Tensor
    peso_reg: tensor_defs.Tensor
    bias_reg: tensor_defs.Tensor
}

fn criar_head_vazio(feat_dim: Int, num_anchors: Int, tipo: String = "cpu") -> RetinaHeads:
    var h = RetinaHeads()
    # classification: one logit per anchor
    var shape_c = List[Int]()
    shape_c.append(1); shape_c.append(1) # placeholder
    h.peso_cls = tensor_defs.Tensor(shape_c^, tipo)
    h.bias_cls = tensor_defs.Tensor(shape_c^, tipo)
    # regression: 4 values per anchor
    var shape_r = List[Int]()
    shape_r.append(feat_dim); shape_r.append(4)
    h.peso_reg = tensor_defs.Tensor(shape_r^, tipo)
    var shape_rb = List[Int]()
    shape_rb.append(1); shape_rb.append(4)
    h.bias_reg = tensor_defs.Tensor(shape_rb^, tipo)
    return h
