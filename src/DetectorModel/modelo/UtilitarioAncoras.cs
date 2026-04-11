using System;
using System.Collections.Generic;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace DetectorModel.modelo
{
    public struct BoxF { public double X, Y, W, H; public BoxF(double x,double y,double w,double h){X=x;Y=y;W=w;H=h;} }

    public static class UtilitarioAncoras
    {
        // Generate a grid of anchors centered on feature map of size (fh,fw)
        public static List<BoxF> GenerateAnchors(int fh, int fw, int baseSize, double[] ratios, double[] scales, int stride)
        {
            var anchors = new List<BoxF>();
            for (int y = 0; y < fh; y++)
            for (int x = 0; x < fw; x++)
            {
                double cx = x * stride + stride/2.0;
                double cy = y * stride + stride/2.0;
                foreach (var r in ratios)
                foreach (var s in scales)
                {
                    double area = baseSize * baseSize * s * s;
                    double w = Math.Sqrt(area / r);
                    double h = w * r;
                    anchors.Add(new BoxF(cx - w/2.0, cy - h/2.0, w, h));
                }
            }
            return anchors;
        }

        public static double IoU(BoxF a, BoxF b)
        {
            double ax1 = a.X, ay1 = a.Y, ax2 = a.X + a.W, ay2 = a.Y + a.H;
            double bx1 = b.X, by1 = b.Y, bx2 = b.X + b.W, by2 = b.Y + b.H;
            double ix1 = Math.Max(ax1, bx1);
            double iy1 = Math.Max(ay1, by1);
            double ix2 = Math.Min(ax2, bx2);
            double iy2 = Math.Min(ay2, by2);
            double iw = Math.Max(0, ix2 - ix1);
            double ih = Math.Max(0, iy2 - iy1);
            double inter = iw * ih;
            double areaA = a.W * a.H;
            double areaB = b.W * b.H;
            return inter / (areaA + areaB - inter + 1e-8);
        }

        // Encode ground-truth box relative to anchor: returns [tx,ty,tw,th]
        public static double[] Encode(BoxF anchor, BoxF gt)
        {
            double ax = anchor.X + anchor.W/2.0;
            double ay = anchor.Y + anchor.H/2.0;
            double aw = anchor.W;
            double ah = anchor.H;
            double gx = gt.X + gt.W/2.0;
            double gy = gt.Y + gt.H/2.0;
            double gw = gt.W;
            double gh = gt.H;
            double tx = (gx - ax) / aw;
            double ty = (gy - ay) / ah;
            double tw = Math.Log(gw / aw + 1e-8);
            double th = Math.Log(gh / ah + 1e-8);
            return new double[]{tx,ty,tw,th};
        }

        // Encode landmarks (10 values) relative to anchor center/size: returns [lx1,ly1,...]
        public static double[] EncodeLandmarks(BoxF anchor, double[] landmarks)
        {
            // landmarks: lx1,ly1,...,lx5,ly5 (absolute image coords)
            var outL = new double[10];
            double ax = anchor.X + anchor.W/2.0;
            double ay = anchor.Y + anchor.H/2.0;
            double aw = anchor.W;
            double ah = anchor.H;
            for (int i = 0; i < 5; i++)
            {
                double lx = landmarks[i*2];
                double ly = landmarks[i*2 + 1];
                outL[i*2] = (lx - ax) / aw;
                outL[i*2 + 1] = (ly - ay) / ah;
            }
            return outL;
        }

        // Decode predicted deltas to box coordinates
        public static BoxF Decode(BoxF anchor, double[] delta)
        {
            double ax = anchor.X + anchor.W/2.0;
            double ay = anchor.Y + anchor.H/2.0;
            double aw = anchor.W;
            double ah = anchor.H;
            double gx = delta[0] * aw + ax;
            double gy = delta[1] * ah + ay;
            double gw = Math.Exp(delta[2]) * aw;
            double gh = Math.Exp(delta[3]) * ah;
            return new BoxF(gx - gw/2.0, gy - gh/2.0, gw, gh);
        }

        // Non-maximum suppression on boxes with scores
        public static List<int> NMS(List<BoxF> boxes, List<double> scores, double iouThresh)
        {
            var idxs = new List<int>();
            for (int i = 0; i < scores.Count; i++) idxs.Add(i);
            idxs.Sort((a,b)=>scores[b].CompareTo(scores[a]));
            var keep = new List<int>();
            while (idxs.Count>0)
            {
                int i = idxs[0];
                keep.Add(i);
                var rem = new List<int>();
                for (int k=1;k<idxs.Count;k++){
                    int j = idxs[k];
                    if (IoU(boxes[i], boxes[j]) <= iouThresh) rem.Add(j);
                }
                idxs = rem;
            }
            return keep;
        }

        // Match anchors to ground-truth boxes using IoU thresholds.
        // labels: 1 = positive, 0 = negative, -1 = ignore
        // matchedGt: index of matched GT for positives, -1 otherwise
        // bboxTargets: encoded [tx,ty,tw,th] for each anchor (zeros for non-positives)
        public static void MatchAnchors(List<BoxF> anchors, List<BoxF> gts, double posIou, double negIou,
                                        out int[] labels, out int[] matchedGt, out double[][] bboxTargets)
        {
            int A = anchors.Count;
            labels = new int[A];
            matchedGt = new int[A];
            bboxTargets = new double[A][];
            for (int i = 0; i < A; i++) { labels[i] = -1; matchedGt[i] = -1; bboxTargets[i] = new double[4]; }

            if (gts == null || gts.Count == 0)
            {
                // all negatives
                for (int i = 0; i < A; i++) labels[i] = 0;
                return;
            }

            for (int i = 0; i < A; i++)
            {
                double bestIoU = -1.0;
                int bestIdx = -1;
                for (int j = 0; j < gts.Count; j++)
                {
                    double iou = IoU(anchors[i], gts[j]);
                    if (iou > bestIoU) { bestIoU = iou; bestIdx = j; }
                }
                if (bestIdx == -1) { labels[i] = 0; continue; }
                if (bestIoU >= posIou)
                {
                    labels[i] = 1;
                    matchedGt[i] = bestIdx;
                    bboxTargets[i] = Encode(anchors[i], gts[bestIdx]);
                }
                else if (bestIoU < negIou)
                {
                    labels[i] = 0;
                }
                else
                {
                    labels[i] = -1; // ignore
                }
            }
            // ensure each GT has at least one positive (best anchor)
            for (int j = 0; j < gts.Count; j++)
            {
                double bestIoU = -1.0; int bestA = -1;
                for (int i = 0; i < A; i++)
                {
                    double iou = IoU(anchors[i], gts[j]);
                    if (iou > bestIoU) { bestIoU = iou; bestA = i; }
                }
                if (bestA >= 0)
                {
                    labels[bestA] = 1;
                    matchedGt[bestA] = j;
                    bboxTargets[bestA] = Encode(anchors[bestA], gts[j]);
                }
            }
        }

        // Extended matching that also returns landmark targets (10 floats) per anchor (zero for non-positives)
        public static void MatchAnchorsWithLandmarks(List<BoxF> anchors, List<BoxF> gts, List<double[]> landmarks,
                                                     double posIou, double negIou,
                                                     out int[] labels, out int[] matchedGt, out double[][] bboxTargets, out double[][] landmarkTargets)
        {
            MatchAnchors(anchors, gts, posIou, negIou, out labels, out matchedGt, out bboxTargets);
            int A = anchors.Count;
            landmarkTargets = new double[A][];
            for (int i = 0; i < A; i++) landmarkTargets[i] = new double[10];
            for (int i = 0; i < A; i++)
            {
                if (labels[i] == 1 && matchedGt[i] >= 0 && matchedGt[i] < landmarks.Count)
                {
                    landmarkTargets[i] = EncodeLandmarks(anchors[i], landmarks[matchedGt[i]]);
                }
            }
        }
    }
}
