package sy.core.attention;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class Attention {
    List<INDArray> params;
    List<INDArray> grads;
    AttentionWeight attentionWeightLayer;
    WeightSum weightSumLayer;
    INDArray attentionWeight;

    public Attention() {
        this.params = new ArrayList();
        this.grads = new ArrayList();
        this.attentionWeightLayer = new AttentionWeight();
        this.weightSumLayer = new WeightSum();
        this.attentionWeight = null;
    }

    public INDArray forward(INDArray hs, INDArray h) {
        INDArray a = this.attentionWeightLayer.forward(hs, h);
        INDArray out = this.weightSumLayer.forward(hs, a);
        this.attentionWeight = a;
        return out;
    }

    public Pair<INDArray, INDArray> backward(INDArray dout) {
        Pair<INDArray, INDArray> pairWeightSumLayer = this.weightSumLayer.backward(dout);
        INDArray dhs0 = pairWeightSumLayer.getLeft();
        INDArray da = pairWeightSumLayer.getRight();

        Pair<INDArray, INDArray> pairAttentionWeightLayer = this.attentionWeightLayer.backward(da);
        INDArray dhs1 = pairAttentionWeightLayer.getLeft();
        INDArray dh = pairAttentionWeightLayer.getRight();
        INDArray dhs = dhs0.add(dhs1);
        return Pair.of(dhs, dh);
    }

}


