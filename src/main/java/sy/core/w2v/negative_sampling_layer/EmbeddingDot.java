package sy.core.w2v.negative_sampling_layer;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sy.common.layers.Embedding;

import java.util.List;

/**
 * @author sy
 * @date 2022/3/23 23:02
 */
public class EmbeddingDot {

    Embedding embed = null;
    List<INDArray> params = null;
    List<INDArray> grads = null;
    Pair<INDArray, INDArray> cache = null;

    public EmbeddingDot(INDArray w) {
        this.embed = new Embedding(w);
        this.params = this.embed.params;
        this.grads = this.embed.grads;
    }

    public INDArray forward(INDArray h, INDArray idx) {
        INDArray targetW = this.embed.forward(idx);
        INDArray out = Nd4j.sum(targetW.mul(h), 1);
        this.cache = Pair.of(h, targetW);
        return out;
    }

    public INDArray backward(INDArray dout) {
        INDArray h = this.cache.getLeft();
        INDArray targetW = this.cache.getRight();
        dout = dout.reshape(dout.shape()[0], 1);

        INDArray dTargetW = dout.mul(h);
        this.embed.backward(dTargetW);
        INDArray dh = dout.mul(targetW);
        return dh;
    }

}

