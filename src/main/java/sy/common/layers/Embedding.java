package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:13
 */
public class Embedding {
    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    INDArray idx;
    public Embedding() {}
    public Embedding(INDArray W) {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        this.params.add(W);
        this.grads.add(Nd4j.zerosLike(W));
    }

    public INDArray forward(INDArray idx) {
        INDArray W = this.params.get(0);
        this.idx = idx;
        INDArray out = W.get(idx);
        return out;
    }

    public INDArray backward(INDArray dout) {
        INDArray dW = this.grads.get(0);
        dW = Nd4j.zeros();
        dW.put(this.idx, dW.get(this.idx).add(dout));
        return null;
    }

}

