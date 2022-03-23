package sy.common.time_layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.layers.Embedding;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:31
 */
public class TimeEmbedding {
    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    List<Embedding> layers = null;
    INDArray W = null;

    public TimeEmbedding() {}

    public TimeEmbedding(INDArray W) {
        this.params = new ArrayList<>();
        this.params.add(W);
        this.grads = new ArrayList<>();
        this.grads.add(Nd4j.zerosLike(W));
        this.W = W;
    }

    public INDArray forward(INDArray xs) {
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        long V = this.W.shape()[0];
        long D = this.W.shape()[1];

        INDArray out = Nd4j.rand(DataType.FLOAT, N, T, D);
        this.layers = new ArrayList<>();

        for(int t=0; t < T; t++) {
            Embedding layer = new Embedding(this.W);
            out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, layer.forward(xs.get(NDArrayIndex.all(), NDArrayIndex.point(t))));
            this.layers.add(layer);
        }

        return out;
    }

    public INDArray backward(INDArray dout) {
        long N = dout.shape()[0];
        long T = dout.shape()[1];
        long D = dout.shape()[2];

        INDArray grad = Nd4j.zeros();
        for(int t=0; t < T; t++) {
            Embedding layer = this.layers.get(t);
            layer.backward(dout.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()));
            grad.addi(layer.grads.get(0));
        }

        this.grads.remove(0);
        this.grads.add(grad);

        return null;
    }

}

