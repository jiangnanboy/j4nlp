package sy.common.time_layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.layers.Affine;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:32
 */
public class SimpleTimeAffine {

    INDArray W = null;
    INDArray b = null;
    INDArray dW = null;
    INDArray db = null;
    List<Affine> layers = null;

    public SimpleTimeAffine() {}
    public SimpleTimeAffine(INDArray W, INDArray b) {
        this.W = W;
        this.b = b;
    }

    public INDArray forward(INDArray xs) {
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        long D = xs.shape()[2];
        D = this.W.shape()[0];
        long M = this.W.shape()[1];

        this.layers = new ArrayList<>();
        INDArray out = Nd4j.rand(DataType.FLOAT, N, T, M);
        for(int t=0; t < T; t++) {
            Affine layer = new Affine(this.W, this.b);
            out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, layer.forward(xs.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all())));
            this.layers.add(layer);
        }

        return out;
    }

    public INDArray backward(INDArray dout) {
        long N = dout.shape()[0];
        long T = dout.shape()[1];
        long M = dout.shape()[2];
        long D = this.W.shape()[0];
        M = this.W.shape()[1];

        INDArray dxs = Nd4j.rand(DataType.FLOAT, N, T, D);
        this.dW = Nd4j.zeros();
        this.db = Nd4j.zeros();
        for(int t=0; t < T; t++) {
            Affine layer = this.layers.get(t);
            dxs.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, layer.backward(dout.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all())));
            this.dW.addi(layer.grads.get(0));
            this.db.addi(layer.grads.get(1));
        }
        return dxs;
    }

}

