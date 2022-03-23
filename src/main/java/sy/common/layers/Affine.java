package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 20:00
 */
public class Affine {
    List<INDArray> params = null;
    public List<INDArray>  grads = null;
    INDArray x = null;
    public Affine() {}
    public Affine(INDArray W, INDArray b) {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();

        this.params.add(W);
        this.params.add(b);

        this.grads.add(Nd4j.zerosLike(W));
        this.grads.add(Nd4j.zerosLike(b));
    }

    public INDArray forward(INDArray x) {
        INDArray W = this.params.get(0);
        INDArray b = this.params.get(1);
        INDArray out = Transforms.dot(x, W).add(b);
        this.x = x;
        return out;
    }

    public INDArray backward(INDArray dout) {
        INDArray W = this.params.get(0);
        INDArray b = this.params.get(1);
        INDArray dX = Transforms.dot(dout, W.transpose());
        INDArray dW = Transforms.dot(this.x.transpose(), dout);
        INDArray dB = Nd4j.sum(dout, 0);

        this.grads.remove(0);
        this.grads.remove(1);
        this.grads.add(dW);
        this.grads.add(dB);
        return dX;
    }
}


