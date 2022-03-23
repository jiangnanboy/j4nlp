package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:17
 */
public class MatMul {

    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    INDArray x = null;
    public MatMul() {}
    public MatMul(INDArray W) {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        this.params.add(W);
        this.grads.add(Nd4j.zerosLike(W));
    }

    public INDArray forward(INDArray x) {
        INDArray W = this.params.get(0);
        INDArray out = Transforms.dot(x, W);
        this.x = x;
        return out;
    }

    public INDArray backward(INDArray dout) {
        INDArray W = this.params.get(0);
        INDArray dX = Transforms.dot(dout, W.transpose());
        INDArray dW = Transforms.dot(this.x.transpose(), dout);
        this.grads.remove(0);
        this.grads.add(dW);
        return dX;
    }
}

