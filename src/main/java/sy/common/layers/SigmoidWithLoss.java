package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import sy.common.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:50
 */
public class SigmoidWithLoss {
    List<INDArray> params = null;
    List<INDArray> grads = null;
    INDArray loss = null;
    INDArray y = null; // the output of the sigmoid
    INDArray t = null; // the target

    public SigmoidWithLoss() {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
    }

    public INDArray forward(INDArray x, INDArray t) {
        this.t = t;
        this.y = Nd4j.ones().div(Transforms.exp(x.neg()).add(1));
        this.loss = Functions.crossEntropyError(Nd4j.concat(1, this.y.neg().add(1), this.y), this.t);
        return this.loss;
    }

    public INDArray backward(int dout) {
        long batchSize = this.t.shape()[0];
        INDArray dX = this.y.sub(this.t).mul(dout).div(batchSize);
        return dX;
    }

    public INDArray backward() {
        return this.backward(1);
    }

}

