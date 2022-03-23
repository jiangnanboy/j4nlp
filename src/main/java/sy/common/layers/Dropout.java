package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:04
 */
public class Dropout {
    List<INDArray> params = null;
    List<INDArray> grads = null;
    double dropoutRatio = 0.5;
    INDArray mask = null;

    public Dropout() {}
    public Dropout(double dropoutRatio) {
        this.dropoutRatio = dropoutRatio;
    }

    public INDArray forward(INDArray x, boolean trianFlg) {
        if(trianFlg) {
            this.mask = Nd4j.rand(x.shape()).gt(this.dropoutRatio);
            return x.mul(this.mask);
        } else {
            return x.mul(Nd4j.ones().sub(this.dropoutRatio));
        }
    }

    public INDArray forward(INDArray x) {
        return this.forward(x, true);
    }

    public INDArray backward(int dout) {
        return this.mask.mul(dout);
    }

}

