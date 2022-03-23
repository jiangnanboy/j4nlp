package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sy.common.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:16
 */
public class Softmax {
    List<INDArray> params = null;
    List<INDArray>  grads = null;
    INDArray out = null;

    public Softmax() {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
    }

    public INDArray forward(INDArray x) {
        this.out = Functions.softmax(x);
        return this.out;
    }

    public INDArray backward(INDArray dout) {
        INDArray dX = this.out.mul(dout);
        INDArray sumDX = Nd4j.sum(dX, 1);
        dX.subi(sumDX.mul(this.out));
        return dX;
    }

}

