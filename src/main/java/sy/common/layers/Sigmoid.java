package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 20:41
 */
public class Sigmoid {
    List<INDArray> params = null;
    List<INDArray> grads = null;
    INDArray out = null;

    public Sigmoid(){
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
    }

    public INDArray forward(INDArray x) {
        INDArray indArray = (Transforms.exp(Transforms.neg(x)).add(1));
        this.out = Nd4j.ones().div(indArray);
        return this.out;
    }

    public INDArray backward(INDArray dout) {
        INDArray dX = dout.mul(Nd4j.ones().sub(this.out)).mul(this.out);
        return dX;
    }

}

