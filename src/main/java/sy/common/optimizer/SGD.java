package sy.common.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * stochastic gradient descent
 * @author sy
 * @date 2022/3/14 19:21
 */
public class SGD extends Optimizer{

    double lr = 0.01;
    public SGD() {}
    public SGD(double lr) {
        this.lr = lr;
    }

    @Override
    public void update(List<INDArray> params, List<INDArray> grads) {
        for(int i=0; i<params.size(); i++) {
            params.get(i).subi(grads.get(i).mul(this.lr));
        }
    }

}
