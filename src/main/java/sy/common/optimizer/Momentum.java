package sy.common.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 *  momentum sgd
 * @author sy
 * @date 2022/3/14 20:29
 */
public class Momentum extends Optimizer {
    double lr = 0.01;
    double momentum = 0.9;
    List<INDArray> v = null;
    public Momentum(){}
    public Momentum(double lr, double momentum) {
        this.lr =lr;
        this.momentum = momentum;
    }

    @Override
    public void update(List<INDArray> params, List<INDArray> grads) {
        if(null == this.v) {
            this.v = new ArrayList<>();
            for(INDArray param : params) {
                this.v.add(Nd4j.zerosLike(param));
            }
        }

        for(int i=0; i<params.size(); i++) {
            INDArray vIndArray = this.v.get(i).mul(this.momentum).sub(grads.get(i).mul(this.lr));
            this.v.remove(i);
            this.v.add(i, vIndArray);
            params.get(i).addi(this.v.get(i));
        }

    }

}
