package sy.common.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * nesterov's accelerated gradient (http://arxiv.org/abs/1212.0901)
 * @author sy
 * @date 2022/3/14 20:43
 */
public class Nesterov extends Optimizer{

    double lr = 0.01;
    double momentum = 0.9;
    List<INDArray> v = null;
    public Nesterov() {}
    public Nesterov(double lr, double momentum) {
        this.lr = lr;
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
            this.v.get(i).muli(this.momentum);
            this.v.get(i).subi(grads.get(i).mul(this.lr));
            params.get(i).addi(this.v.get(i).mul(this.momentum).mul(this.momentum));
            params.get(i).subi(Nd4j.ones().add(this.momentum).mul(this.lr).mul(grads.get(i)));
        }
    }

}

