package sy.common.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:44
 */
public class RMSprop extends Optimizer{
    double lr = 0.01;
    double decayRate = 0.99;
    List<INDArray> h = null;

    public RMSprop() {}
    public RMSprop(double lr, double decayRate) {
        this.lr = lr;
        this.decayRate = decayRate;
    }

    @Override
    public void update(List<INDArray> params, List<INDArray> grads) {
        if(null == this.h) {
            this.h = new ArrayList<>();
            for(INDArray param: params) {
                this.h.add(Nd4j.zerosLike(param));
            }
        }
        for(int i=0; i<params.size(); i++) {
            this.h.get(i).muli(this.decayRate);
            this.h.get(i).addi(Nd4j.ones().sub(this.decayRate).mul(grads.get(i)).mul(grads.get(i)));
            params.get(i).subi(grads.get(i).mul(this.lr).div(Transforms.sqrt(this.h.get(i)).add(1e-7)));
        }
    }
}

