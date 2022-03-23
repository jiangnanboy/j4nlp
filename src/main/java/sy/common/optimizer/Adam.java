package sy.common.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 20:44
 */
public class Adam extends Optimizer{
    double lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    int iter = 0;
    List<INDArray> m = null;
    List<INDArray> v = null;

    public Adam() {}
    public Adam(double lr, double beta1, double beta2) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public void update(List<INDArray> params, List<INDArray> grads) {
        if((null == this.m) && (null == this.v)) {
            this.m = new ArrayList<>();
            this.v = new ArrayList<>();
            for(INDArray param : params) {
                this.m.add(Nd4j.zerosLike(param));
                this.v.add(Nd4j.zerosLike(param));
            }
        }
        this.iter += 1;
        double lrT = this.lr * Math.sqrt(1.0 - Math.pow(this.beta2, this.iter)) / (1.0 - Math.pow(this.beta1, this.iter));

        for(int i=0; i<params.size(); i++) {
            this.m.get(i).addi((Nd4j.ones().sub(this.beta1)).mul(grads.get(i).sub(this.m.get(i))));
            this.v.get(i).addi((Nd4j.ones().sub(this.beta2)).mul(Transforms.pow(grads.get(i), 2).sub(this.v.get(i))));
            params.get(i).subi(this.m.get(i).mul(lrT).div(Transforms.sqrt(this.v.get(i)).add(1e-7)));
        }
    }
}
