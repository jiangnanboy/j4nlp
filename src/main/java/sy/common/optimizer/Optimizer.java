package sy.common.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * @author sy
 * @date 2022/3/23 21:56
 */
public abstract class Optimizer {
    public abstract void update(List<INDArray> params, List<INDArray> grads);
}
