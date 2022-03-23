package sy.core.w2v;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * @author sy
 * @date 2022/3/22 23:52
 */
public abstract class AbsW2V {
    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    public INDArray wordVecs = null;

    public abstract INDArray forward(INDArray context, INDArray target);

    public abstract INDArray backward();

    public abstract INDArray backward(int dout);
}
