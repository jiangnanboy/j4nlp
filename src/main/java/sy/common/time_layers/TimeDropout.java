package sy.common.time_layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:31
 */
public class TimeDropout {

    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    double dropoutRatio;
    INDArray mask = null;
    public boolean trainFlg = true;

    public TimeDropout() {
        new TimeDropout(0.5);
    }
    public TimeDropout(double dropoutRatio) {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        this.dropoutRatio = dropoutRatio;
    }

    public INDArray forward(INDArray xs) {
        if(this.trainFlg) {
            INDArray flg = Nd4j.rand(DataType.FLOAT, xs.shape()).gt(this.dropoutRatio);
            double scale = 1 / (1.0 - this.dropoutRatio);
            this.mask = flg.castTo(DataType.FLOAT).mul(scale);
            return xs.mul(this.mask);
        } else {
            return xs;
        }
    }

    public INDArray backward(INDArray dout) {
        return this.mask.mul(dout);
    }

}

