package sy.common.time_layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:31
 */
public class TimeAffine {

    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    INDArray x = null;

    public TimeAffine() {}
    public TimeAffine(INDArray W, INDArray b) {
        this.params = new ArrayList<>();
        this.params.add(W);
        this.params.add(b);
        this.grads = new ArrayList<>();
        this.grads.add(Nd4j.zerosLike(W));
        this.grads.add(Nd4j.zerosLike(b));
    }

    public INDArray forward(INDArray x) {
        long N = x.shape()[0];
        long T = x.shape()[1];
        long D = x.shape()[2];
        INDArray W = this.params.get(0);
        INDArray b = this.params.get(1);

        INDArray rx = x.reshape(N * T, -1);
        INDArray out = Transforms.dot(rx, W).add(b);
        this.x = x;

        return out.reshape(N, T, -1);
    }

    public INDArray backward(INDArray dout) {
        INDArray x = this.x;
        long N = x.shape()[0];
        long T = x.shape()[1];
        long D = x.shape()[2];
        INDArray W = this.params.get(0);
        INDArray b = this.params.get(1);

        dout = dout.reshape(N * T, -1);
        INDArray rx = x.reshape(N * T, -1);

        INDArray db = Nd4j.sum(dout, 0);
        INDArray dW = Transforms.dot(rx.transpose(), dout);
        INDArray dx = Transforms.dot(dout, W.transpose());
        dx = dx.reshape(x.shape());

        this.grads.remove(0);
        this.grads.remove(1);
        this.grads.add(dW);
        this.grads.add(db);

        return dx;
    }

}

