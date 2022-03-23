package sy.common.time_layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.apache.commons.lang3.tuple.Pair;
import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:30
 */
public class RNN {
    List<INDArray> params = null;
    List<INDArray> grads = null;
    List<INDArray> cache = null;

    public RNN() {}
    public RNN(INDArray Wx, INDArray Wh, INDArray b) {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        this.params.add(Wx);
        this.params.add(Wh);
        this.params.add(b);
        this.grads.add(Nd4j.zerosLike(Wx));
        this.grads.add(Nd4j.zerosLike(Wh));
        this.grads.add(Nd4j.zerosLike(b));
    }

    public INDArray forward(INDArray x, INDArray hPrev) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);
        INDArray t = Transforms.dot(hPrev, Wh).add(Transforms.dot(x, Wx)).add(b);
        INDArray hNext = Transforms.tanh(t);
        this.cache = new ArrayList<>();
        this.cache.add(x);
        this.cache.add(hPrev);
        this.cache.add(hNext);
        return hNext;
    }

    public Pair<INDArray, INDArray> backward(INDArray dHNext) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);
        INDArray x = this.cache.get(0);
        INDArray hPrev = this.cache.get(1);
        INDArray hNext = this.cache.get(2);

        INDArray dT = dHNext.mul(Nd4j.ones().sub(Transforms.pow(hNext, 2)));
        INDArray dB = Nd4j.sum(dT, 0);
        INDArray dWH = Transforms.dot(hPrev.transpose(), dT);
        INDArray dHPrev = Transforms.dot(dT, Wh.transpose());
        INDArray dWX = Transforms.dot(x.transpose(), dT);
        INDArray dX = Transforms.dot(dT, Wx.transpose());

        this.grads.remove(0);
        this.grads.remove(1);
        this.grads.remove(2);
        this.grads.add(dWX);
        this.grads.add(dWH);
        this.grads.add(dB);

        Pair<INDArray, INDArray> pair = Pair.of(dX, dHPrev);

        return pair;
    }

}

