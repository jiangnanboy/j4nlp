package sy.core.attention;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sy.common.layers.Softmax;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class AttentionWeight {
    List<INDArray> params;
    List<INDArray> grads;
    Softmax softmax;
    Pair<INDArray, INDArray> cache;

    public AttentionWeight() {
        this.params = new ArrayList();
        this.grads = new ArrayList();
        this.softmax = new Softmax();
        this.cache = null;
    }

    public INDArray forward(INDArray hs, INDArray h) {
        long N = hs.shape()[0];
        long T = hs.shape()[1];
        long H = hs.shape()[2];
        INDArray hr = h.reshape(N, 1, H).repeat(1, T);
        INDArray t = hs.mul(hr);
        INDArray s = Nd4j.sum(t, 2);
        INDArray a = this.softmax.forward(s);

        this.cache = Pair.of(hs, hr);
        return a;
    }

    public Pair<INDArray, INDArray> backward(INDArray da) {
        INDArray hs = this.cache.getLeft();
        INDArray hr = this.cache.getRight();
        long N = hs.shape()[0];
        long T = hs.shape()[1];
        long H = hs.shape()[2];
        INDArray ds = this.softmax.backward(da);
        INDArray dt = ds.reshape(N, T, 1).repeat(2, H);
        INDArray dhs = dt.mul(hr);
        INDArray dhr = dt.mul(ds);
        INDArray dh = Nd4j.sum(dhr, 1);

        return Pair.of(dhs, dh);
    }
}

