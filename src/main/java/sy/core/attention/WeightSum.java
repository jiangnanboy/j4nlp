package sy.core.attention;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class WeightSum {
    List<INDArray> params;
    List<INDArray> grads;
    Pair<INDArray, INDArray> cache;

    public WeightSum() {
        this.params = new ArrayList();
        this.grads = new ArrayList();
        this.cache = null;
    }

    public INDArray forward(INDArray hs, INDArray a) {
        long N = hs.shape()[0];
        long T = hs.shape()[1];
        long H = hs.shape()[2];

        INDArray ar = a.reshape(N, T, 1).repeat(2, H);
        INDArray t = hs.mul(ar);
        INDArray c = Nd4j.sum(t, 1);
        this.cache = Pair.of(hs, ar);
        return c;
    }

    public Pair<INDArray, INDArray> backward(INDArray dc) {
        INDArray hs = this.cache.getLeft();
        INDArray ar = this.cache.getRight();
        long N = hs.shape()[0];
        long T = hs.shape()[1];
        long H = hs.shape()[2];
        INDArray dt = dc.reshape(N, 1, H).repeat(1, T);
        INDArray dar = dt.mul(hs);
        INDArray dhs = dt.mul(ar);
        INDArray da = Nd4j.sum(dar, 2);
        return Pair.of(dhs, da);
    }

}

