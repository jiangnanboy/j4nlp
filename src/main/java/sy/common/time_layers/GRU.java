package sy.common.time_layers;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import sy.common.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:31
 */
public class GRU {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    List<INDArray> cache = null;

    INDArray dWx = null;
    INDArray dWh = null;
    INDArray db = null;

    public GRU() {}
    public GRU(INDArray Wx, INDArray Wh, INDArray b) {
        this.params = new ArrayList<>();
        this.params.add(Wx);
        this.params.add(Wh);
        this.params.add(b);
        this.grads = new ArrayList<>();
        this.grads.add(Nd4j.zerosLike(Wx));
        this.grads.add(Nd4j.zerosLike(Wh));
        this.grads.add(Nd4j.zerosLike(b));
    }

    public INDArray forward(INDArray x, INDArray hPrev) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);

        long H = Wh.shape()[0];

        INDArray Wxz = Wx.get(NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        INDArray Wxr = Wx.get(NDArrayIndex.all(), NDArrayIndex.interval(H, 2 * H));
        INDArray Wxh = Wx.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * H, Wx.shape()[1]));

        INDArray Whz = Wh.get(NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        INDArray Whr = Wh.get(NDArrayIndex.all(), NDArrayIndex.interval(H, 2 * H));
        INDArray Whh = Wh.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * H, Wh.shape()[1]));

        INDArray bz = b.get(NDArrayIndex.interval(0, H));
        INDArray br = b.get(NDArrayIndex.interval(H, 2 * H));
        INDArray bh = b.get(NDArrayIndex.interval(2 * H, b.shape()[0]));

        INDArray z = Functions.sigmoid(Transforms.dot(x, Wxz).add(Transforms.dot(hPrev, Whz)).add(bz));
        INDArray r = Functions.sigmoid(Transforms.dot(x, Wxr).add(Transforms.dot(hPrev, Whr)).add(br));
        INDArray hHat = Transforms.tanh(Transforms.dot(x, Wxh).add(Transforms.dot(r.mul(hPrev), Whh)).add(bh));
        INDArray hNext = Nd4j.ones().sub(z).mul(hPrev).add(z.mul(hHat));

        this.cache = new ArrayList<>();
        this.cache.add(x);
        this.cache.add(hPrev);
        this.cache.add(z);
        this.cache.add(r);
        this.cache.add(hHat);

        return hNext;
    }

    public Pair<INDArray, INDArray> backward(INDArray dHNext) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);

        long H = Wh.shape()[0];

        INDArray Wxz = Wx.get(NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        INDArray Wxr = Wx.get(NDArrayIndex.all(), NDArrayIndex.interval(H, 2 * H));
        INDArray Wxh = Wx.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * H, Wx.shape()[1]));

        INDArray Whz = Wh.get(NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        INDArray Whr = Wh.get(NDArrayIndex.all(), NDArrayIndex.interval(H, 2 * H));
        INDArray Whh = Wh.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * H, Wh.shape()[1]));

        INDArray x = this.cache.get(0);
        INDArray hPrev = this.cache.get(1);
        INDArray z = this.cache.get(2);
        INDArray r = this.cache.get(3);
        INDArray hHat = this.cache.get(4);

        INDArray dhHat = dHNext.mul(z);
        INDArray dhPrev = dHNext.mul(Nd4j.ones().sub(z));

        // tanh
        INDArray dt = dhHat.mul(Nd4j.ones().sub(Transforms.pow(hHat, 2)));
        INDArray dbh = Nd4j.sum(dt, 0);
        INDArray dWhh = Transforms.dot((r.mul(hPrev).transpose()), dt);
        INDArray dhr = Transforms.dot(dt, Whh.transpose());
        INDArray dWxh = Transforms.dot(x.transpose(), dt);
        INDArray dx = Transforms.dot(dt, Wxh.transpose());
        dhPrev.add(r.mul(dhr));

        // update gate(z)
        INDArray dz = dHNext.mul(hHat).sub(dHNext.mul(hPrev));
        dt = dz.mul(z).mul(Nd4j.ones().sub(z));
        INDArray dbz = Nd4j.sum(dt, 0);
        INDArray dWhz = Transforms.dot(hPrev.transpose(), dt);
        dhPrev.add(Transforms.dot(dt, Whz.transpose()));
        INDArray dWxz = Transforms.dot(x.transpose(), dt);
        dx.add(Transforms.dot(dt, Wxz.transpose()));

        // reset gate(r)
        INDArray dr = dhr.mul(hPrev);
        dt = dr.mul(r).mul(Nd4j.ones().sub(r));
        INDArray dbr = Nd4j.sum(dt, 0);
        INDArray dWhr = Transforms.dot(hPrev.transpose(), dt);
        dhPrev.add(Transforms.dot(dt, Whr.transpose()));
        INDArray dWxr = Transforms.dot(x.transpose(), dt);
        dx.add(Transforms.dot(dt, Wxr.transpose()));

        this.dWx = Nd4j.hstack(dWxz, dWxr, dWxh);
        this.dWh = Nd4j.hstack(dWhz, dWhr, dWhh);
        this.db = Nd4j.hstack(dbz, dbr, dbh);

        this.grads.remove(0);
        this.grads.remove(1);
        this.grads.remove(2);
        this.grads.add(dWx);
        this.grads.add(dWh);
        this.grads.add(db);

        Pair<INDArray, INDArray> pair = Pair.of(dx, dhPrev);

        return pair;

    }

}

