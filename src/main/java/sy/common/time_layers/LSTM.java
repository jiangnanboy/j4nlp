package sy.common.time_layers;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import sy.common.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:30
 */
public class LSTM {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    List<INDArray> cache = null;

    public LSTM() {}
    public LSTM(INDArray Wx, INDArray Wh, INDArray b) {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        this.params.add(Wx);
        this.params.add(Wh);
        this.params.add(b);
        this.grads.add(Nd4j.zerosLike(Wx));
        this.grads.add(Nd4j.zerosLike(Wh));
        this.grads.add(Nd4j.zerosLike(b));
    }

    public Pair<INDArray, INDArray> forward(INDArray x, INDArray hPrev, INDArray cPrev) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);
        long N = hPrev.shape()[0];
        long H = hPrev.shape()[1];

        INDArray A = Transforms.dot(x, Wx).add(Transforms.dot(hPrev, Wh)).add(b);

        INDArray f = A.get(NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        INDArray g = A.get(NDArrayIndex.all(), NDArrayIndex.interval(H, 2 * H));
        INDArray i = A.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * H, 3 * H));
        INDArray o = A.get(NDArrayIndex.all(), NDArrayIndex.interval(3 * H, A.shape()[1]));

        f = Functions.sigmoid(f);
        g = Transforms.tanh(g);
        i = Functions.sigmoid(i);
        o = Functions.sigmoid(o);

        INDArray cNext = f.mul(cPrev).add(g.mul(i));
        INDArray hNext = o.mul(Transforms.tanh(cNext));

        this.cache = new ArrayList<>();
        this.cache.add(x);
        this.cache.add(hPrev);
        this.cache.add(cPrev);
        this.cache.add(i);
        this.cache.add(f);
        this.cache.add(g);
        this.cache.add(o);
        this.cache.add(cNext);

        Pair<INDArray, INDArray> pair = Pair.of(hNext, cNext);

        return pair;
    }

    public Triple<INDArray, INDArray, INDArray> backward(INDArray dHNext, INDArray dCNext) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);

        INDArray x = this.cache.get(0);
        INDArray hPrev = this.cache.get(1);
        INDArray cPrev = this.cache.get(2);
        INDArray i = this.cache.get(3);
        INDArray f = this.cache.get(4);
        INDArray g = this.cache.get(5);
        INDArray o = this.cache.get(6);
        INDArray cNext = this.cache.get(7);

        INDArray tanHCNext = Transforms.tanh(cNext);

        INDArray dS = dCNext.add((dHNext.mul(o)).mul(Nd4j.ones().sub(Transforms.pow(tanHCNext, 2))));

        INDArray dCPrev = dS.mul(f);

        INDArray dI = dS.mul(g);
        INDArray dF = dS.mul(cPrev);
        INDArray dO = dHNext.mul(tanHCNext);
        INDArray dG = dS.mul(i);

        dI.muli(i.mul(Nd4j.ones().sub(i)));
        dF.muli(f.mul(Nd4j.ones().sub(f)));
        dO.muli(o.mul(Nd4j.ones().sub(o)));
        dG.muli(Nd4j.ones().sub(Transforms.pow(g, 2)));

        INDArray dA = Nd4j.hstack(dF, dG, dI, dO);

        INDArray dWh = Transforms.dot(hPrev.transpose(), dA);
        INDArray dWx = Transforms.dot(x.transpose(), dA);
        INDArray dB = dA.sum(0);

        this.grads.remove(0);
        this.grads.remove(1);
        this.grads.remove(2);
        this.grads.add(dWx);
        this.grads.add(dWh);
        this.grads.add(dB);

        INDArray dX = Transforms.dot(dA, Wx.transpose());
        INDArray dHPrev = Transforms.dot(dA, Wh.transpose());

        Triple<INDArray, INDArray, INDArray> triple = Triple.of(dX, dHPrev, dCPrev);

        return triple;
    }

}

