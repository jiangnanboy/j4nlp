package sy.common.time_layers;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:31
 */
public class TimeLSTM {

    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    List<LSTM> layers = null;
    INDArray h = null;
    INDArray c = null;
    public INDArray dH = null;
    boolean stateful = false;

    public TimeLSTM() {}
    public TimeLSTM(INDArray Wx, INDArray Wh, INDArray b) {
        new TimeLSTM(Wx, Wh, b, false);
    }
    public TimeLSTM(INDArray Wx, INDArray Wh, INDArray b, boolean stateful) {
        this.params = new ArrayList<>();
        this.params.add(Wx);
        this.params.add(Wh);
        this.params.add(b);

        this.grads = new ArrayList<>();
        this.grads.add(Nd4j.zerosLike(Wx));
        this.grads.add(Nd4j.zerosLike(Wh));
        this.grads.add(Nd4j.zerosLike(b));

        this.stateful = stateful;
    }

    public INDArray forward(INDArray xs) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        long D = xs.shape()[2];
        long H = Wh.shape()[0];

        this.layers = new ArrayList<>();
        INDArray hs = Nd4j.rand(DataType.FLOAT, N, T, H);

        if((!this.stateful) || (null == this.h)) {
            this.h = Nd4j.zeros(DataType.FLOAT, N, H);
        }
        if((!this.stateful) || (null == this.c)) {
            this.c = Nd4j.zeros(DataType.FLOAT, N, H);
        }

        for(int t=0; t<T; t++) {
            LSTM layer = new LSTM(this.params.get(0), this.params.get(1), this.params.get(2));
            Pair<INDArray, INDArray> pair = layer.forward(xs.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()), this.h, this.c);
            this.h = pair.getLeft();
            this.c = pair.getRight();
            hs.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, this.h);
            this.layers.add(layer);
        }

        return hs;
    }

    public INDArray backward(INDArray dhs) {
        INDArray Wx = this.params.get(0);
        INDArray Wh = this.params.get(1);
        INDArray b = this.params.get(2);
        long N = dhs.shape()[0];
        long T = dhs.shape()[1];
        long H = dhs.shape()[2];
        long D = Wx.shape()[0];

        INDArray dxs = Nd4j.rand(DataType.FLOAT, N, T, D);
        INDArray dh = Nd4j.zeros();
        INDArray dc = Nd4j.zeros();

        List<INDArray> grads = new ArrayList<>();
        grads.add(Nd4j.zeros());
        grads.add(Nd4j.zeros());
        grads.add(Nd4j.zeros());

        for(int t = (int) (T-1); t >=0; t--) {
            LSTM layer = this.layers.get(t);
            Triple<INDArray, INDArray, INDArray> triple = layer.backward(dhs.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()).add(dh), dc);
            INDArray dx = triple.getLeft();
            dh = triple.getMiddle();
            dc = triple.getRight();
            dxs.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, dx);
            for(int i=0;i < layer.grads.size();i++) {
                grads.get(i).addi(layer.grads.get(i));
            }
        }

        for(int i=0; i < grads.size(); i++) {
            this.grads.remove(i);
            this.grads.add(i, grads.get(i));
        }

        this.dH = dh;

        return dxs;
    }

    public void setState(INDArray h) {
        this.setState(h, null);
    }

    public void setState(INDArray h, INDArray c) {
        this.h = h;
        this.c = c;
    }

    public void resetState() {
        this.h = null;
        this.c = null;
    }

}

