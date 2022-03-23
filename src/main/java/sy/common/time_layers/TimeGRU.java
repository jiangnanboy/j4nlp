package sy.common.time_layers;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:32
 */
public class TimeGRU {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    List<GRU> layers = null;
    INDArray h = null;
    INDArray dh = null;
    boolean stateful = false;

    public TimeGRU() {}

    public TimeGRU(INDArray Wx, INDArray Wh, INDArray b) {
        new TimeGRU(Wx, Wh, b, false);
    }
    public TimeGRU(INDArray Wx, INDArray Wh, INDArray b, boolean stateful) {
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

        for(int t=0; t<T; t++) {
            GRU layer = new GRU(this.params.get(0), this.params.get(1), this.params.get(2));
            this.h = layer.forward(xs.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()), this.h);
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
        long D= Wx.shape()[0];

        INDArray dxs = Nd4j.rand(DataType.FLOAT, N, T, D);
        INDArray dh = Nd4j.zeros();
        List<INDArray> grads = new ArrayList<>();
        grads.add(Nd4j.zeros());
        grads.add(Nd4j.zeros());
        grads.add(Nd4j.zeros());

        for(int t = (int) (T-1); t >=0; t--) {
            GRU layer = this.layers.get(t);
            Pair<INDArray, INDArray> pair = layer.backward(dhs.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()).add(dh));
            INDArray dx =  pair.getLeft();
            dh = pair.getRight();
            dxs.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, dx);

            for(int i=0;i < layer.grads.size();i++) {
                grads.get(i).addi(layer.grads.get(i));
            }
        }

        for(int i=0; i < grads.size(); i++) {
            this.grads.remove(i);
            this.grads.add(i, grads.get(i));
        }

        this.dh = dh;

        return dxs;
    }

    public void setState(INDArray h) {
        this.h = h;
    }

    public void resetState() {
        this.h = null;
    }

}

