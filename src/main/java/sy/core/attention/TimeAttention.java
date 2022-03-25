package sy.core.attention;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.time_layers.TimeAffine;
import sy.common.time_layers.TimeSoftmaxWithLoss;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class TimeAttention {
    List<INDArray> params;
    List<INDArray> grads;
    List<Attention> layers;
    List<INDArray> attentionWeights;

    public TimeAttention() {
        this.params = new ArrayList();
        this.grads = new ArrayList();
        this.layers = null;
        this.attentionWeights = null;
    }

    public INDArray forward(INDArray hsEnc, INDArray hsDec) {
        long N = hsDec.shape()[0];
        long T = hsDec.shape()[1];
        long H = hsDec.shape()[2];
        INDArray out = Nd4j.zerosLike(hsDec);
        this.layers = new ArrayList();
        this.attentionWeights = new ArrayList();

        for(int t=0; t<T; t++) {
            Attention layer = new Attention();
            out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, layer.forward(hsEnc, hsDec.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all())));
            this.layers.add(layer);
            this.attentionWeights.add(layer.attentionWeight);
        }

        return out;
    }

    public Pair<INDArray, INDArray> backward(INDArray dout) {
        long N = dout.shape()[0];
        long T = dout.shape()[1];
        long H = dout.shape()[2];

        INDArray dhsEnc = Nd4j.zeros();
        INDArray dhsDec = Nd4j.zerosLike(dout);
        for(int t=0; t<T; t++) {
            Attention layer = this.layers.get(t);
            Pair<INDArray, INDArray> pair = layer.backward(dout.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()));
            INDArray dhs = pair.getLeft();
            INDArray dh = pair.getRight();
            dhsEnc.addi(dhs);
            dhsDec.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, dh);
        }

        return Pair.of(dhsEnc, dhsDec);
    }

}

