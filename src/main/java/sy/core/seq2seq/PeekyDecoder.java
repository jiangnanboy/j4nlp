package sy.core.seq2seq;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.time_layers.TimeAffine;
import sy.common.time_layers.TimeEmbedding;
import sy.common.time_layers.TimeLSTM;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class PeekyDecoder {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    TimeEmbedding embed = null;
    TimeLSTM lstm = null;
    TimeAffine affine = null;
    long cache;

    public PeekyDecoder(int vocabSize, int wordVecSize, int hiddenSize) {
        // init weights
        INDArray embedW = Nd4j.rand(DataType.FLOAT, vocabSize, wordVecSize).div(100);
        INDArray lstmWx = Nd4j.rand(DataType.FLOAT, hiddenSize + wordVecSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize + wordVecSize));
        INDArray lstmWh = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmB = Nd4j.zeros(DataType.FLOAT, 4 * hiddenSize);
        INDArray affineW = Nd4j.rand(DataType.FLOAT, hiddenSize + hiddenSize, vocabSize).div(Math.sqrt(hiddenSize + hiddenSize));
        INDArray affineB = Nd4j.zeros(DataType.FLOAT, vocabSize);

        this.embed = new TimeEmbedding(embedW);
        this.lstm = new TimeLSTM(lstmWx, lstmWh, lstmB, true);
        this.affine = new TimeAffine(affineW, affineB);

        this.params = new ArrayList();
        this.params.addAll(this.embed.params);
        this.params.addAll(this.lstm.params);
        this.params.addAll(this.affine.params);
        this.grads = new ArrayList();
        this.grads.addAll(this.embed.grads);
        this.grads.addAll(this.lstm.grads);
        this.grads.addAll(this.affine.grads);
    }

    public INDArray forward(INDArray xs, INDArray h) {
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        N = h.shape()[0];
        long H = h.shape()[1];
        this.lstm.setState(h);

        INDArray out = this.embed.forward(xs);
        INDArray hs = h.repeat(0, T).reshape(N, T, H);
        out = Nd4j.concat(2, hs, out);
        out = this.lstm.forward(out);
        out = Nd4j.concat(2, hs, out);

        INDArray score = this.affine.forward(out);
        this.cache = H;
        return score;
    }

    public INDArray backward(INDArray dScore) {
        long H = this.cache;
        INDArray dout = this.affine.backward(dScore);
        dout = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(H, dout.shape()[2]));
        INDArray dhs0 = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(H, dout.shape()[2]));
        dout = this.lstm.backward(dout);
        INDArray dembed = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(H, dout.shape()[2]));
        INDArray dhs1 = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        this.embed.backward(dembed);
        INDArray dhs = dhs0.add(dhs1);
        INDArray dh = this.lstm.dH.add(Nd4j.sum(dhs, 1));
        return dh;
    }

    public List<Long> generate(INDArray h, long startId, int sampleSize) {
        List<Long> sampled = new ArrayList();
        long charId = startId;
        this.lstm.setState(h);

        long H = h.shape()[1];
        INDArray peekyH = h.reshape(1, 1, H);
        for(int i=0; i<sampleSize; i++) {
            List<Long> charIdList = new ArrayList();
            charIdList.add(charId);
            INDArray x = Nd4j.create(charIdList).reshape(1, 1);
            INDArray out = this.embed.forward(x);
            out = Nd4j.concat(2, peekyH, out);
            out = this.lstm.forward(out);
            out = Nd4j.concat(2, peekyH, out);
            INDArray score = this.affine.forward(out);
            charId = Nd4j.argMax(Nd4j.toFlattened(score)).getNumber().longValue();
            sampled.add(charId);
        }

        return sampled;
    }

}

