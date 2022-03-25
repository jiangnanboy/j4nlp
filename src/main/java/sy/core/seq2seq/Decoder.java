package sy.core.seq2seq;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sy.common.time_layers.TimeAffine;
import sy.common.time_layers.TimeEmbedding;
import sy.common.time_layers.TimeLSTM;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class Decoder {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    TimeEmbedding embed = null;
    TimeLSTM lstm = null;
    TimeAffine affine = null;

    public Decoder(int vocabSize, int wordVecSize, int hiddenSize) {
        // init weights
        INDArray embedW = Nd4j.rand(DataType.FLOAT, vocabSize, wordVecSize).div(100);
        INDArray lstmWx = Nd4j.rand(DataType.FLOAT, wordVecSize, 4 * hiddenSize).div(Math.sqrt(wordVecSize));
        INDArray lstmWh = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmB = Nd4j.zeros(DataType.FLOAT, 4 * hiddenSize);
        INDArray affineW = Nd4j.rand(DataType.FLOAT, hiddenSize, vocabSize).div(Math.sqrt(hiddenSize));
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
        this.lstm.setState(h);
        INDArray out = this.embed.forward(xs);
        out = this.lstm.forward(out);
        INDArray score = this.affine.forward(out);
        return score;
    }

    public INDArray backward(INDArray dScore) {
        INDArray dOut = this.affine.backward(dScore);
        dOut = this.lstm.backward(dOut);
        dOut = this.embed.backward(dOut);
        INDArray dh = this.lstm.dH;
        return dh;
    }

    public List<Long> generate(INDArray h, long startId, int sampleSize) {
        List<Long> sampled = new ArrayList();
        long sampleId = startId;
        this.lstm.setState(h);
        for(int i=0; i<sampleSize; i++) {
            List<Long> sampleStartId = new ArrayList();
            sampleStartId.add(sampleId);
            INDArray x = Nd4j.create(sampleStartId).reshape(1,1);
            INDArray out = this.embed.forward(x);
            out = this.lstm.forward(out);
            INDArray score = this.affine.forward(out);
            sampleId = Nd4j.argMax(Nd4j.toFlattened(score)).getNumber().longValue();
            sampled.add(sampleId);
        }

        return sampled;
    }

}

