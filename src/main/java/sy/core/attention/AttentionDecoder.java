package sy.core.attention;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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
public class AttentionDecoder {

    List<INDArray> params;
    List<INDArray> grads;
    TimeEmbedding embed;
    TimeLSTM lstm;
    TimeAttention attention;
    TimeAffine affine;

    public AttentionDecoder(int vocabSize, int wordVecSize, int hiddenSize) {
        // init weights
        INDArray embedW = Nd4j.rand(DataType.FLOAT, vocabSize, wordVecSize).div(100);
        INDArray lstmWx = Nd4j.rand(DataType.FLOAT, wordVecSize, 4 * hiddenSize).div(Math.sqrt(wordVecSize));
        INDArray lstmWh = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmB = Nd4j.zeros(DataType.FLOAT, 4 * hiddenSize);

        INDArray affineW = Nd4j.rand(DataType.FLOAT, 2 * hiddenSize, vocabSize).div(Math.sqrt(wordVecSize));
        INDArray affineB = Nd4j.zeros(DataType.FLOAT, vocabSize);

        this.embed = new TimeEmbedding(embedW);
        this.lstm = new TimeLSTM(lstmWx, lstmWh, lstmB, true);
        this.attention = new TimeAttention();
        this.affine = new TimeAffine(affineW, affineB);

        this.params = new ArrayList();
        this.params.addAll(this.embed.params);
        this.params.addAll(this.lstm.params);
        this.params.addAll(this.attention.params);
        this.params.addAll(this.affine.params);
        this.grads = new ArrayList();
        this.grads.addAll(this.embed.grads);
        this.grads.addAll(this.lstm.grads);
        this.grads.addAll(this.attention.grads);
        this.grads.addAll(this.affine.grads);
    }

    public INDArray forward(INDArray xs, INDArray encHs) {
        INDArray h = encHs.get(NDArrayIndex.all(), NDArrayIndex.point(encHs.shape()[1] - 1));
        this.lstm.setState(h);

        INDArray out = this.embed.forward(xs);
        INDArray decHs = this.lstm.forward(out);
        INDArray c = this.attention.forward(encHs, decHs);
        out = Nd4j.concat(2, c, decHs);
        INDArray score = this.affine.forward(out);

        return score;
    }

    public INDArray backward(INDArray dScore) {
        INDArray dout = this.affine.backward(dScore);
        long N = dout.shape()[0];
        long T = dout.shape()[1];
        long H2 = dout.shape()[2];
        long H = H2 / 2;

        INDArray dc = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        INDArray ddecHs0 = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(H, dout.shape()[2] - 1));
        Pair<INDArray, INDArray> pair = this.attention.backward(dc);
        INDArray dencHs = pair.getLeft();
        INDArray ddecHs1 = pair.getRight();
        INDArray ddecHs = ddecHs0.add(ddecHs1);
        dout = this.lstm.backward(ddecHs);
        INDArray dh = this.lstm.dH;
        INDArray dencHsIndArray = dencHs.get(NDArrayIndex.all(), NDArrayIndex.point(dencHs.shape()[1] - 1));
        dencHsIndArray.addi(dh);
        dencHs.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(dencHs.shape()[1] - 1)}, dencHsIndArray);
        this.embed.backward(dout);

        return dencHs;
    }

    public List<Long> generate(INDArray encHs, long startId, int sampleSize) {
        List<Long> sampled = new ArrayList();
        long sampleId = startId;
        INDArray h = encHs.get(NDArrayIndex.all(), NDArrayIndex.point(encHs.shape()[1] - 1));
        this.lstm.setState(h);

        for(int i=0; i<sampleSize; i++) {
            List<Long> sampleIdList = new ArrayList();
            sampled.add(sampleId);
            INDArray x = Nd4j.create(sampleIdList).reshape(1, 1);
            INDArray out = this.embed.forward(x);
            INDArray decHs = this.lstm.forward(out);
            INDArray c = this.attention.forward(encHs, decHs);
            out = Nd4j.concat(2, c, decHs);
            INDArray score = this.affine.forward(out);

            sampleId = Nd4j.argMax(Nd4j.toFlattened(score)).getNumber().longValue();
            sampled.add(sampleId);
        }

        return sampled;
    }
}
