package sy.core.seq2seq;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.time_layers.TimeEmbedding;
import sy.common.time_layers.TimeLSTM;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class Encoder {

    public TimeEmbedding embed = null;
    public TimeLSTM lstm = null;
    public List<INDArray> params = null;
    public List<INDArray> grads = null;
    INDArray hs = null;

    public Encoder(int vocabSize, int wordVecSize, int hiddenSize) {
        // init weights
        INDArray embedW = Nd4j.rand(DataType.FLOAT, vocabSize, wordVecSize).div(100);
        INDArray lstmWx = Nd4j.rand(DataType.FLOAT, wordVecSize, 4 * hiddenSize).div(Math.sqrt(wordVecSize));
        INDArray lstmWh = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmB = Nd4j.zeros(DataType.FLOAT, 4 * hiddenSize);

        this.embed = new TimeEmbedding(embedW);
        this.lstm = new TimeLSTM(lstmWx, lstmWh, lstmB, false);

        this.params = new ArrayList();
        this.params.addAll(this.embed.params);
        this.params.addAll(this.lstm.params);
        this.grads = new ArrayList();
        this.grads.addAll(this.embed.grads);
        this.grads.addAll(this.lstm.grads);
    }

    public INDArray forward(INDArray xs) {
        xs = this.embed.forward(xs);
        this.hs = this.lstm.forward(xs);
        return hs.get(NDArrayIndex.all(), NDArrayIndex.point(this.hs.shape()[1] - 1), NDArrayIndex.all());
    }

    public INDArray backward(INDArray dh) {
        INDArray dhs = Nd4j.zerosLike(this.hs);
        dhs.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(dhs.shape()[1] - 1), NDArrayIndex.all()}, dh);
        INDArray dout = this.lstm.backward(dhs);
        dout = this.embed.backward(dout);
        return dout;
    }

}

