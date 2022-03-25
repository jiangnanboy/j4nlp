package sy.core.seq2seq;

import org.nd4j.linalg.api.ndarray.INDArray;
import sy.common.time_layers.TimeSoftmaxWithLoss;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class PeekySeq2Seq extends Seq2Seq{

    Encoder encoder = null;
    PeekyDecoder decoder = null;
    List<INDArray> params = null;
    List<INDArray> grads = null;
    TimeSoftmaxWithLoss softmax = null;

    public PeekySeq2Seq(int vocabSize, int wordVecSize, int hiddenSize) {
        super(vocabSize, wordVecSize, hiddenSize);
        this.encoder = new Encoder(vocabSize, wordVecSize, hiddenSize);
        this.decoder = new PeekyDecoder(vocabSize, wordVecSize, hiddenSize);
        this.softmax = new TimeSoftmaxWithLoss();
        this.params = new ArrayList();
        this.params.addAll(this.encoder.params);
        this.params.addAll(this.decoder.params);
        this.grads = new ArrayList();
        this.grads.addAll(this.encoder.grads);
        this.grads.addAll(this.decoder.grads);
    }

}

