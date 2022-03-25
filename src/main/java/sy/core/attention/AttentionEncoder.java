package sy.core.attention;

import org.nd4j.linalg.api.ndarray.INDArray;
import sy.core.seq2seq.Encoder;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class AttentionEncoder extends Encoder {

    public AttentionEncoder(int vocabSize, int wordVecSize, int hiddenSize) {
        super(vocabSize, wordVecSize, hiddenSize);
    }

    @Override
    public INDArray forward(INDArray xs) {
        xs = this.embed.forward(xs);
        INDArray hs = this.lstm.forward(xs);
        return hs;
    }

    @Override
    public INDArray backward(INDArray dhs) {
        INDArray dout = this.lstm.backward(dhs);
        dout = this.embed.backward(dout);
        return dout;
    }

}

