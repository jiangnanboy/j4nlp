package sy.core.seq2seq;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.BaseModel;
import sy.common.time_layers.TimeSoftmaxWithLoss;
import sy.define_exception.NumberParameterException;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/24 22:30
 */
public class Seq2Seq extends BaseModel {

    Encoder encoder = null;
    Decoder decoder = null;
    TimeSoftmaxWithLoss softmax = null;

    public Seq2Seq(int vocabSize, int wordVecSize, int hiddenSize) {
        this.encoder = new Encoder(vocabSize, wordVecSize, hiddenSize);
        this.decoder = new Decoder(vocabSize, wordVecSize, hiddenSize);
        this.softmax = new TimeSoftmaxWithLoss();
        this.params = new ArrayList();
        this.params.addAll(this.encoder.params);
        this.params.addAll(this.decoder.params);
        this.grads = new ArrayList();
        this.grads.addAll(this.encoder.grads);
        this.grads.addAll(this.decoder.grads);
    }

    @Override
    public INDArray forward(Object... os) throws NumberParameterException {
        if(os.length != 2) {
            throw new NumberParameterException("Need 2 parameters -> " + os);
        }
        INDArray xs = (INDArray)os[0];
        INDArray ts = (INDArray)os[1];
        INDArray decoderXs = ts.get(NDArrayIndex.all(), NDArrayIndex.interval(0, ts.shape()[1] - 1));
        INDArray decoderTs = ts.get(NDArrayIndex.all(), NDArrayIndex.interval(1, ts.shape()[1]));
        INDArray h = this.encoder.forward(decoderXs);
        INDArray score = this.decoder.forward(decoderXs, h);
        INDArray loss = this.softmax.forward(score, decoderTs);
        return loss;
    }

    @Override
    public INDArray backward() throws NumberParameterException {
        return backward(Nd4j.ones());
    }

    @Override
    public INDArray backward(Object... ob) throws NumberParameterException {
        if(ob.length != 1) {
            throw new NumberParameterException("Need 1 parameters -> " + ob);
        }
        INDArray dout = (INDArray)ob[0];
        dout = this.softmax.backward(dout);
        INDArray dh = this.decoder.backward(dout);
        dout = this.encoder.backward(dh);
        return dout;
    }

    public List<Long> generate(INDArray xs, long startId, int sampleSize) {
        INDArray h = this.encoder.forward(xs);
        List<Long> sampled = this.decoder.generate(h, startId, sampleSize);
        return sampled;
    }

}
