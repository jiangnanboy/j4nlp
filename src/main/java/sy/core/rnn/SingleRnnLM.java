package sy.core.rnn;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sy.common.BaseModel;
import sy.common.time_layers.TimeAffine;
import sy.common.time_layers.TimeEmbedding;
import sy.common.time_layers.TimeLSTM;
import sy.common.time_layers.TimeSoftmaxWithLoss;
import sy.define_exception.NumberParameterException;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/23 22:05
 */
public class SingleRnnLM extends BaseModel {

    List<Object> layers = null;
    TimeSoftmaxWithLoss lossLayer = null;
    TimeLSTM lstmLayer = null;

    public SingleRnnLM() {
        new SingleRnnLM(10000, 100, 100);
    }
    public SingleRnnLM(int vocabSize, int wordVecSize, int hiddenSize) {
        // init weights
        INDArray embedW = Nd4j.rand(DataType.FLOAT, vocabSize, wordVecSize).div(100);
        INDArray lstmWx = Nd4j.rand(DataType.FLOAT, wordVecSize, 4 * hiddenSize).div(Math.sqrt(wordVecSize));
        INDArray lstmWh = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmB = Nd4j.zeros(DataType.FLOAT, 4 * hiddenSize);

        INDArray affineW = Nd4j.rand(DataType.FLOAT, hiddenSize, vocabSize).div(Math.sqrt(hiddenSize));
        INDArray affineB = Nd4j.zeros(DataType.FLOAT, vocabSize);

        // build generation layer
        this.layers = new ArrayList<>();
        this.layers.add(new TimeEmbedding(embedW));
        this.layers.add(new TimeLSTM(lstmWx, lstmWh, lstmB, true));
        this.layers.add(new TimeAffine(affineW, affineB));

        this.lossLayer = new TimeSoftmaxWithLoss();
        this.lstmLayer = (TimeLSTM) this.layers.get(1);

        // put all the weights and gradients into the list
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        this.params.addAll(((TimeEmbedding)this.layers.get(0)).params);
        this.params.addAll(((TimeLSTM) this.layers.get(1)).params);
        this.params.addAll(((TimeAffine)this.layers.get(2)).params);

        this.grads.addAll(((TimeEmbedding)this.layers.get(0)).grads);
        this.grads.addAll(((TimeLSTM) this.layers.get(1)).grads);
        this.grads.addAll(((TimeAffine)this.layers.get(2)).grads);
    }

    public INDArray predict(INDArray xs) {
        xs = ((TimeEmbedding)this.layers.get(0)).forward(xs);
        xs = ((TimeLSTM)this.layers.get(1)).forward(xs);
        xs = ((TimeAffine)this.layers.get(2)).forward(xs);
        return xs;
    }

    @Override
    public INDArray forward(Object ... xs) throws NumberParameterException{
        if(xs.length != 2) {
            throw new NumberParameterException("Need 2 parameters -> " + xs);
        }
        INDArray score = this.predict((INDArray) xs[0]);
        INDArray loss = this.lossLayer.forward(score, (INDArray) xs[1]);
        return loss;
    }

    @Override
    public INDArray backward() throws NumberParameterException {
        return backward(Nd4j.ones());
    }

    @Override
    public INDArray backward(Object ... dout) throws NumberParameterException{
        if(dout.length != 1) {
            throw new NumberParameterException("Need 1 parameters -> " + dout);
        }
        INDArray doutIndArray = this.lossLayer.backward((INDArray) dout[0]);
        doutIndArray = ((TimeAffine)this.layers.get(2)).backward(doutIndArray);
        doutIndArray = ((TimeLSTM)this.layers.get(1)).backward(doutIndArray);
        doutIndArray = ((TimeEmbedding)this.layers.get(0)).backward(doutIndArray);
        return doutIndArray;
    }

    public void resetState() {
        this.lstmLayer.resetState();
    }

}

