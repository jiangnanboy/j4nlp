package sy.core.rnn;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sy.common.BaseModel;
import sy.common.time_layers.*;
import sy.define_exception.NumberParameterException;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/23 22:34
 */
public class MultiRnnLM extends BaseModel {

    List<Object> layers = null;
    TimeSoftmaxWithLoss lossLayer = null;
    List<TimeLSTM> lstmLayers = null;
    List<TimeDropout> dropoutLayers = null;

    public MultiRnnLM() {
        new MultiRnnLM(10000, 1000, 1000, 0.5);
    }
    public MultiRnnLM(int vocabSize, int wordVecSize, int hiddenSize, double dropoutRate) {
        // 2 lstm layers
        // init weights
        INDArray embedW = Nd4j.rand(DataType.FLOAT, vocabSize, wordVecSize).div(100);
        INDArray lstmWx1 = Nd4j.rand(DataType.FLOAT, wordVecSize, 4 * hiddenSize).div(Math.sqrt(wordVecSize));
        INDArray lstmWh1 = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmB1 = Nd4j.zeros(DataType.FLOAT, 4 * hiddenSize);

        INDArray lstmWx2 = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmWh2 = Nd4j.rand(DataType.FLOAT, hiddenSize, 4 * hiddenSize).div(Math.sqrt(hiddenSize));
        INDArray lstmB2 = Nd4j.zeros(DataType.FLOAT, 4 * hiddenSize);

        INDArray affineB = Nd4j.zeros(DataType.FLOAT, vocabSize);

        this.layers = new ArrayList<>();
        this.layers.add(new TimeEmbedding(embedW));
        this.layers.add(new TimeDropout(dropoutRate));
        this.layers.add(new TimeLSTM(lstmWx1, lstmWh1, lstmB1, true));
        this.layers.add(new TimeDropout(dropoutRate));
        this.layers.add(new TimeLSTM(lstmWx2, lstmWh2, lstmB2, true));
        this.layers.add(new TimeDropout(dropoutRate));
        this.layers.add(new TimeAffine(embedW.transpose(), affineB)); // weight typing

        this.lossLayer = new TimeSoftmaxWithLoss();
        this.lstmLayers = new ArrayList<>();
        this.lstmLayers.add((TimeLSTM) this.layers.get(2));
        this.lstmLayers.add((TimeLSTM) this.layers.get(4));

        this.dropoutLayers = new ArrayList<>();
        this.dropoutLayers.add((TimeDropout) this.layers.get(1));
        this.dropoutLayers.add((TimeDropout) this.layers.get(3));
        this.dropoutLayers.add((TimeDropout) this.layers.get(5));

        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();

        this.params.addAll(((TimeEmbedding)this.layers.get(0)).params);
        this.params.addAll(((TimeDropout)this.layers.get(1)).params);
        this.params.addAll(((TimeLSTM)this.layers.get(2)).params);
        this.params.addAll(((TimeDropout)this.layers.get(3)).params);
        this.params.addAll(((TimeLSTM)this.layers.get(4)).params);
        this.params.addAll(((TimeDropout)this.layers.get(5)).params);
        this.params.addAll(((TimeAffine)this.layers.get(6)).params);

        this.grads.addAll(((TimeEmbedding)this.layers.get(0)).grads);
        this.grads.addAll(((TimeDropout)this.layers.get(1)).grads);
        this.grads.addAll(((TimeLSTM)this.layers.get(2)).grads);
        this.grads.addAll(((TimeDropout)this.layers.get(3)).grads);
        this.grads.addAll(((TimeLSTM)this.layers.get(4)).grads);
        this.grads.addAll(((TimeDropout)this.layers.get(5)).grads);
        this.grads.addAll(((TimeAffine)this.layers.get(6)).grads);
    }

    public INDArray predict(INDArray xs, boolean trainFlag) {
        for(TimeDropout layer : this.dropoutLayers) {
            layer.trainFlg = trainFlag;
        }
        xs = ((TimeEmbedding)this.layers.get(0)).forward(xs);
        xs = ((TimeDropout)this.layers.get(1)).forward(xs);
        xs = ((TimeLSTM)this.layers.get(2)).forward(xs);
        xs = ((TimeDropout)this.layers.get(3)).forward(xs);
        xs = ((TimeLSTM)this.layers.get(4)).forward(xs);
        xs = ((TimeDropout)this.layers.get(5)).forward(xs);
        xs = ((TimeAffine)this.layers.get(6)).forward(xs);

        return xs;
    }

    @Override
    public INDArray forward(Object... xs) throws NumberParameterException{
        if((xs.length != 3) || (xs.length != 2)) {
            throw new NumberParameterException("Need 2 or 3 parameters-> " + xs);
        }
        boolean trainFlag = true;
        if(xs.length == 3) {
            trainFlag = (Boolean)xs[2];
        }
        INDArray score = this.predict((INDArray)xs[0], trainFlag);
        INDArray loss = this.lossLayer.forward(score, (INDArray)xs[1]);

        return loss;
    }

    @Override
    public INDArray backward() throws NumberParameterException{
        return backward(Nd4j.ones());
    }

    @Override
    public INDArray backward(Object... dout) throws NumberParameterException{
        if(dout.length != 1) {
            throw new NumberParameterException("Need 1 parameters -> " + dout);
        }
        INDArray doutIndArray = this.lossLayer.backward((INDArray) dout[0]);
        doutIndArray = ((TimeAffine)this.layers.get(6)).backward(doutIndArray);
        doutIndArray = ((TimeDropout)this.layers.get(5)).backward(doutIndArray);
        doutIndArray = ((TimeLSTM)this.layers.get(4)).backward(doutIndArray);
        doutIndArray = ((TimeDropout)this.layers.get(3)).backward(doutIndArray);
        doutIndArray = ((TimeLSTM)this.layers.get(2)).backward(doutIndArray);
        doutIndArray = ((TimeDropout)this.layers.get(1)).backward(doutIndArray);
        doutIndArray = ((TimeEmbedding)this.layers.get(0)).backward(doutIndArray);

        return doutIndArray;
    }

    public void resetState() {
        for(TimeLSTM layer : this.lstmLayers) {
            layer.resetState();
        }
    }

}

