package sy.core.w2v;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.layers.Embedding;
import sy.core.w2v.negative_sampling_layer.NegativeSamplingLoss;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/23 22:59
 */
public class Cbow extends AbsW2V {

    List<Embedding> inLayers;
    List<INDArray> params;
    List<INDArray> grads;
    INDArray wordVecs;
    NegativeSamplingLoss nsLoss;

    public Cbow(int vocabSize, int hiddenSize, int windowSize, INDArray corpus) {
        // init weights
        INDArray wIn = Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize).mul(0.01);
        INDArray wOut = Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize).mul(0.01);

        // build generation layer
        this.inLayers = new ArrayList<>();
        for(int i=0; i<2*windowSize; i++) {
            Embedding layer = new Embedding(wIn);
            this.inLayers.add(layer);
        }
        this.nsLoss = new NegativeSamplingLoss(wOut, corpus, 0.75, 5);

        // put all the weights and gradients into the list
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        for(Embedding layer : this.inLayers) {
            this.params.addAll(layer.params);
            this.grads.addAll(layer.grads);
        }
        this.params.addAll(this.nsLoss.params);
        this.params.addAll(this.nsLoss.grads);
        this.wordVecs = wIn;
    }

    @Override
    public INDArray forward(INDArray context, INDArray target) {
        INDArray h = Nd4j.zeros();
        for(int i=0; i<this.inLayers.size(); i++) {
            h.addi(this.inLayers.get(i).forward(context.get(NDArrayIndex.all(), NDArrayIndex.point(i))));
        }
        h.muli(Nd4j.ones().div(this.inLayers.size()));
        INDArray loss = this.nsLoss.forward(h, target);

        return loss;
    }

    @Override
    public INDArray backward() {
        return backward(1);
    }

    @Override
    public INDArray backward(int dout) {
        INDArray doutArray = this.nsLoss.backward(dout);
        doutArray.muli(Nd4j.ones().div(this.inLayers.size()));
        for(Embedding layer : this.inLayers) {
            layer.backward(doutArray);
        }
        return null;
    }

}

