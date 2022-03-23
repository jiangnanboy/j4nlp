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
 * @date 2022/3/23 23:59
 */
public class SkipGram extends AbsW2V {

    Embedding inLayer;
    List<INDArray> params;
    List<INDArray> grads;
    INDArray wordVecs;
    List<NegativeSamplingLoss> lossLayers;

    public SkipGram(int vocabSize, int hiddenSize, int windowSize, INDArray corpus) {
        // init weights
        INDArray wIn = Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize).mul(0.01);
        INDArray wOut = Nd4j.rand(DataType.FLOAT, vocabSize, hiddenSize).mul(0.01);

        // build generation layer
        this.inLayer = new Embedding(wIn);
        this.lossLayers = new ArrayList<>();
        for(int i=0; i<2*windowSize; i++) {
            NegativeSamplingLoss layer = new NegativeSamplingLoss(wOut, corpus, 0.75, 5);
            this.lossLayers.add(layer);
        }

        // put all the weights and gradients into the list
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        for(NegativeSamplingLoss layer : this.lossLayers) {
            this.params.addAll(layer.params);
            this.grads.addAll(layer.grads);
        }
        this.params.addAll(this.inLayer.params);
        this.params.addAll(this.inLayer.grads);

        wordVecs = wIn;
    }

    @Override
    public INDArray forward(INDArray context, INDArray target) {
        INDArray h = this.inLayer.forward(target);
        INDArray loss = Nd4j.zeros();
        for(int i=0; i<this.lossLayers.size(); i++) {
            loss.addi(this.lossLayers.get(i).forward(h, context.get(NDArrayIndex.all(), NDArrayIndex.point(i))));
        }

        return loss;
    }

    @Override
    public INDArray backward() {
        return backward(1);
    }

    @Override
    public INDArray backward(int dout) {
        INDArray dh = Nd4j.zeros();
        for(NegativeSamplingLoss layer: this.lossLayers) {
            dh.addi(layer.backward(dout));
        }
        this.inLayer.backward(dh);

        return null;
    }

}

