package sy.core.w2v.negative_sampling_layer;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.layers.SigmoidWithLoss;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/23 23:02
 */
public class NegativeSamplingLoss {

    int sampleSize;
    UnigramSampler sampler;
    List<SigmoidWithLoss> lossLayers;
    List<EmbeddingDot> embedDotLayers;
    public List<INDArray> params;
    public List<INDArray> grads;

    public NegativeSamplingLoss(INDArray w, INDArray corpus) {
        new NegativeSamplingLoss(w, corpus, 0.75, 5);
    }

    public NegativeSamplingLoss(INDArray w, INDArray corpus, double power, int sampleSize) {
        this.sampleSize = sampleSize;
        this.sampler = new UnigramSampler(corpus, power, sampleSize);
        this.lossLayers = new ArrayList<>();
        for(int i=0; i<this.sampleSize + 1; i++) {
            this.lossLayers.add(new SigmoidWithLoss());
        }
        for(int i=0; i<this.sampleSize + 1; i++) {
            this.embedDotLayers.add(new EmbeddingDot(w));
        }
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        for(EmbeddingDot layer : this.embedDotLayers) {
            this.params.addAll(layer.params);
            this.grads.addAll(layer.grads);
        }
    }

    public INDArray forward(INDArray h, INDArray target) {
        long batchSize = target.shape()[0];
        INDArray negativeSample = this.sampler.getNegativeSample(target);

        // forward propagation of positive examples
        INDArray score = this.embedDotLayers.get(0).forward(h, target);
        INDArray correctLabel = Nd4j.ones(DataType.INT32, batchSize);
        INDArray loss = this.lossLayers.get(0).forward(score, correctLabel);

        // forward propagation of negative examples
        INDArray negativeLabel = Nd4j.zeros(DataType.INT32, batchSize);
        for(int i=0; i<this.sampleSize; i++) {
            INDArray negativeTarget = negativeSample.get(NDArrayIndex.all(), NDArrayIndex.point(i));
            score = this.embedDotLayers.get(1 + i).forward(h, negativeTarget);
            loss.addi(this.lossLayers.get(1 + i).forward(score, negativeLabel));
        }

        return loss;
    }

    public INDArray backward() {
        return backward(1);
    }

    public INDArray backward(int dout) {
        INDArray dh = Nd4j.zeros();
        for(int i=0; i<this.lossLayers.size(); i++) {
            INDArray dScore = this.lossLayers.get(i).backward(dout);
            dh.addi(this.embedDotLayers.get(i).backward(dScore));
        }

        return dh;
    }

}

