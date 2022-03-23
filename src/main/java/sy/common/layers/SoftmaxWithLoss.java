package sy.common.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import sy.common.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 19:24
 */
public class SoftmaxWithLoss {
    List<INDArray> params = null;
    List<INDArray> grads = null;
    INDArray y = null; // the output of softmax
    INDArray t = null; // the target

    public SoftmaxWithLoss() {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
    }

    public INDArray forward(INDArray x, INDArray t) {
        this.t = t;
        this.y = Functions.softmax(x);
        /**
         * in the case of a supervised label as one-hot-vector,
         * the index is converted to the correct solution label
         */
        if(this.t.length() == this.y.length()) {
            this.t = this.t.argMax(1);
        }
        INDArray loss = Functions.crossEntropyError(this.y, this.t);
        return loss;
    }

    public INDArray backward(int dout) {
        long batchSize = this.t.shape()[0];

        int[] tArgIndexIntVec = t.toIntVector();
        double[] yValueDouVec = new double[tArgIndexIntVec.length];
        for(int idx=0; idx < tArgIndexIntVec.length; idx++) {
            yValueDouVec[idx] = y.getDouble(idx, tArgIndexIntVec[idx]);
        }
        INDArray yValueIndArray = Nd4j.create(yValueDouVec);
        yValueIndArray.subi(1);
        yValueIndArray.muli(dout);
        yValueIndArray = yValueIndArray.div(batchSize);
        return yValueIndArray;
    }

    public INDArray backward() {
        return this.backward(1);
    }

}


