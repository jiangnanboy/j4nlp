package sy.common.time_layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.layers.SigmoidWithLoss;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 20:31
 */
public class TimeSigmoidWithLoss {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    List<SigmoidWithLoss> layers = null;
    long[] xsShape;

    public TimeSigmoidWithLoss() {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
    }

    public double forward(INDArray xs, INDArray ts) {
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        this.xsShape = xs.shape();

        this.layers = new ArrayList<>();
        double loss = 0;

        for(int t=0; t<T; t++) {
            SigmoidWithLoss layer = new SigmoidWithLoss();
            INDArray indArrayLoss = layer.forward(xs.get(NDArrayIndex.all(), NDArrayIndex.point(t)), ts.get(NDArrayIndex.all(), NDArrayIndex.point(t)));
            loss += indArrayLoss.getDouble(0);
            this.layers.addAll(layers);
        }

        return loss / T;
    }

    public INDArray backward() {
        return this.backward(1);
    }

    public INDArray backward(int dout) {
        long N = this.xsShape[0];
        long T = this.xsShape[1];

        INDArray dxs = Nd4j.rand(DataType.FLOAT, this.xsShape);

        dout *= 1 / T;

        for(int t=0; t<T; t++) {
            SigmoidWithLoss layer = this.layers.get(t);
            dxs.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.point(t)}, layer.backward(dout));
        }

        return dxs;
    }

}

