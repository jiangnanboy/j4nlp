package sy.common.time_layers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.layers.SoftmaxWithLoss;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:32
 */
public class SimpleTimeSoftmaxWithLoss {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    List<Object> cache = null;

    public SimpleTimeSoftmaxWithLoss() {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
    }

    public double forward(INDArray xs, INDArray ts) {
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        long V = xs.shape()[2];
        List<SoftmaxWithLoss> layers = new ArrayList<>();
        double loss = 0;

        for(int t=0; t<T; t++) {
            SoftmaxWithLoss layer = new SoftmaxWithLoss();
            INDArray indArrayLoss = layer.forward(xs.get(NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()), ts.get(NDArrayIndex.all(), NDArrayIndex.point(t)));
            loss += indArrayLoss.getDouble(0);
            layers.add(layer);
        }
        loss /= T;

        this.cache = new ArrayList<>();
        this.cache.add(layers);
        this.cache.add(xs);

        return loss;
    }

    public INDArray backward() {
       return this.backward(1);
    }

    public INDArray backward(int dout) {
        List<SoftmaxWithLoss> layers = (List<SoftmaxWithLoss>) this.cache.get(0);
        INDArray xs = (INDArray) this.cache.get(1);
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        long V = xs.shape()[2];
        INDArray dxs = Nd4j.rand(DataType.FLOAT, xs.shape());

        dout *= 1/T;
        for(int t=0; t<T; t++) {
            SoftmaxWithLoss layer = layers.get(t);
            dxs.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(t), NDArrayIndex.all()}, layer.backward(dout));
        }

        return dxs;
    }

}

