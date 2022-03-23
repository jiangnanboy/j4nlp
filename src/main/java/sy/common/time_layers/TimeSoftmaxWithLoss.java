package sy.common.time_layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import sy.common.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 20:31
 */
public class TimeSoftmaxWithLoss {

    List<INDArray> params = null;
    List<INDArray> grads = null;
    List<Object> cache = null;
    int ignoreLabel = -1;

    public TimeSoftmaxWithLoss() {}

    public INDArray forward(INDArray xs, INDArray ts) {
        long N = xs.shape()[0];
        long T = xs.shape()[1];
        long V = xs.shape()[2];

        if(ts.rank() == 3) { //在监督标签为one-hot向量的情况下
            ts = ts.argMax(2);
        }

        INDArray mask = ts.neq(this.ignoreLabel);
        // 按批次大小和时序大小进行整理（reshape）
        xs = xs.reshape(N * T, V);
        ts = ts.reshape(N * T);
        mask = mask.reshape(N * T);

        INDArray ys = Functions.softmax(xs);

        int[] tArgIndexIntVec = ts.toIntVector();
        double[] yValueDouVec = new double[tArgIndexIntVec.length];
        for(int idx=0; idx < tArgIndexIntVec.length; idx++) {
            yValueDouVec[idx] = ys.getDouble(idx, tArgIndexIntVec[idx]);
        }
        INDArray yValueIndArray = Nd4j.create(yValueDouVec);

        INDArray ls = Transforms.log(yValueIndArray);
        ls.muli(mask); //与ignore_label相应的数据将损失设为0

        INDArray loss = Nd4j.sum(ls).neg();
        loss.divi(mask);

        this.cache = new ArrayList<>();
        this.cache.add(ts);
        this.cache.add(ys);
        this.cache.add(mask);
        this.cache.add(N);
        this.cache.add(T);
        this.cache.add(V);

        return loss;
    }

    public  INDArray backward() {
        return this.backward(Nd4j.ones());
    }

    public INDArray backward(INDArray dout) {
        INDArray ts = (INDArray) this.cache.get(0);
        INDArray ys = (INDArray) this.cache.get(1);
        INDArray mask = (INDArray) this.cache.get(2);
        long N = (long) this.cache.get(3);
        long T = (long) this.cache.get(4);
        long V = (long) this.cache.get(5);

        INDArray dx = ys;

        int[] tArgIndexIntVec = ts.toIntVector();
        double[] yValueDouVec = new double[tArgIndexIntVec.length];
        for(int idx=0; idx < tArgIndexIntVec.length; idx++) {
            yValueDouVec[idx] = dx.getDouble(idx, tArgIndexIntVec[idx]);
        }
        dx = Nd4j.create(yValueDouVec);
        dx.subi(1);
        dx.muli(dout);
        dx.divi(mask.sumNumber());
        dx.muli(mask.get(NDArrayIndex.all(), NDArrayIndex.newAxis())); // 与ignore_label相应的数据将梯度设为0

        dx = dx.reshape(N, T, V);

        return dx;
    }

}

