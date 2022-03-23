package sy.common.time_layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/14 21:31
 */
public class TimeBiLSTM {

    TimeLSTM forwardLstm = null;
    TimeLSTM backwardLstm = null;
    List<INDArray> params = null;
    List<INDArray> grads = null;

    public TimeBiLSTM() {}
    public TimeBiLSTM(INDArray Wx1, INDArray Wh1, INDArray b1, INDArray Wx2, INDArray Wh2, INDArray b2) {
        new TimeBiLSTM(Wx1, Wh1, b1, Wx2, Wh2, b2, false);
    }
    public TimeBiLSTM(INDArray Wx1, INDArray Wh1, INDArray b1, INDArray Wx2, INDArray Wh2, INDArray b2, boolean stateful) {
        this.forwardLstm = new TimeLSTM(Wx1, Wh1, b1, stateful);
        this.backwardLstm = new TimeLSTM(Wx2, Wh2, b2, stateful);
        this.params = new ArrayList<>();
        this.params.addAll(this.forwardLstm.params);
        this.params.addAll(this.backwardLstm.params);
        this.grads = new ArrayList<>();
        this.grads.addAll(this.forwardLstm.grads);
        this.grads.addAll(this.backwardLstm.grads);
    }

    public INDArray forward(INDArray xs) {
        INDArray o1 = this.forwardLstm.forward(xs);
        INDArray o2 = this.backwardLstm.forward(xs.get());
        CustomOp op = DynamicCustomOp.builder("reverse")
                .addInputs(o2)
                .addOutputs(o2)
                .addIntegerArguments(1)
                .build();
        Nd4j.getExecutioner().exec(op);

        INDArray out = Nd4j.concat(2, o1, o2);

        return out;
    }

    public INDArray backward(INDArray dhs) {
        long H = dhs.shape()[2] / 2;
        INDArray do1 = dhs.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, H));
        INDArray do2 = dhs.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(H, dhs.shape()[2]));

        INDArray dxs1 = this.forwardLstm.backward(do1);
        CustomOp op1 = DynamicCustomOp.builder("reverse")
                .addInputs(do2)
                .addOutputs(do2)
                .addIntegerArguments(1)
                .build();
        Nd4j.getExecutioner().exec(op1);

        INDArray dxs2 = this.backwardLstm.backward(do2);
        CustomOp op2 = DynamicCustomOp.builder("reverse")
                .addInputs(dxs2)
                .addOutputs(dxs2)
                .addIntegerArguments(1)
                .build();
        Nd4j.getExecutioner().exec(op2);
        INDArray dxs = dxs1.add(dxs2);

        return dxs;
    }

}

