package sy.common;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author sy
 * @date 2022/3/10 22:24
 */
public class Functions {

    /**
     * @param indArray
     * @return
     */
    public static INDArray sigmoid(INDArray indArray) {
        indArray = (Transforms.exp(Transforms.neg(indArray)).add(1));
        INDArray one = Nd4j.ones();
        return one.div(indArray);
    }

    /**
     * @param indArray
     * @return
     */
    public static INDArray relu(INDArray indArray) {
        INDArray zero = Nd4j.zeros();
        return Transforms.max(zero, indArray);
    }

    /**
     * @param indArray
     * @return
     */
    public static INDArray softmax(INDArray indArray) {
        if(2 == indArray.rank()) {
            indArray = indArray.sub(indArray.max(true, 1));
            indArray = Transforms.exp(indArray);
            indArray = indArray.div(indArray.sum(true, 1));
        } else if(1 == indArray.rank()) {
            indArray = indArray.sub(Nd4j.max(indArray));
            indArray = Transforms.exp(indArray).div(Nd4j.sum(Transforms.exp(indArray)));
        }
        return indArray;
    }

    /**
     * @param y
     * @param t
     * @return
     */
    public static INDArray crossEntropyError(INDArray y, INDArray t) {
        if(1 == y.rank()) {
            y = y.reshape(1, y.length());
            t = t.repeat(1, t.length());
        }
        /**
         * in the case of a supervised label as one-hot-vector,
         * the index is converted to the correct solution label
         */
        if(t.length() == y.length()) {
            t = t.argMax(1);
        }
        long batchSize = y.shape()[0];
        int[] tArgIndexIntVec = t.toIntVector();
        double[] yValueDouVec = new double[tArgIndexIntVec.length];
        for(int idx=0; idx < tArgIndexIntVec.length; idx++) {
            yValueDouVec[idx] = y.getDouble(idx, tArgIndexIntVec[idx]);
        }
        INDArray yValueIndArray = Nd4j.create(yValueDouVec);
        INDArray crossEntropyResult = (Nd4j.sum(Transforms.log(yValueIndArray.add(1e-7))).neg()).div(batchSize);
        return crossEntropyResult;
    }

    /**
     * supervised data is in the form of one-hot -> (t)
     * @param y
     * @param t
     * @return
     */
    public static INDArray crossEntropyErrorOneHot(INDArray y, INDArray t) {
        if(1 == y.rank()) {
            y = y.reshape(1, y.length());
            t = t.repeat(1, t.length());
        }
        long batchSize = y.shape()[0];
        INDArray crossEntropyResult = (Nd4j.sum(t.mul(Transforms.log(y.add(1e-7)))).neg()).div(batchSize);
        return crossEntropyResult;
    }

}


