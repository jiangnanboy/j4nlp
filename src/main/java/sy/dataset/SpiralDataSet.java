package sy.dataset;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2022/3/22 21:50
 */
public class SpiralDataSet {

    public static Pair<INDArray, INDArray> loadData() {
        long N = 100;
        long rank = 2;
        long clsNum = 3;
        INDArray x = Nd4j.zeros(N * clsNum, rank);
        INDArray t = Nd4j.zeros(DataType.INT32, N * clsNum, clsNum);

        for(long j=0; j<clsNum; j++) {
            for(long i=0; i<N; i++) {
                double rate = (double) i / N;
                double radius = 1.0 * rate;
                double theta = j * 4.0 + 4.0*rate + Math.random() * 0.2;

                long ix = N * j + i;
                List<Double> xList = new ArrayList<>();
                xList.add(radius * Math.sin(theta));
                xList.add(radius * Math.cos(theta));
                x.put(new INDArrayIndex[]{NDArrayIndex.point(ix)}, Nd4j.toFlattened(Nd4j.create(xList)));
                t.put(new INDArrayIndex[]{NDArrayIndex.point(ix), NDArrayIndex.point(j)}, 1);
            }
        }

        return Pair.of(x, t);
    }

}

