package sy.core.w2v.negative_sampling_layer;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

/**
 * @author sy
 * @date 2022/3/23 23:02
 */
public class UnigramSampler {

    int sampleSize;
    int vocabSize;
    INDArray wordP;

    public UnigramSampler(INDArray corpus, double power, int sampleSize) {
        this.sampleSize = sampleSize;

        Map<Long, Integer> countMap = new HashMap<>();
        for(int i=0; i<corpus.length(); i++) {
            long key = corpus.getLong(i);
            if (countMap.containsKey(key)) {
                countMap.put(key, countMap.get(key) + 1);
            } else {
                countMap.put(key, 1);
            }
        }

        this.vocabSize = countMap.size();
        this.wordP = Nd4j.zeros(vocabSize);
        for(int i=0; i<vocabSize; i++) {
            this.wordP.put(new INDArrayIndex[]{NDArrayIndex.point(i)}, countMap.get(i));
        }
        this.wordP = Transforms.pow(this.wordP, power);
        this.wordP.divi(Nd4j.sum(this.wordP));
    }

    public INDArray getNegativeSample(INDArray target) {
        long batchSize = target.shape()[0];
        INDArray negativeSample = Nd4j.zeros(DataType.INT32, batchSize, this.sampleSize);
        for(int i=0; i<batchSize; i++) {
            INDArray p = Nd4j.zerosLike(this.wordP);
            Nd4j.copy(this.wordP, p);
            long targetIdx = target.get(NDArrayIndex.point(i)).getNumber().longValue();
            p.put(new INDArrayIndex[]{NDArrayIndex.point(targetIdx)}, 0);
            p.divi(p.sum());
            negativeSample.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all()}, Nd4j.choice(Nd4j.arange(this.vocabSize), p, this.sampleSize, Nd4j.getRandom()));
        }

        return negativeSample;
    }

}

