package sy.common.trainer;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.Util;
import sy.common.optimizer.Optimizer;
import sy.core.w2v.AbsW2V;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

/**
 * @author sy
 * @date 2022/3/18 22:57
 */


public class RnnLmTrainer<M extends AbsW2V, O extends Optimizer> {

    M model = null;
    O optimizer = null;
    int timeIdx = 0;
    List<Double> pplList = null;
    int evalInterval = 0;
    int currentEpoch = 0;
    public RnnLmTrainer(M model, O optimizer) {
        this.model = model;
        this.optimizer = optimizer;
    }

    public Pair<INDArray, INDArray> getBatch(INDArray x, INDArray t, long batchSize, int timeSize) {
        INDArray batchX =  Nd4j.rand(DataType.INT32, batchSize, timeSize);
        INDArray batchT = Nd4j.rand(DataType.INT32, batchSize, timeSize);
        long dataSize = x.shape()[0];
        long jump = dataSize / batchSize;
        // mini-batch的各笔样本数据的开始位置
        List<Long> offsets = LongStream.range(0, batchSize).map(i -> i * jump).boxed().collect(Collectors.toList());

        for(int time=0; time < timeSize; time++) {
            for(int i=0; i < offsets.size(); i ++) {
                batchX.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(time)}, x.get(NDArrayIndex.point((offsets.get(i) + this.timeIdx) % dataSize)));
                batchT.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(time)}, t.get(NDArrayIndex.point((offsets.get(i) + this.timeIdx) % dataSize)));
            }
            this.timeIdx += 1;
        }
        return Pair.of(batchX, batchT);
    }

    public void fit(INDArray xs, INDArray ts, int maxEpoch, long batchSize, int timeSize, Object maxGrad, int evalInterval) {
        long dataSize = xs.shape()[0];
        long maxIters = dataSize / (batchSize * timeSize);
        this.timeIdx = 0;
        this.pplList = new ArrayList<>();
        this.evalInterval = evalInterval;
        double totalLosss = 0;
        int lossCount = 0;

        Instant stime = java.time.Instant.now();
        for(int epoch=0; epoch < maxEpoch; epoch ++) {
            for(int iter=0; iter < maxIters; iter ++) {
                Pair<INDArray, INDArray> dataPair = this.getBatch(xs, ts, batchSize, timeSize);
                // 计算梯度，更新参数
                double loss = this.model.forward(dataPair.getLeft(), dataPair.getRight()).getNumber().doubleValue();
                this.model.backward();
                // 将共享的权重整合为1个
                Pair<List<INDArray>, List<INDArray>> paramsGradPpair = Util.removeDuplicate(this.model.params, this.model.grads);
                if(Optional.ofNullable(maxGrad).isPresent()) {
                    Util.clipGrads(paramsGradPpair.getRight(), (Double) maxGrad);
                }
                this.optimizer.update(paramsGradPpair.getLeft(), paramsGradPpair.getRight());
                totalLosss += loss;
                lossCount += 1;

                // 评价困惑度
                if((evalInterval != 0) && ((iter % evalInterval) == 0)) {
                    double ppl = Math.exp(totalLosss / lossCount);
                    Instant etime = java.time.Instant.now();
                    long elapsedTime = Duration.between(stime, etime).getSeconds();
                    System.out.println("epoch" +"\t" + "iter/maxIters" + "\t" + "time" + "\t" + "perplexity");
                    System.out.println(this.currentEpoch+1 + "\t" + iter+1+"/"+maxIters + "\t" + elapsedTime + "\t" + ppl);
                    this.pplList.add(ppl);
                    totalLosss = 0;
                    lossCount = 0;
                }
            }
            this.currentEpoch += 1;
        }

    }

}


/*
    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()
 */
