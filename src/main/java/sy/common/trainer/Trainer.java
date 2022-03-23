package sy.common.trainer;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import sy.common.Util;
import sy.common.optimizer.Optimizer;
import sy.core.w2v.AbsW2V;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * @author sy
 * @date 2022/3/10 22:26
 */

public class Trainer<M extends AbsW2V, O extends Optimizer> {
    M model = null;
    O optimizer = null;
    List<Double> lossList = null;
    int evalInterval;
    int currentEpoch = 0;

    public Trainer(M model, O optimizer) {
        this.model = model;
        this.optimizer = optimizer;
        this.lossList = new ArrayList<>();
    }

    public void fit(INDArray x, INDArray t, int maxEpoch, long batchSize, Object maxGrad, int evalInterval) {
        long dataSize = x.shape()[0];
        long maxIters = dataSize / batchSize;
        if(evalInterval != 0) {
            this.evalInterval = evalInterval;
        }
        double totalLoss = 0;
        int lossCount = 0;

        Instant stime = java.time.Instant.now();

        for(int epoch=0; epoch < maxEpoch; epoch ++) {
            // 打乱
            Nd4j.shuffle(x, 1);
            Nd4j.shuffle(t, 1);
            for(int iter=0; iter < maxIters; iter ++) {
                INDArray batchX = x.get(NDArrayIndex.interval(iter * batchSize, (iter + 1) * batchSize));
                INDArray batchT = t.get(NDArrayIndex.interval(iter * batchSize, (iter + 1) * batchSize));

                // 计算梯度，更新参数
                double loss = this.model.forward(batchX, batchT).getNumber().doubleValue();
                this.model.backward();
                Pair<List<INDArray>, List<INDArray>> pair = Util.removeDuplicate(this.model.params, this.model.grads);
                if(Optional.ofNullable(maxGrad).isPresent()) {
                    Util.clipGrads(pair.getRight(), (Double) maxGrad);
                }
                this.optimizer.update(pair.getLeft(), pair.getRight());
                totalLoss += loss;
                lossCount += 1;

                // 评估
                if((evalInterval != 0) && ((iter % evalInterval) == 0)) {
                    double avgLoss = totalLoss / lossCount;
                    Instant etime = java.time.Instant.now();
                    long elapsedTime = Duration.between(stime, etime).getSeconds();
                    System.out.println("epoch" +"\t" + "iter/maxIters" + "\t" + "time" + "\t" + "loss");
                    System.out.println(this.currentEpoch+1 + "\t" + iter+1+"/"+maxIters + "\t" + elapsedTime + "\t" + avgLoss);
                    this.lossList.add(avgLoss);
                    totalLoss = 0;
                    lossCount = 0;
                }
            }
            this.currentEpoch += 1;
        }

    }

}

/*
    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()
     */