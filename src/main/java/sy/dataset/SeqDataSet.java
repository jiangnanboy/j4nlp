package sy.dataset;

import com.sun.corba.se.impl.resolver.SplitLocalResolverImpl;
import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author sy
 * @date 2022/3/22 21:49
 */
public class SeqDataSet {

    private static final Logger LOGGER = LoggerFactory.getLogger(SeqDataSet.class);

     static Map<Long, String> idToChar;
     static Map<String, Long> charToId;

    public static void main(String[] args) {
        List<Double> list = new ArrayList<>();
        list.add(1.4);
        list.add(5.6);
        System.out.println(Nd4j.create(list));
        System.out.println(Nd4j.toFlattened(Nd4j.create(list)));
    }

    private static void updateVocab(String text) {
        List<String> charList = Stream.of(text.split("")).collect(Collectors.toList());
        for(int i=0; i<charList.size(); i++) {
            if(!charToId.containsKey(charList.get(i))) {
                long id = charToId.size();
                charToId.put(charList.get(i), id);
                idToChar.put(id, charList.get(i));
            }
        }
    }

    public static Pair<Map<String, Long>, Map<Long, String>> getVocab() {
        return Pair.of(charToId, idToChar);
    }

    public static Pair<List<INDArray>, List<INDArray>> loadData(String fileName) {
        if(Files.notExists(Paths.get(fileName))) {
            LOGGER.info("No file -> " + fileName);
            return null;
        }

        List<String> questionList = new ArrayList<>();
        List<String> answerList = new ArrayList<>();
        try(Stream<String> stream = Files.lines(Paths.get(fileName))) {
            stream.forEach(line -> {
                line = line.trim();
                int idx = line.indexOf("_");
                questionList.add(line.substring(0, idx));
                questionList.add(line.substring(idx));
            });
        } catch (IOException e) {
            e.printStackTrace();
        }

        // create vocab dict
        for(int i=0; i<questionList.size(); i++) {
            updateVocab(questionList.get(i));
            updateVocab(answerList.get(i));
        }

        // create ndarray
        INDArray x = Nd4j.zeros(DataType.INT32, questionList.size(), questionList.get(0).length());
        INDArray t = Nd4j.zeros(DataType.INT32, questionList.size(), answerList.get(0).length());

        for(int i=0; i<questionList.size(); i++) {
            List<Long> quesIdList = Stream.of(questionList.get(i).split("")).map(chr -> charToId.get(chr)).collect(Collectors.toList());
            x.put(new INDArrayIndex[]{NDArrayIndex.point(i)}, Nd4j.create(quesIdList));
        }

        for(int j=0; j<answerList.size(); j++) {
            List<Long> ansIdList = Stream.of(answerList.get(j).split("")).map(chr -> charToId.get(chr)).collect(Collectors.toList());
            t.put(new INDArrayIndex[]{NDArrayIndex.point(j)}, Nd4j.create(ansIdList));
        }

        // shuffle
        INDArray indices = Nd4j.arange(x.shape()[0]);
        Nd4j.shuffle(indices, 1);

        x = x.get(indices);
        t = t.get(indices);

        // 10% for validation set
        long splitAt = x.shape()[0] - x.shape()[0] / 10;
        INDArray xTrain = x.get(NDArrayIndex.interval(0, splitAt));
        INDArray xTest = x.get(NDArrayIndex.interval(splitAt, x.shape()[0]));
        INDArray tTrain = t.get(NDArrayIndex.interval(0, splitAt));
        INDArray tTest = t.get(NDArrayIndex.interval(splitAt, t.shape()[0]));

        List<INDArray> trianList = Stream.of(xTrain, tTrain).collect(Collectors.toList());
        List<INDArray> testList = Stream.of(xTest, tTest).collect(Collectors.toList());

        return Pair.of(trianList, testList);
    }

}

