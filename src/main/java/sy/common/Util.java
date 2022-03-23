package sy.common;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sy.define_exception.NumberParameterException;

import java.util.*;

/**
 * @author sy
 * @date 2022/3/10 20:26
 */
public class Util {

    private static final Logger LOGGER = LoggerFactory.getLogger(Util.class);

    /**
     * 将参数列表中重复的权重整合为1个，
     * 加上与该权重对应的梯度
     * @param params
     * @param grads
     */
    public static Pair<List<INDArray>, List<INDArray>> removeDuplicate(List<INDArray> params, List<INDArray> grads) {
        List<INDArray> paramsList = new ArrayList<>();
        params.forEach(param -> {
            INDArray paramCopy = Nd4j.zerosLike(param);
            Nd4j.copy(param, paramCopy);
            paramsList.add(paramCopy);
        });
        List<INDArray> gradsList = new ArrayList<>();
        grads.forEach(grad -> {
            INDArray gradCopy = Nd4j.zerosLike(grad);
            Nd4j.copy(grad, gradCopy);
            gradsList.add(gradCopy);
        });

        while (true) {
            boolean findFlg = false;
            int l = paramsList.size();

            for(int i=0; i < l-1; i++) {
                for(int j=i+1; j < l; j++) {
                    // 在共享权重的情况下
                    if(paramsList.get(i) == paramsList.get(j)) {
                        gradsList.get(i).addi(gradsList.get(j)); // 加上梯度
                        findFlg = true;
                        paramsList.remove(j);
                        gradsList.remove(j);
                    } else if((paramsList.get(i).rank() == 2) && (paramsList.get(j).rank() == 2) &&
                            (paramsList.get(i).transpose().equalShapes(paramsList.get(j))) &&
                            (paramsList.get(i).transpose().eq(paramsList.get(j)).all())) { // 在作为转置矩阵共享权重的情况下（weight tying）
                        paramsList.get(i).addi(gradsList.get(j).transpose());
                        findFlg = true;
                        paramsList.remove(j);
                        gradsList.remove(j);
                    }
                    if(findFlg) {
                        break;
                    }
                }
                if(findFlg) {
                    break;
                }
            }
            if(!findFlg) {
                break;
            }
        }

        Pair<List<INDArray>, List<INDArray>> pair = Pair.of(paramsList, gradsList);

        return pair;
    }

    public static void clipGrads(List<INDArray> gradsList, double maxNorm) {
        double totalNorm = 0;
        for(INDArray grad : gradsList) {
            totalNorm += Nd4j.sum(Transforms.pow(grad, 2)).getNumber().doubleValue();
        }
        totalNorm = Math.sqrt(totalNorm);
        double rate = maxNorm / (totalNorm + 1e-6);
        if(rate < 1) {
            for(INDArray grad : gradsList) {
                grad.muli(rate);
            }
        }
    }

    public static Triple<INDArray, Map<String, Long>, Map<Long, String>> preProcess(String text) {
        text = text.toLowerCase();
        text = text.replace(".", " .");
        String[] words = text.split(" ");

        Map<String, Long> wordToId = Collections.EMPTY_MAP;
        Map<Long, String> idToWord = Collections.EMPTY_MAP;
        for(String word : words) {
            if(!wordToId.containsKey(word)) {
                long newId = wordToId.size();
                wordToId.put(word, newId);
                idToWord.put(newId, word);
            }
        }
        long[] longArray = new long[words.length];
        for(int i=0; i<words.length; i++) {
            longArray[i] = wordToId.get(words[i]);
        }
        INDArray corpusINArray = Nd4j.create(longArray);

        return Triple.of(corpusINArray, wordToId, idToWord);
    }

    /**
     * calculate cosine similarity
     * @param x
     * @param y
     */
    public static double cosSimilarity(INDArray x, INDArray y) {
        return cosSimilarity(x, y, 1e-8);
    }

    public static double cosSimilarity(INDArray x, INDArray y, double eps) {
        INDArray nx = x.div(Transforms.sqrt(Nd4j.sum(Transforms.pow(x, 2))).add(eps));
        INDArray ny = y.div(Transforms.sqrt(Nd4j.sum(Transforms.pow(y, 2))).add(eps));
        return Transforms.dot(nx, ny).getNumber().doubleValue();
    }

    public static void mostSimilar(String query, Map<String, Long> wordToId, Map<Long, String> idToWord, INDArray wordMatrix) {
        mostSimilar(query, wordToId, idToWord, wordMatrix, 5);
    }

    /**
     * find similar words
     * @param query word
     * @param wordToId word-id map
     * @param idToWord id-word map
     * @param wordMatrix matrix of word vectors [A matrix of word vectors is summarized, assuming that word vectors corresponding to each row are saved]
     * @param top
     * @return
     */
    public static Pair<List<String>, List<Double>> mostSimilar(String query, Map<String, Long> wordToId, Map<Long, String> idToWord, INDArray wordMatrix, int top) {
        if(!wordToId.containsKey(query)) {
            LOGGER.info(query + " is not found!");
            return null;
        }
        LOGGER.info("query -> " + query);

        long queryId = wordToId.get(query);
        INDArray queryVec = wordMatrix.get(NDArrayIndex.point(queryId));

        int vocabSize = idToWord.size();

        INDArray similarity = Nd4j.zeros(vocabSize);
        for(int i=0; i < vocabSize; i++) {
            similarity.put(new INDArrayIndex[]{NDArrayIndex.point(i)}, cosSimilarity(wordMatrix.get(NDArrayIndex.point(i)), queryVec));
        }

        List<String> simResult = new ArrayList<>();
        List<Double> simScore = new ArrayList<>();
        int count = 0;
        INDArray similarityCopy = Nd4j.zerosLike(similarity);
        long[] sortIndex = Nd4j.sortWithIndices(similarityCopy, 1, false)[0].toLongVector();
        for(long i : sortIndex) {
            if(StringUtils.equals(idToWord.get(i), query)) {
                continue;
            }
            simResult.add(idToWord.get(i));
            simScore.add(similarity.get(NDArrayIndex.point(i)).getNumber().doubleValue());
            LOGGER.info(idToWord.get(i) + " " + similarity.get(NDArrayIndex.point(i)));
            count += 1;
            if(count >= top) {
                break;
            }
        }
        return Pair.of(simResult, simScore);
    }

    /**
     * convert one-hot
     * @param corpus list of word ids(one or two dimensions)
     * @param vocabSize number of words
     * @return two or three dimensions
     */
    public static INDArray convertOneHot(INDArray corpus, int vocabSize) {
        INDArray oneHot = null;
        long N = corpus.shape()[0];
        if(corpus.rank() == 1) {
            oneHot = Nd4j.zeros(DataType.INT32, N, vocabSize);
            for(int i=0; i<N; i++) {
                oneHot.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(corpus.get(NDArrayIndex.point(i)).getNumber().longValue())}, 1);
            }
        } else if(corpus.rank() == 2) {
            long C = corpus.shape()[1];
            oneHot = Nd4j.zeros(DataType.INT32, N, C, vocabSize);
            for(int i=0; i<N; i++) {
                INDArray wordIds = corpus.get(NDArrayIndex.point(i));
                for(int j=0; j<C; j++) {
                    oneHot.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.point(wordIds.get(NDArrayIndex.point(j)).getNumber().longValue())}, 1);
                }
            }
        }

        return oneHot;
    }

    public static INDArray createCoMatrix(INDArray corpus, int vocabSize) {
        return createCoMatrix(corpus, vocabSize, 1);
    }

    /**
     * create the co-occurrence matrix
     * @param corpus list of word ids
     * @param vocabSize number of words
     * @param windowSize window size (when the window size is 1, one word on each side is the context)
     * @return
     */
    public static INDArray createCoMatrix(INDArray corpus, int vocabSize, int windowSize) {
        long corpusSize = corpus.shape()[0];
        INDArray coMatrix = Nd4j.zeros(DataType.INT32, vocabSize, vocabSize);

        for(int i=0; i<corpusSize; i++) {
            long wordId = corpus.get(NDArrayIndex.point(i)).getNumber().longValue();
            for(int j=1; j<windowSize+1; j++) {
                int leftIdx = i - j;
                int rightIdx =i + j;
                if(leftIdx >= 0) {
                    long leftWordId = corpus.get(new INDArrayIndex[]{NDArrayIndex.point(leftIdx)}).getNumber().longValue();
                    coMatrix.get(NDArrayIndex.point(wordId), NDArrayIndex.point(leftWordId)).addi(1);
                }
                if(rightIdx < corpusSize) {
                    long rightWordId = corpus.get(new INDArrayIndex[]{NDArrayIndex.point(rightIdx)}).getNumber().longValue();
                    coMatrix.get(NDArrayIndex.point(wordId), NDArrayIndex.point(rightWordId)).addi(1);
                }
            }
        }

        return coMatrix;
    }

    public static INDArray ppmi(INDArray coMatrix) {
        return ppmi(coMatrix, false, 1e-8);
    }

    /**
     * generate ppmi (positive point mutual information)
     * @param coMatrix co-occurrence matrix
     * @param verbose output progress
     * @param eps
     * @return
     */
    public static INDArray ppmi(INDArray coMatrix, boolean verbose, double eps) {
        INDArray PPMI = Nd4j.zerosLike(coMatrix);
        INDArray N = Nd4j.sum(coMatrix);
        INDArray S = Nd4j.sum(coMatrix, 0);
        long total = coMatrix.length();
        long cnt = 0;

        for(int i=0; i<coMatrix.shape()[0]; i++) {
            for(int j=0; j<coMatrix.shape()[1]; j++) {
                double pmi = Transforms.log(coMatrix.get(NDArrayIndex.point(i), NDArrayIndex.point(j)).mul(N).div(S.get(NDArrayIndex.point(j)).mul(S.get(NDArrayIndex.point(i)))).add(eps),2).getNumber().doubleValue();
                PPMI.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(j)}, Math.max(0, pmi));
                if(verbose) {
                    cnt += 1;
                    if(cnt % (total/100 +1) == 0) {
                        LOGGER.info(100*cnt/total + " done!");
                    }
                }
            }
        }

        return PPMI;
    }

    public static <M extends BaseModel> double evalPerplexity(M model, INDArray corpus) {
        return evalPerplexity(model, corpus, 10, 50);
    }

    public static <M extends BaseModel> double evalPerplexity(M model, INDArray corpus, int batchSize, int timeSize) {
        LOGGER.info("evaluating perplexity...");
        long corpusSize = corpus.shape()[0];
        double totalLoss = 0;
        long lossCnt = 0;
        long maxIters = (corpusSize - 1) / (batchSize * timeSize);
        long jump = (corpusSize - 1) / batchSize;

        for(long iter=0; iter<maxIters; iter++) {
            INDArray xs = Nd4j.zeros(DataType.INT32, batchSize, timeSize);
            INDArray ts = Nd4j.zeros(DataType.INT32, batchSize, timeSize);
            long timeOffset = iter * timeSize;
            List<Long> offsetList = new ArrayList<>();
            for(int i=0; i<batchSize; i++) {
                offsetList.add(timeOffset + i * jump);
            }
            for(int t=0; t<timeSize; t++) {
                for(int j=0; j< offsetList.size(); j++) {
                    xs.put(new INDArrayIndex[]{NDArrayIndex.point(j), NDArrayIndex.point(t)}, corpus.get(NDArrayIndex.point((offsetList.get(j) + t) % corpusSize)));
                    ts.put(new INDArrayIndex[]{NDArrayIndex.point(j), NDArrayIndex.point(t)}, corpus.get(NDArrayIndex.point((offsetList.get(j) + t + 1) % corpusSize)));
                }
            }
            INDArray loss = null;
            try {
                loss = model.forward(xs, ts, false);
            } catch (NumberParameterException e) {
                try {
                    loss = model.forward(xs, ts);
                } catch (NumberParameterException ex) {
                    ex.printStackTrace();
                }
                e.printStackTrace();
            }
            totalLoss += loss.getNumber().doubleValue();
            LOGGER.info(iter + " : " + maxIters);
        }
        double ppl = Math.exp(totalLoss / maxIters);

        return ppl;
    }

    public static Pair<INDArray, INDArray> createContextTarget(INDArray corpus) {
        return createContextTarget(corpus, 1);
    }

    /**
     * generate context and target words
     * @param corpus list of word ids
     * @param windowSize window size (when the window size is 1, one word on each side is the context)
     * @return
     */
    public static Pair<INDArray, INDArray> createContextTarget(INDArray corpus, int windowSize) {
        INDArray target = corpus.get(new INDArrayIndex[]{NDArrayIndex.interval(windowSize, corpus.shape()[0] - windowSize)});
        List<List<Long>> context = new ArrayList<>();
        for(int idx=windowSize; idx< corpus.shape()[0] - windowSize; idx++) {
            List<Long> cs = new ArrayList<>();
            for(int t=-windowSize; t<windowSize+1; t++) {
                if(t == 0) {
                    continue;
                }
                cs.add(corpus.get(NDArrayIndex.point(idx + t)).getNumber().longValue());
            }
            context.add(cs);
        }

        INDArray contextIndArray = Nd4j.zeros(context.size(), context.get(0).size());

        for(int i=0; i<context.size(); i++) {
            contextIndArray.put(new INDArrayIndex[]{NDArrayIndex.point(i)}, Nd4j.create(context.get(i)));
        }

        return Pair.of(contextIndArray, target);
    }

    public static INDArray normalize(INDArray x) {
        if(x.rank() == 2) {
            INDArray s = Transforms.sqrt(x.mul(x).sum(1));
            x.divi(s.reshape(s.shape()[0], 1));
        } else if(x.rank() == 1) {
            INDArray s = Transforms.sqrt(x.mul(x).sum());
            x.divi(s);
        }

        return x;
    }

    public static int evalSeq2Seq(Object model, INDArray question, INDArray correct,  Map<Long, String> idToChar) {
        return evalSeq2Seq(model, question, correct, idToChar, false, false);
    }

    public static int evalSeq2Seq(Object model, INDArray question, INDArray correct,  Map<Long, String> idToChar, boolean verbos, boolean isReverse) {
        correct = Nd4j.toFlattened(correct);
        // delimiter of the beginning
        long startId = correct.get(NDArrayIndex.point(0)).getNumber().longValue();
        correct = correct.get(NDArrayIndex.interval(1, correct.shape()[0]));
//        INDArray guess = model.generate(question, startId, correct.shape()[0]);

        //convert to string
        long[] questionLong = question.toLongVector();
        String questionString = null;
        for(long q : questionLong) {
            questionString += idToChar.get(q);
        }
        long[] correctLong = correct.toLongVector();
        String correctString = null;
        for(long c : correctLong) {
            correctString += idToChar.get(c);
        }
//        long[] guessLong = guess.toLongVector();
        String guessString = null;
//        for(long g : guessLong) {
//            guessString += idToChar.get(g);
//        }

        if(verbos) {
            if(isReverse) {
                questionString = StringUtils.reverse(questionString);
            }
            Map<String, String> colorsMap = new HashMap<String, String>() {
                {
                    put("ok", "\\033[92m");
                    put("fail", "\\033[91m");
                    put("close", "\\033[0m");
                }
            };
            LOGGER.info("Q -> " + questionString);
            LOGGER.info("T -> " + correctString);

            boolean isWindows = System.getProperty("os.name").toLowerCase().contains("windows");
            if(StringUtils.equals(correctString, guessString)) {
                String mark = colorsMap.get("ok") + "☑" + colorsMap.get("close");
                if(isWindows){
                    mark = "0";
                }
                LOGGER.info(mark + " " + guessString);
            } else {
                String mark = colorsMap.get("fail") + "☒"  + colorsMap.get("close");
                if(isWindows) {
                    mark = "X";
                }
                LOGGER.info(mark + " " + guessString);
            }
            LOGGER.info("------");
        }
        return StringUtils.equals(guessString, correctString)?1:0;
    }

    public static void analogy(String a, String b, String c, Map<String, Long> wordToId, Map<Long, String> idToWord, INDArray wordMatrix) {
        analogy(a, b, c, wordToId, idToWord, wordMatrix, 5, null);
    }

    public static void analogy(String a, String b, String c, Map<String, Long> wordToId, Map<Long, String> idToWord, INDArray wordMatrix, int top, String answer) {
        if(!wordToId.containsKey(a)) {
            LOGGER.info(a + " is not found!");
            return;
        }
        if(!wordToId.containsKey(b)) {
            LOGGER.info(b + " is not found!");
            return;
        }
        if(!wordToId.containsKey(c)) {
            LOGGER.info(c + " is not found!");
            return;
        }

        LOGGER.info("\n[analogy] " + a + ":" + b + " = " + c + ":?");
        INDArray aVec = wordMatrix.get(NDArrayIndex.point(wordToId.get(a)));
        INDArray bVec = wordMatrix.get(NDArrayIndex.point(wordToId.get(b)));
        INDArray cVec = wordMatrix.get(NDArrayIndex.point(wordToId.get(c)));
        INDArray queryVec = bVec.sub(aVec).add(cVec);
        queryVec = normalize(queryVec);
        INDArray similarity = Transforms.dot(wordMatrix, queryVec);

        if(Optional.ofNullable(answer).isPresent()) {
            LOGGER.info("==> " + answer + ":" + Transforms.dot(wordMatrix.get(NDArrayIndex.point(wordToId.get(answer))), queryVec).toString());
        }

        int count = 0;
        INDArray similarityCopy = Nd4j.zerosLike(similarity);
        long[] sortIndex = Nd4j.sortWithIndices(similarityCopy, 1, false)[0].toLongVector();
        for(long i : sortIndex) {
            if(similarity.get(NDArrayIndex.point(i)).isNaN().all()) {
                continue;
            }
            if(Arrays.asList(a, b, c).contains(idToWord.get(i))) {
                continue;
            }
            LOGGER.info(idToWord.get(i) + ": " + similarity.get(NDArrayIndex.point(i)));
            count += 1;
            if(count >= top) {
                return;
            }
        }
    }

}

