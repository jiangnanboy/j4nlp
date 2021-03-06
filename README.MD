# j4nlp

j4nlp为java nlp项目，考虑到目前市面上主要是python及其生态库的天下，所以作者不打算使用开源深度学习库去实现，而是利用纯java和少量依赖包实现自然语言处理的一些常规模型，

**项目结构，后期会更新和优化**

- j4nlp
    - common
        - layers
            - [x] Affine
            - [x] Dropout
            - [x] Embedding
            - [x] MatMul
            - [x] Sigmoid
            - [x] SigmoidWithLoss
            - [x] Softmax
            - [x] SoftmaxWithLoss
        - optimizer
            - [x] AdaGrad
            - [x] Adam
            - [x] Momentum
            - [x] Nesterov
            - [x] RMSprop
            - [x] SGD
        - time_layers
            - [x] GRU
            - [x] LSTM
            - [x] RNN
            - [x] SimpleTimeAffine
            - [x] SimpleTimeSoftmaxWithLoss
            - [x] TimeAffine
            - [x] TimeBiLSTM
            - [x] TimeDropout
            - [x] TimeEmbedding
            - [x] TimeGRU
            - [x] TimeLSTM
            - [x] TimeRNN
            - [x] TimeSigmoidWithLoss
            - [x] TimeSoftmaxWithLoss
        - trainer
            - [x] RnnLmTrainer
            - [x] Trainer
        - [x] BaseModel
        - [x] Functions
        - [x] Util     
    - core
        - attention
            - [x] Attention
            - [x] AttentionDecoder
            - [x] AttentionEncoder
            - [x] AttentionSeq2Seq
            - [x] AttentionWeight
            - [x] TimeAttention
            - [x] WeightSum
        - w2v
            - negative_sampling_layer
                - [x] EmbeddingDot
                - [x] NegativeSamplingLoss
                - [x] UnigramSampler
            - [x] Cbow
            - [x] SkipGram
        - rnn
            - [x] SingleRnnLM
            - [x] MultiRnnLM
        - seq2seq
            - [x] Decoder
            - [x] Encoder
            - [x] PeekyDecoder
            - [x] PeekySeq2Seq
            - [x] Seq2Seq
    - dataset
        - [x] SeqDataset
        - [x] SpiralDataSet
    - define_exception
        - [x] NumberParameterException
    - utils