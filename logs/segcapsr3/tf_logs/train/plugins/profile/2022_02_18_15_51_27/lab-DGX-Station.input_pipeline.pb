  *	�V���@2p
9Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map�4c�t�3@!��2N7H@)���T�3@1�]�N�H@:Preprocessing2z
CIterator::Model::Prefetch::Rebatch::Prefetch::FlatMap[0]::Generator5B?S�)@!�9U��>@)5B?S�)@1�9U��>@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2�):��;@!fn�fy�P@)s۾G��@1u����1@:Preprocessing2T
Iterator::Prefetch::Generator�������?!8���$�?)�������?18���$�?:Preprocessing2k
4Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat�rJ@L�3@!�bn��qH@)Q��r���?1E�!��R�?:Preprocessing2�
LIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::Shuffle���Bt�?!J<"4	t�?)���Bt�?1J<"4	t�?:Preprocessing2
HIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2T㥛� �?!\5��?)T㥛� �?1\5��?:Preprocessing2F
Iterator::Model2ZGUD�?!MX'�k��?)4�l\�?1b�ZB��?:Preprocessing2P
Iterator::Model::Prefetch/�혺+�?!�N���~�?)/�혺+�?1�N���~�?:Preprocessing2Y
"Iterator::Model::Prefetch::Rebatch���'�?!M�y���?) Q����?14N�f�?:Preprocessing2I
Iterator::Prefetch�=ϟ6��?!E��L��?)�=ϟ6��?1E��L��?:Preprocessing2�
QIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2::Shuffle9�~߿y�?!O��X�j�?)9�~߿y�?1O��X�j�?:Preprocessing2z
CIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2�� �K�?!?��:���?)�� �K�?1?��:���?:Preprocessing2c
,Iterator::Model::Prefetch::Rebatch::PrefetchN�����?!�&W�6�?)N�����?1�&W�6�?:Preprocessing2l
5Iterator::Model::Prefetch::Rebatch::Prefetch::FlatMapb�*�3)@!�e1��>@)5�l�/�?15�o�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.