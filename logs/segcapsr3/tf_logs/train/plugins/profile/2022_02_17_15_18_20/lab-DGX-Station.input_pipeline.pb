  *	�ZdcJ A2p
9Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map ߿yq*J@!�xiY�C@)AH0�J@1v�'��C@:Preprocessing2z
CIterator::Model::Prefetch::Rebatch::Prefetch::FlatMap[0]::GeneratorU����=@!��5�y6@)U����=@1��5�y6@:Preprocessing2T
Iterator::Prefetch::Generator�?�Z��9@!��hc63@)�?�Z��9@1��hc63@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2���53S@!�+�N;�L@)jj�Z8@1���-f2@:Preprocessing2�
LIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::Shuffle ��	���?!�R�.�?)��	���?1�R�.�?:Preprocessing2k
4Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat@��a��bJ@!�Z�7��C@) �����?1s'��Z�?:Preprocessing2
HIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2 �=��I��?!����A��?)�=��I��?1����A��?:Preprocessing2�
QIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2::Shuffle [� ���?!sQ�.ֻ?)[� ���?1sQ�.ֻ?:Preprocessing2F
Iterator::Model����A�?!�{��,�?)nߣ�z��?1�4n7B�?:Preprocessing2Y
"Iterator::Model::Prefetch::Rebatch�~l��?!N��^�?)O��e��?1�%��C��?:Preprocessing2P
Iterator::Model::Prefetch�tp�x�?!�{���?)�tp�x�?1�{���?:Preprocessing2z
CIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2 �+����?!���Zb(�?)�+����?1���Zb(�?:Preprocessing2I
Iterator::PrefetchLR�b��?!�J�Pܦ?)LR�b��?1�J�Pܦ?:Preprocessing2c
,Iterator::Model::Prefetch::Rebatch::PrefetchB_z�sѠ?!�z'�j4�?)B_z�sѠ?1�z'�j4�?:Preprocessing2l
5Iterator::Model::Prefetch::Rebatch::Prefetch::FlatMap5�� >@!�SL��z6@)���9]�?1�	g�|?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.