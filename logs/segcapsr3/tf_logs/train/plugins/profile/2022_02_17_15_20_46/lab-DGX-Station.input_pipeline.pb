  *	�$�V� A2p
9Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map ��0{يK@!TdQ	KD@)�>XƆnK@1�+6D@:Preprocessing2z
CIterator::Model::Prefetch::Rebatch::Prefetch::FlatMap[0]::Generator2!撪)?@!٤>��6@)2!撪)?@1٤>��6@:Preprocessing2T
Iterator::Prefetch::GeneratorO$�jf�;@!�wtΊ4@)O$�jf�;@1�wtΊ4@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2H4�"S@!��{L@)��
~�4@1_@s@.@:Preprocessing2k
4Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat@�{�?m�K@!�Y�F�sD@)	3m��J�?1s�b\)�?:Preprocessing2
HIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2 �@�شR�?!n�lEJ��?)�@�شR�?1n�lEJ��?:Preprocessing2�
LIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::Shuffle �H�����?!��p�0�?)�H�����?1��p�0�?:Preprocessing2�
QIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2::Shuffle �iQ��?!r�-sq�?)�iQ��?1r�-sq�?:Preprocessing2F
Iterator::Model]��e�?!���Z��?)��%ǝҹ?1�����?:Preprocessing2z
CIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2 ��P�\��?!�a�K�?)��P�\��?1�a�K�?:Preprocessing2I
Iterator::Prefetch�u7Ouȱ?!�x�v4�?)�u7Ouȱ?1�x�v4�?:Preprocessing2Y
"Iterator::Model::Prefetch::Rebatch�K�1�=�?!����b�?)��q6�?1t��G8�?:Preprocessing2P
Iterator::Model::PrefetchܼqR���?!���i� �?)ܼqR���?1���i� �?:Preprocessing2c
,Iterator::Model::Prefetch::Rebatch::Prefetch��h o��?!��Tx�7�?)��h o��?1��Tx�7�?:Preprocessing2l
5Iterator::Model::Prefetch::Rebatch::Prefetch::FlatMap&VF#�+?@!��Z�6@)�?OI?1b���w?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.