�  *	���S�  A2p
9Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map �z0)�J@!˿}�^D@)�^���J@1��(GD@:Preprocessing2z
CIterator::Model::Prefetch::Rebatch::Prefetch::FlatMap[0]::Generatorni5$��<@!(�r^�6@)ni5$��<@1(�r^�6@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2�,T@!��h��N@)���q :@1���("P4@:Preprocessing2T
Iterator::Prefetch::Generator5�l�/�4@!,���/@)5�l�/�4@1,���/@:Preprocessing2k
4Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat@q�-�K@!����ɞD@)j�TQ<�?1n?3�K�?:Preprocessing2
HIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2 2*A*�?!n�@�,��?)2*A*�?1n�@�,��?:Preprocessing2�
QIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2::Shuffle ��T����?!K���q�?)��T����?1K���q�?:Preprocessing2�
LIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::Shuffle 7����?!	���j�?)7����?1	���j�?:Preprocessing2F
Iterator::Model�QH2�w�?!�IuD]��?)���5w�?1�ܲ�B,�?:Preprocessing2z
CIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2 Ts��P��?!��v����?)Ts��P��?1��v����?:Preprocessing2Y
"Iterator::Model::Prefetch::Rebatchf���8�?!�=�/E��?)�"���S�?1�^:�D�?:Preprocessing2P
Iterator::Model::Prefetch�Ɋ�� �?!?mo�?)�Ɋ�� �?1?mo�?:Preprocessing2I
Iterator::Prefetch�cx�g�?!�W��|a�?)�cx�g�?1�W��|a�?:Preprocessing2c
,Iterator::Model::Prefetch::Rebatch::Prefetch�����?!|}Y�	�?)�����?1|}Y�	�?:Preprocessing2l
5Iterator::Model::Prefetch::Rebatch::Prefetch::FlatMap�qo~�<@!'�4��6@)�J�óy?1�"�s?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q�_E:��?"�
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Jlab-DGX-Station: Failed to load libcupti (is it installed and accessible?)