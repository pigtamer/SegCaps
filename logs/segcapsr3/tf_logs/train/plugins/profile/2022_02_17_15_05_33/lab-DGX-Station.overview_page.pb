�  *	��|?���@2p
9Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map\W��2@!Q��2��?@)���>Ȇ2@1�SFV�i?@:Preprocessing2z
CIterator::Model::Prefetch::Rebatch::Prefetch::FlatMap[0]::Generator��ID�_1@!���
v=@)��ID�_1@1���
v=@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2�e�X@@!ܰ�K!�K@)��&&+@1T�|�c7@:Preprocessing2T
Iterator::Prefetch::Generator�I��Ж @!S 5�� ,@)�I��Ж @1S 5�� ,@:Preprocessing2k
4Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat </O�3@!2M.b�5@@)�2�g�?1������?:Preprocessing2�
LIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2::Shuffleb����k�?!;k��?)b����k�?1;k��?:Preprocessing2
HIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2 �d�F �?!)*�7��?) �d�F �?1)*�7��?:Preprocessing2�
QIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::ParallelMapV2::Shuffle�/�
Ҵ?!iir٦�?)�/�
Ҵ?1iir٦�?:Preprocessing2P
Iterator::Model::Prefetch*����γ?!h�<���?)*����γ?1h�<���?:Preprocessing2z
CIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::ParallelMapV2�oB@�?!�\~eY�?)�oB@�?1�\~eY�?:Preprocessing2F
Iterator::Model �M����?!�g�~�?)֩�=#�?1�uG�s��?:Preprocessing2Y
"Iterator::Model::Prefetch::RebatchuXᖏ��?!�ew�q��?)�"�~��?1���º�?:Preprocessing2I
Iterator::Prefetchq:�V�S�?!ۻ��Q�?)q:�V�S�?1ۻ��Q�?:Preprocessing2c
,Iterator::Model::Prefetch::Rebatch::Prefetch[a�^Cp�?!cNԬB�?)[a�^Cp�?1cNԬB�?:Preprocessing2l
5Iterator::Model::Prefetch::Rebatch::Prefetch::FlatMap���
c1@!�Yn?{=@)�K�Ƽ��?1D�(�є?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q��`���?"�
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Jlab-DGX-Station: Failed to load libcupti (is it installed and accessible?)