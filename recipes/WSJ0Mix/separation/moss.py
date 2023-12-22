import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input can be a URL or a local path
input = 'https://modelscope.cn/api/v1/models/damo/speech_mossformer_separation_temporal_8k/repo?Revision=master&FilePath=examples/mix_speech1.wav'
separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer_separation_temporal_8k')
result = separation(input)
#for i, signal in enumerate(result['output_pcm_list']):
#    save_file = f'output_spk{i}.wav'
#    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
import torch
import ptflops

with torch.no_grad():
    with torch.cuda.amp.autocast():
        macs, _ = ptflops.get_model_complexity_info(
            separation.model, (8000,),  print_per_layer_stat=True, verbose=True,
        )
        print(macs)
