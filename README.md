# ECoG2Speech
Neural speech synthesis represents a critical ad
vancement in assistive technology for individuals with severe
motor impairments, particularly those afflicted with dysarthria
related amyotrophic lateral sclerosis (ALS). This study introduces
a novel neural decoding framework that leverages diffusion
probabilistic models to generate synthetic neural data, effec
tively addressing the persistent challenge of limited training
samples while enhancing overall decoding performance. The
proposed system employs an automatic hybrid feature extrac
tion architecture that combines convolutional neural networks
and bidirectional long short-term memory networks to capture
complex spatio-temporal neural patterns from neural signals.
Furthermore, the framework incorporates a multi-domain loss
function that integrates time-domain, frequency-domain, and
perceptual constraints to optimize reconstruction fidelity across
multiple signal characteristics. Comprehensive evaluation on an
open-access electrocorticogram (ECoG) dataset and a custom
electroencephalogram dataset (EEG2Speech) demonstrates state
of-the-art (SOTA) performance on a standardized 50-word vo
cabulary benchmark. Our framework achieves a word error
rate (WER) of 2.5 ± 0.7% on the ECoG dataset and 3.8 ±
0.8% on the EEG2Speech dataset, significantly outperforming
existing methods and establishing a strong baseline for EEG
based speech decoding applications. The achieved WER on the
ECoG dataset represents around 73% reduction compared to
prior SOTA results (WER of 9.1%), further highlighting the
substantial improvement and superior decoding accuracy of our
approach. These findings underscore the substantial potential
for developing advanced brain-computer interfaces capable of
restoring communication abilities in individuals with ALS and
other neurodegenerative conditions.
