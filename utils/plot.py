from matplotlib import pyplot as plt
import numpy as np

def plot_mspec(ori_mspec, gen_mspec, title=None, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    im0 = axs[0].imshow(ori_mspec.T, aspect='auto', origin='lower', interpolation='none', cmap='magma')
    axs[0].set_title('Original Mel-spectrogram')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Mel Bin')
    fig.colorbar(im0, ax=axs[0], format='%+2.0f dB')
    
    im1 = axs[1].imshow(gen_mspec.T, aspect='auto', origin='lower', interpolation='none', cmap='magma')
    axs[1].set_title('Generated Mel-spectrogram')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Mel Bin')
    fig.colorbar(im1, ax=axs[1], format='%+2.0f dB')
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()


def plot_audio(audio, title=None, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    if title:
        plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()