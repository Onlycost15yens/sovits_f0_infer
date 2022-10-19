import logging
import os
import shutil
import subprocess
import time

import librosa
import numpy as np
import torch
import torchaudio

from sovits import hubert_model
from sovits import utils
from sovits.mel_processing import spectrogram_torch
from sovits.models import SynthesizerTrn
from sovits.preprocess_wave import FeatureInput

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def cut_wav(raw_audio_path, out_audio_name, input_wav_path, cut_time):
    raw_audio, raw_sr = torchaudio.load(raw_audio_path)
    if raw_audio.shape[-1] / raw_sr > cut_time:
        subprocess.Popen(
            f"python ./sovits/slicer.py {raw_audio_path} --out_name {out_audio_name} --out {input_wav_path}  --db_thresh -30",
            shell=True).wait()
    else:
        shutil.copy(raw_audio_path, f"{input_wav_path}/{out_audio_name}-00.wav")


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def plt_pitch(input_pitch):
    input_pitch = input_pitch.astype(float)
    input_pitch[input_pitch == 1] = np.nan
    return input_pitch


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch


def del_temp_wav(path_data):
    for i in get_end_file(path_data, "wav"):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        os.remove(i)


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def get_units(in_path, hubert_soft, dev):
    source, sr = torchaudio.load(in_path)
    source = torchaudio.functional.resample(source, sr, 16000)
    if len(source.shape) == 2 and source.shape[1] >= 2:
        source = torch.mean(source, dim=0).unsqueeze(0)
    source = source.unsqueeze(0).to(dev)
    with torch.inference_mode():
        units = hubert_soft.units(source)
        return units


def transcribe(source_path, length, transform, feature_input):
    feature_pit = feature_input.compute_f0(source_path)
    feature_pit = feature_pit * 2 ** (transform / 12)
    feature_pit = resize2d_f0(feature_pit, length)
    coarse_pit = feature_input.coarse_f0(feature_pit)
    return coarse_pit


def get_unit_pitch(in_path, tran, hubert_soft, feature_input, dev):
    soft = get_units(in_path, hubert_soft, dev).squeeze(0).cpu().numpy()
    input_pitch = transcribe(in_path, soft.shape[0], tran, feature_input)
    return soft, input_pitch


class Svc(object):
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.speakers = self.hps_ms.speakers
        # 加载hubert
        self.hubert_soft = hubert_model.hubert_soft(get_end_file("./pth", "pt")[0])
        self.feature_input = FeatureInput(self.hps_ms.data.sampling_rate, self.hps_ms.data.hop_length)

        self.load_model()

    def load_model(self):
        # 获取模型配置
        self.net_g_ms = SynthesizerTrn(
            178,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers,
            **self.hps_ms.model)
        _ = utils.load_checkpoint(self.model_path, self.net_g_ms, None)
        if "half" in self.model_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)

    def calc_error(self, in_path, out_path, tran):
        input_pitch = self.feature_input.compute_f0(in_path)
        output_pitch = self.feature_input.compute_f0(out_path)
        sum_y = []
        if np.sum(input_pitch == 0) / len(input_pitch) > 0.9:
            mistake, var_take = 0, 0
        else:
            for i in range(min(len(input_pitch), len(output_pitch))):
                if input_pitch[i] > 0 and output_pitch[i] > 0:
                    sum_y.append(abs(f0_to_pitch(output_pitch[i]) - (f0_to_pitch(input_pitch[i]) + tran)))
            num_y = 0
            for x in sum_y:
                num_y += x
            len_y = len(sum_y) if len(sum_y) else 1
            mistake = round(float(num_y / len_y), 2)
            var_take = round(float(np.std(sum_y, ddof=1)), 2)
        return mistake, var_take

    def infer(self, speaker_id, tran, raw_path):
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev)
        soft, pitch = get_unit_pitch(raw_path, tran, self.hubert_soft, self.feature_input, self.dev)
        pitch = torch.LongTensor(clean_pitch(pitch)).unsqueeze(0).to(self.dev)
        if "half" in self.model_path and torch.cuda.is_available():
            stn_tst = torch.HalfTensor(soft)
        else:
            stn_tst = torch.FloatTensor(soft)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.dev)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.dev)
            audio = self.net_g_ms.infer(x_tst, x_tst_lengths, pitch, sid=sid, noise_scale=0.3, noise_scale_w=0.5,
                                        length_scale=1)[0][0, 0].data.float()
        return audio, audio.shape[-1]

    def load_audio_to_torch(self, full_path):
        audio, sampling_rate = librosa.load(full_path, sr=self.target_sample, mono=True)
        return torch.FloatTensor(audio.astype(np.float32))

    def vc(self, origin_id, target_id, raw_path):
        audio = self.load_audio_to_torch(raw_path)
        y = audio.unsqueeze(0).to(self.dev)

        spec = spectrogram_torch(y, self.hps_ms.data.filter_length,
                                 self.hps_ms.data.sampling_rate, self.hps_ms.data.hop_length,
                                 self.hps_ms.data.win_length, center=False)
        spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.dev)
        sid_src = torch.LongTensor([origin_id]).to(self.dev)

        with torch.no_grad():
            sid_tgt = torch.LongTensor([target_id]).to(self.dev)
            audio = self.net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.float()
        return audio, audio.shape[-1]

    def format_wav(self, audio_path):
        raw_audio, raw_sample_rate = torchaudio.load(audio_path)
        if len(raw_audio.shape) == 2 and raw_audio.shape[1] >= 2:
            raw_audio = torch.mean(raw_audio, dim=0).unsqueeze(0)
        tar_audio = torchaudio.functional.resample(raw_audio, raw_sample_rate, self.target_sample)
        torchaudio.save(audio_path[:-4] + ".wav", tar_audio, self.target_sample)
        return tar_audio, self.target_sample

    def flask_format_wav(self, input_wav_path, daw_sample):
        raw_audio, raw_sample_rate = torchaudio.load(input_wav_path)
        tar_audio = torchaudio.functional.resample(raw_audio, daw_sample, self.target_sample)
        if len(tar_audio.shape) == 2 and tar_audio.shape[1] >= 2:
            tar_audio = torch.mean(tar_audio, dim=0).unsqueeze(0)
        return tar_audio.cpu().numpy(), self.target_sample
