import logging
import os
import shutil
import subprocess

import numpy as np
import torch
import torchaudio

from sovits import hubert_model
from sovits import utils
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


class Svc(object):
    def __init__(self, model_path, config_path):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.speakers = self.hps_ms.speakers
        # 加载hubert
        self.hubert_soft = hubert_model.hubert_soft(get_end_file("./pth", "pt")[0])
        self.feature_input = FeatureInput(self.hps_ms.data.sampling_rate, self.hps_ms.data.hop_length)

        self.load_model(model_path)

    def load_model(self, model_path):
        # 获取模型配置
        self.n_g_ms = SynthesizerTrn(
            178,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers,
            **self.hps_ms.model)
        _ = utils.load_checkpoint(model_path, self.n_g_ms, None)
        _ = self.n_g_ms.eval().to(self.dev)

    def get_units(self, audio):
        audio = audio.unsqueeze(0).to(self.dev)
        with torch.inference_mode():
            units = self.hubert_soft.units(audio)
            return units

    def transcribe(self, audio, sr, length, transform):
        feature_pit = self.feature_input.compute_f0(audio, sr)
        feature_pit = feature_pit * 2 ** (transform / 12)
        feature_pit = resize2d_f0(feature_pit, length)
        coarse_pit = self.feature_input.coarse_f0(feature_pit)
        return coarse_pit

    def get_unit_pitch(self, audio, sr, tran):
        audio = torchaudio.functional.resample(audio, sr, 16000)
        if len(audio.shape) == 2 and audio.shape[1] >= 2:
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        soft = self.get_units(audio).squeeze(0).cpu().numpy()
        input_pitch = self.transcribe(audio.cpu().numpy()[0], 16000, soft.shape[0], tran)
        return soft, input_pitch

    def calc_error(self, in_path, out_path, tran):
        audio, sr = torchaudio.load(in_path)
        input_pitch = self.feature_input.compute_f0(audio.cpu().numpy()[0], sr)
        audio, sr = torchaudio.load(out_path)
        output_pitch = self.feature_input.compute_f0(audio.cpu().numpy()[0], sr)
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

    def infer(self, speaker_id, tran, model_input_audio, model_input_sr=None):
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev)
        if model_input_sr is None:
            model_input_sr = self.target_sample
        soft, pitch = self.get_unit_pitch(model_input_audio, model_input_sr, tran)
        pitch = torch.LongTensor(clean_pitch(pitch)).unsqueeze(0).to(self.dev)
        stn_tst = torch.FloatTensor(soft)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.dev)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.dev)
            audio = self.n_g_ms.infer(x_tst, x_tst_lengths, pitch, sid=sid, noise_scale=0.3, noise_scale_w=0.5,
                                      length_scale=1)[0][0, 0].data.float()
        return audio, audio.shape[-1]

    def format_wav(self, audio_path):
        raw_audio, raw_sample_rate = torchaudio.load(audio_path)
        if len(raw_audio.shape) == 2 and raw_audio.shape[1] >= 2:
            raw_audio = torch.mean(raw_audio, dim=0).unsqueeze(0)
        tar_audio = torchaudio.functional.resample(raw_audio, raw_sample_rate, self.target_sample)
        torchaudio.save(audio_path[:-4] + ".wav", tar_audio, self.target_sample)
        return tar_audio, self.target_sample
