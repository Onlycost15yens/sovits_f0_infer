import logging
import os
import shutil
import subprocess

import demjson
import soundfile
import torch
import torchaudio

import infer_tool
from wav_temp import merge

logging.getLogger('numba').setLevel(logging.WARNING)
# 自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下
# https://github.com/bshall/hubert/releases/tag/v0.1
if not os.path.exists("./pth"):
    os.mkdir("./pth")
if not os.path.exists("./raw"):
    os.mkdir("./raw")
# pth文件夹，放置hubert、sovits模型
# 可填写音源文件列表，音源文件格式为wav，放置于raw文件夹下
clean_names = ["嘘月"]
# bgm、trans分别对应歌曲列表，若能找到相应文件、则自动合并伴奏，若找不到bgm，则输出干声（不使用bgm合成多首歌时，可只随意填写一个不存在的bgm名）
bgm_names = ["嘘月bgm"]
# 合成多少歌曲时，若半音数量不足、自动补齐相同数量（按第一首歌的半音）
trans = [-9]  # 加减半音数（可为正负）
# 每首歌同时输出的speaker_id
id_list = [0]

# 每次合成长度，建议30s内，太高了爆显存(gtx1066一次30s以内）
cut_time = 30
model_name = "476_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "sovits_pre.json"  # 模型配置（config文件夹下）

# 加载sovits模型
net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(f"pth/{model_name}", f"configs/{config_name}")
target_sample = hps_ms.data.sampling_rate
# 以下内容无需修改
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动补齐
infer_tool.fill_a_to_b(bgm_names, clean_names)
infer_tool.fill_a_to_b(trans, clean_names)

print("mis连续超过10%时，考虑升降半音\n")
# 遍历列表
for clean_name, bgm_name, tran in zip(clean_names, bgm_names, trans):
    infer_tool.format_wav(f'./raw/{clean_name}.wav', target_sample)
    for speaker_id in id_list:
        speakers = demjson.decode_file(f"configs/{config_name}")["speakers"]
        out_audio_name = model_name.split(".")[0] + f"_{clean_name}_{speakers[speaker_id]}"

        raw_audio_path = f"./raw/{clean_name}.wav"
        audio, sample_rate = torchaudio.load(raw_audio_path)
        audio_time = audio.shape[-1] / target_sample
        val_list = []
        # 清除缓存文件
        infer_tool.del_file("./wav_temp/input/")
        infer_tool.del_file("./wav_temp/output/")
        # 源音频切割方案
        if audio_time > 1.3 * int(cut_time):
            # infer_tool.cut(int(cut_time), raw_audio_path, out_audio_name, "./wav_temp/input")
            proc = subprocess.Popen(
                f"python slicer.py {raw_audio_path} --out_name {out_audio_name} --out ./wav_temp/input  --db_thresh -30",
                shell=True)
            proc.wait()
        else:
            shutil.copy(f"./raw/{clean_name}.wav", f"./wav_temp/input/{out_audio_name}-00.wav")

        count = 0
        file_list = os.listdir("./wav_temp/input")
        len_file_list = len(file_list)
        for file_name in file_list:
            source_path = "./wav_temp/input/" + file_name
            out_audio, out_sr = infer_tool.infer(source_path, speaker_id, tran, net_g_ms, hubert_soft, feature_input)
            out_path = f"./wav_temp/output/{file_name}"
            soundfile.write(out_path, out_audio, target_sample)
            mistake = infer_tool.calc_error(source_path, out_path, tran, hubert_soft, feature_input)
            val_list.append(mistake)
            count += 1
            print(f"{file_name}: {round(100 * count / len_file_list, 2)}%   mis:{mistake}%")
        print(f"\n分段误差参考：1%优秀，3%左右合理，5%-8%可以接受\n{val_list}")
        if os.path.exists(f"./wav_temp/output/temp.wav"):
            os.remove(f"./wav_temp/output/temp.wav")
        merge.run(out_audio_name, bgm_name, out_audio_name)
