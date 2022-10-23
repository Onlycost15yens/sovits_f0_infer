import logging
import os

import soundfile

from sovits import infer_tool
from sovits.infer_tool import Svc
from wav_temp import merge

logging.getLogger('numba').setLevel(logging.WARNING)

model_name = "121_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "config.json"
svc_model = Svc(f"./pth/{model_name}", f"./configs/{config_name}")
infer_tool.mkdir(["./raw", "./pth", "./results"])

# 支持多个wav文件，放在raw文件夹下
clean_names = ["十年"]
trans = [0]  # 音高调整，支持正负（半音）
id_list = [0, 1, 2, 3, 4, 5, 6, 7]  # 每次同时合成多序号音色

input_wav_path = "./wav_temp/input"
out_wav_path = "./wav_temp/output"
cut_time = 30

infer_tool.fill_a_to_b(trans, clean_names)
infer_tool.mkdir([input_wav_path, out_wav_path])
print("mis连续超过10%时，考虑升降半音\n")
# 清除缓存文件

for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"./raw/{clean_name}.wav"
    svc_model.format_wav(raw_audio_path)
    infer_tool.del_temp_wav("./wav_temp")
    out_audio_name = model_name.split(".")[0] + f"_{clean_name}"
    infer_tool.cut_wav(raw_audio_path, out_audio_name, input_wav_path, cut_time)
    for spk_id in id_list:
        var_list = []
        mis_list = []
        count = 0
        file_list = os.listdir(input_wav_path)
        print(f"{clean_name}_{svc_model.speakers[spk_id]}")
        for file_name in file_list:
            raw_path = f"{input_wav_path}/{file_name}"
            out_path = f"{out_wav_path}/{file_name}"

            out_audio, out_sr = svc_model.infer(spk_id, tran, raw_path)
            # svc方式，仅支持模型内部音色互转，不建议使用
            # out_audio, out_sr = svc_model.vc(2, spk_id, raw_path)
            soundfile.write(out_path, out_audio.cpu().numpy(), svc_model.target_sample)

            mistake, var = svc_model.calc_error(raw_path, out_path, tran)
            mis_list.append(mistake)
            var_list.append(var)
            count += 1
            print(f"{file_name}: {round(100 * count / len(file_list), 2)}%  mis:{mistake} var:{var}")
        print(
            f"分段误差参考：0.3优秀，0.5左右合理，少量0.8-1可以接受\n若偏差过大，请调整升降半音数；多次调整均过大、说明超出歌手音域\n半音偏差：{mis_list}\n半音方差：{var_list}")
        merge.run(out_audio_name, f"_{svc_model.speakers[spk_id]}")
        # 清除缓存文件
        infer_tool.del_temp_wav("./wav_temp/output")
