import io
import logging
from pathlib import Path

import soundfile

from sovits import infer_tool
from sovits import slicer
from sovits.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)

model_name = "354_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "config.json"
svc_model = Svc(f"./pth/{model_name}", f"./configs/{config_name}")
infer_tool.mkdir(["./raw", "./pth", "./results"])

# 支持多个wav文件，放在raw文件夹下
clean_names = ["十年"]
trans = [-3]  # 音高调整，支持正负（半音）
id_list = [1]  # 每次同时合成多序号音色

cut_time = 30

infer_tool.fill_a_to_b(trans, clean_names)
print("mis连续超过10%时，考虑升降半音\n")
# 清除缓存文件

for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"./raw/{clean_name}.wav"
    svc_model.format_wav(raw_audio_path)
    audio_data, audio_sr = slicer.cut(Path(raw_audio_path).with_suffix('.wav'))

    audio = []

    for spk_id in id_list:
        var_list = []
        mis_list = []
        count = 0
        for data in audio_data:
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)

            out_audio, out_sr = svc_model.infer(spk_id, tran, raw_path)
            # svc方式，仅支持模型内部音色互转，不建议使用
            # out_audio, out_sr = svc_model.vc(2, spk_id, raw_path)
            _audio = out_audio.cpu().numpy()
            audio.extend(list(_audio))

            out_path = io.BytesIO()
            soundfile.write(out_path, _audio, svc_model.target_sample, format="wav")

            raw_path.seek(0)
            out_path.seek(0)

            mistake, var = svc_model.calc_error(raw_path, out_path, tran)
            mis_list.append(mistake)
            var_list.append(var)
            count += 1
            print(f"{clean_name}: {round(100 * count / len(audio_data), 2)}%  mis:{mistake} var:{var}")
        print(
            f"分段误差参考：0.3优秀，0.5左右合理，少量0.8-1可以接受\n若偏差过大，请调整升降半音数；多次调整均过大、说明超出歌手音域\n半音偏差：{mis_list}\n半音方差：{var_list}")
        res_path = f'./results/{clean_name}_{tran}key_{svc_model.speakers[spk_id]}.wav'
        soundfile.write(res_path, audio, svc_model.target_sample)
