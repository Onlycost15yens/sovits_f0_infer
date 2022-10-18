import io
import logging

import soundfile
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from sovits.infer_tool import Svc

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # http获得wav文件并转换
    input_wav_path = "http_temp.wav"  # io.BytesIO(wave_file.read())
    with open(input_wav_path, "wb") as f:
        f.write(wave_file.read())
    # 模型推理
    out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
    # 模型输出音频重采样到DAW所需采样率
    tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample).cpu().numpy()
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # 每个模型和config是唯一对应的
    model_name = "243_epochs.pth"
    config_name = "config.json"
    svc_model = Svc(f"pth/{model_name}", f"configs/{config_name}")
    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
