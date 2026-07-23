import glob
import os
from pydub import AudioSegment
from tqdm import tqdm


def convert_m4a_to_wav(m4a_file_path, wav_file_path=None):
    try:
        if not os.path.exists(m4a_file_path):
            raise FileNotFoundError(f"cannot find: {m4a_file_path}")
        
        if wav_file_path is None:
            wav_file_path = os.path.splitext(m4a_file_path)[0] + '.wav'
        
        audio = AudioSegment.from_file(m4a_file_path, format="m4a")
        audio.export(wav_file_path, format="wav")

        return wav_file_path
        
    except Exception as e:
        print(f"fail to convert: {str(e)}")
        return None



def adjust_volume(input_path, output_path, db_change):
    input_format = input_path.split('.')[-1]
    assert format.lower() in ['wav', 'mp3', 'm4a']
    
    if input_format.lower() == 'm4a':
        audio = AudioSegment.from_file(input_path, format="m4a")
    else:
        audio = AudioSegment.from_file(input_path, format=format)
    
    adjusted_audio = audio + db_change  
    
    if input_format.lower() == 'm4a':
        adjusted_audio.export(output_path, format="mp4", codec="aac")
    else:
        adjusted_audio.export(output_path, format=input_format)


if __name__ == "__main__":

    # wav_dir = "/media/box/BoxAI/Datasets/mead3D/wav/"   # +15dB
    # wav_dir = "/media/box/Elements/Datasets/ScanTalk_dataset/Multiface/wav/"   # +40dB
    # wav_dir = "/media/box/Elements/Datasets/ScanTalk_dataset/vocaset/wav/"   # +12dB
    wav_dir = "/media/box/Elements/Datasets/ScanTalk_dataset/Biwi_6/wav/"   # +22dB
    wav_files = glob.glob(wav_dir + "*.wav")

    print("Total number of wav files found: ", len(wav_files))

    new_wav_dir = wav_dir.replace("wav", "wav_aug")
    if not os.path.exists(new_wav_dir):
        os.makedirs(new_wav_dir)

    tqdm.write("Adjusting volume...")
    for idx, wav_file in tqdm(enumerate(wav_files), total=len(wav_files)):
        name = wav_file.split("/")[-1]
        adjust_volume(wav_file, os.path.join(new_wav_dir, name), db_change=12)

    new_wave_files = glob.glob(new_wav_dir + "*.wav")
    print("Total number of wav files processed: ", len(wav_files))


