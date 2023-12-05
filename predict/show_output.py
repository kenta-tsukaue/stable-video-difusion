import pickle

# 読み込むファイル名
file_name = 'output.pkl'

# Pickleファイルから読み込み
with open(file_name, 'rb') as f:
    data = pickle.load(f)

# 画像のリストを取得
images = data.frames[0]

# GIFとして保存するファイル名
gif_filename = 'output.gif'

# GIFを作成
images[0].save(
    gif_filename,
    save_all=True,
    append_images=images[1:],
    duration=100,  # 各フレームの表示時間（ミリ秒）
    loop=0  # ループ回数（0は無限ループ）
)

print(f"GIF saved as {gif_filename}")