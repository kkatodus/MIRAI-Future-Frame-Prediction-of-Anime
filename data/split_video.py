import util
import ast

if __name__ == "__main__":
    timestamp_path = "/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/SpiritedAway.txt"
    video_path = "/Volumes/OneTouch/SpiritedAway/spirited_away480.mp4"
    output_path = "/Volumes/OneTouch/SpiritedAway/"
    with open(timestamp_path, "r") as f:
        text = f.read()
    timestamps = ast.literal_eval(text)

    util.split_to_images(
        video_path, output_path, timestamps)
    """
    for i in range(10, 21):
        timestamp_path = "/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/avatar2_" + \
            str(i) + ".txt"
        video_path = "/Volumes/OneTouch/Avatar/avatar2_" + str(i) + ".mp4"
        output_path = "/Volumes/OneTouch/Avatar/avatar2_" + str(i) + "/"
        with open(timestamp_path, "r") as f:
            text = f.read()
        timestamps = ast.literal_eval(text)

        util.split_to_images(
            video_path, output_path, timestamps)
            """
