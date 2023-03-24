import util

if __name__ == "__main__":
    # note: higher threshold means that the video will be split into more parts
    threshold = 0.8
    print("Threshold: ", threshold)
    path = "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/videos/out_of_sight_1"
    list_of_frames, text = util.get_one_cut(path)
    print(len(list_of_frames))
    timestamps = util.get_cut_timestamps(
        "/Users/punyaphatsuk/Documents/ECE324Data/Out of Sight/out_of_sight.mp4", threshold)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    timestamps = util.remove_consecutive_timestamps(timestamps)
    print("timestamps: ", timestamps)
    print("len(timestamps): ", len(timestamps))
    util.save_list_to_file(
        timestamps, "/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight_timestamps.txt")
