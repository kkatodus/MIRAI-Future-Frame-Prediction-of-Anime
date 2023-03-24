import util

if __name__ == "__main__":
    for i in range(463):
        path = "/Users/punyaphatsuk/Documents/ECE324Data/avatar2/" + \
            str(i)
        list_of_frames, text = util.get_one_cut(path)
        numpy_path = path + "/" + str(i) + ".npy"
        util.save_frames_to_file(path, list_of_frames)
