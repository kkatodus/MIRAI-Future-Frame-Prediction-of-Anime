import util
import numpy as np

if __name__ == "__main__":
    path = "/Users/punyaphatsuk/Documents/ECE324Data/Out_of_Sight/autogenerated/" + \
        str(0)
    res = []
    # list_of_frames, text = util.get_one_cut(path)
    # resized_list = util.reshape(list_of_frames, 20, 128, 128)
    # array = np.array(resized_list)

    num_cuts = 41
    for i in range(num_cuts):
        path = "/Users/punyaphatsuk/Documents/ECE324Data/Out_of_Sight/autogenerated/" + \
            str(i)
        list_of_frames, text = util.get_one_cut(path)
        resized_list = util.reshape(list_of_frames, 20, 128, 128)
        # resized_list = np.expand_dims(resized_list, axis=0)
        res.append(resized_list)
    array = np.array(res)
    path = "/Users/punyaphatsuk/Documents/ECE324Data/Out_of_Sight/autogenerated/0.npy"
    print("array.shape", array.shape)
    np.save(path, array)