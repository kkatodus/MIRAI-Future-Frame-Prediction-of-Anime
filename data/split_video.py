import util
import ast

if __name__ == "__main__":
    with open("/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/out_of_sight.txt", "r") as f:
        text = f.read()
    timestamps = ast.literal_eval(text)
    output_path = "/Users/punyaphatsuk/Documents/ECE324Data/Out_Of_Sight/"
    util.split_to_images(
        "/Users/punyaphatsuk/Documents/ECE324Data/Out_of_Sight/out_of_sight.mp4", output_path, timestamps)
