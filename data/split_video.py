import util
import ast

if __name__ == "__main__":
    with open("/Users/punyaphatsuk/MIRAI-Future-Frame-Prediction-of-Anime/data/dataset/avatar12.txt", "r") as f:
        text = f.read()
    timestamps = ast.literal_eval(text)
    output_path = "/Users/punyaphatsuk/Documents/ECE324Data/avatar2/"
    util.split_to_images(
        "/Users/punyaphatsuk/Documents/ECE324Data/avatar-the-last-airbender-book-1-water-e02-the-a.mp4", output_path, timestamps)
